//! 后台任务执行器：扫描队列、并发限流、分发到本地 / MVSEP 后端。
//!
//! 单扫描循环（保证分发无竞态），任务实际执行并行（受 `workers` 信号量限制）。
//! 取消经任务的 `CancellationToken` 下发（DELETE / 取消接口触发）。

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Semaphore};

use audio_separator_core::backend::{Input, SeparationRequest, Separator};
use audio_separator_core::config::{BackendKind, Config};
use audio_separator_core::error::Error;
use audio_separator_core::job::ProgressEvent;
use audio_separator_core::model::ModelRef;

use crate::store::{TaskStatus, TaskStore};

/// 任务执行器（AppState 持有；启动即 spawn 扫描循环）。
pub struct WorkerPool {
    store: Arc<dyn TaskStore>,
    /// 预构建后端：本地必建；MVSEP 仅在配置了 API Key 时构建。
    local: Arc<dyn Separator>,
    mvsep: Option<Arc<dyn Separator>>,
    /// 并发信号量。
    semaphore: Arc<Semaphore>,
    /// 上传根目录（输入与输出都在其下按任务 id 分目录）。
    upload_dir: std::path::PathBuf,
    /// 任务终态回调 URL（可选，M5-C）。
    webhook_url: Option<String>,
}

impl WorkerPool {
    pub fn spawn(
        store: Arc<dyn TaskStore>,
        local: Arc<dyn Separator>,
        mvsep: Option<Arc<dyn Separator>>,
        workers: usize,
        upload_dir: std::path::PathBuf,
        webhook_url: Option<String>,
    ) -> Arc<Self> {
        let pool = Arc::new(Self {
            store,
            local,
            mvsep,
            semaphore: Arc::new(Semaphore::new(workers.max(1))),
            upload_dir,
            webhook_url,
        });
        let me = Arc::clone(&pool);
        tokio::spawn(async move {
            me.scan_loop().await;
        });
        pool
    }

    /// 循环扫描排队任务并按并发上限分发。
    async fn scan_loop(self: Arc<Self>) {
        let mut ticker = tokio::time::interval(Duration::from_millis(500));
        loop {
            ticker.tick().await;
            let tasks = self.store.list();
            let mut queued: Vec<_> = tasks
                .into_iter()
                .filter(|t| t.status == TaskStatus::Queued)
                .collect();
            queued.sort_by_key(|t| t.created_at);
            for task in queued {
                let Ok(permit) = self.semaphore.clone().try_acquire_owned() else {
                    break; // 并发已满，余下等下一轮
                };
                let me = Arc::clone(&self);
                let id = task.id.clone();
                tokio::spawn(async move {
                    let _guard = permit;
                    if let Err(e) = me.run_one(&id).await {
                        eprintln!("[worker] 任务 {id} 异常: {e}");
                    }
                });
            }
        }
    }

    /// 执行单个任务：状态推进、进度转发、后端分发。
    async fn run_one(&self, id: &str) -> Result<(), String> {
        let task = self
            .store
            .get(id)
            .ok_or_else(|| format!("任务不存在: {id}"))?;
        let cancel = task.cancel.clone().unwrap_or_default();
        let (tx, mut rx) = mpsc::channel::<ProgressEvent>(32);

        // 标记 running
        self.store
            .update(id, Box::new(|t| {
                t.status = TaskStatus::Running;
                t.message = "排队中".to_string();
            }))
            .map_err(|e| e.to_string())?;

        // 分发进度事件 → 任务表
        {
            let store = Arc::clone(&self.store);
            let tid = id.to_string();
            let _fwd = tokio::spawn(async move {
                while let Some(ev) = rx.recv().await {
                    let store = Arc::clone(&store);
                    let tid = tid.clone();
                    let _ = store.update(&tid, Box::new(move |t| {
                        // 终态保护：Done/Failed/Cancelled 后不再回退状态（避免进度事件竞态覆盖）
                        if !t.status.is_terminal() {
                            t.status = TaskStatus::Running;
                        }
                        apply_progress(t, &ev);
                    }));
                }
            });
        }

        // 选择后端
        let sep: Arc<dyn Separator> = match task.backend {
            BackendKind::Local => Arc::clone(&self.local),
            BackendKind::Mvsep => self
                .mvsep
                .clone()
                .ok_or_else(|| "服务端未配置 MVSEP API Key（config.mvsep.api_key）".to_string())?,
        };

        // 输入路径：上传落盘位置（M4 上传即落盘，不保留内存态）
        let input_path = self.upload_dir.join(id).join(&task.input_name);
        let req = SeparationRequest {
            input: Input::Path(input_path),
            model: ModelRef::Name(task.model.clone()),
            output_format: task.format,
            output_dir: task.out_dir.clone(),
            select_stems: if task.select_stems.is_empty() {
                None
            } else {
                Some(task.select_stems.clone())
            },
        };

        let result = match sep.separate(req, Some(tx), Some(cancel)).await {
            Ok(r) => r,
            Err(Error::Cancelled) => {
                self.store.update(id, Box::new(|t| {
                    t.status = TaskStatus::Cancelled;
                    t.message = "已取消".to_string();
                }))?;
                self.fire_webhook(id);
                return Ok(());
            }
            Err(e) => {
                self.store.update(id, Box::new(move |t| {
                    t.status = TaskStatus::Failed;
                    t.message = e.to_string();
                    t.error = Some(e.to_string());
                }))?;
                self.fire_webhook(id);
                return Ok(());
            }
        };

        // 收集输出文件
        let mut files = Vec::new();
        for (name, path) in &result.stems {
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            files.push(crate::store::TaskFile {
                name: name.clone(),
                path: path.clone(),
                size,
            });
        }
        self.store.update(id, Box::new(move |t| {
            t.status = TaskStatus::Done;
            t.message = format!("完成（{:.1}s）", result.elapsed.as_secs_f64());
            t.progress = 1.0;
            t.files = files;
        }))?;
        self.fire_webhook(id);
        Ok(())
    }

    /// 任务进入终态后向配置的 webhook URL 推送 JSON（fire-and-forget，失败仅打印）。
    fn fire_webhook(&self, id: &str) {
        let Some(url) = &self.webhook_url else { return };
        let Some(task) = self.store.get(id) else { return };
        let stems: Vec<_> = task
            .files
            .iter()
            .map(|f| serde_json::json!({ "name": f.name, "size": f.size }))
            .collect();
        let body = serde_json::json!({
            "task_id": task.id,
            "status": match task.status {
                TaskStatus::Done => "done",
                TaskStatus::Failed => "failed",
                TaskStatus::Cancelled => "cancelled",
                _ => "unknown",
            },
            "message": task.message,
            "error": task.error,
            "stems": stems,
        });
        let client = reqwest::Client::new();
        let url = url.clone();
        tokio::spawn(async move {
            match client.post(&url).json(&body).send().await {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        eprintln!("[webhook] 回调 {url} 返回 {}", resp.status());
                    }
                }
                Err(e) => eprintln!("[webhook] 回调 {url} 失败: {e}"),
            }
        });
    }
}

/// 把核心库进度事件映射进任务表（阶段 → 估算进度）。
fn apply_progress(t: &mut crate::store::TaskRecord, ev: &ProgressEvent) {
    match ev {
        ProgressEvent::Stage(s) => {
            t.message = stage_label(s);
            t.progress = stage_progress(s);
        }
        ProgressEvent::Download { .. } => {
            t.progress = t.progress.max(0.2);
            t.message = "下载模型文件…".to_string();
        }
        ProgressEvent::Process { percent, message } => {
            t.progress = *percent;
            if let Some(m) = message {
                t.message = m.clone();
            }
        }
        ProgressEvent::Writing { stem, percent } => {
            t.progress = 0.85 + 0.15 * percent;
            t.message = format!("写入分轨 {stem}…");
        }
        ProgressEvent::Finished => {
            t.progress = 1.0;
            t.message = "完成".to_string();
        }
    }
}

fn stage_label(s: &str) -> String {
    match s {
        "resolve_model" => "解析模型".to_string(),
        "download" => "下载模型文件".to_string(),
        "convert" => "转换权重格式".to_string(),
        "load_model" => "加载模型".to_string(),
        "read_audio" => "读取音频".to_string(),
        "infer" => "推理中".to_string(),
        other => other.to_string(),
    }
}

fn stage_progress(s: &str) -> f32 {
    match s {
        "resolve_model" => 0.05,
        "download" => 0.20,
        "convert" => 0.28,
        "load_model" => 0.38,
        "read_audio" => 0.48,
        "infer" => 0.75,
        _ => 0.5,
    }
}

/// 构建已配置的分离器后端集合（启动时调用一次）。
///
/// - local：始终构建（模型清单经 `config.models` 懒加载）。
/// - mvsep：仅当配置了 API Key；manifest 清单用于按名映射 sep_type。
pub fn build_separators(cfg: &Config) -> Result<(Arc<dyn Separator>, Option<Arc<dyn Separator>>), String> {
    use audio_separator_core::backend::local::LocalSeparator;
    use audio_separator_core::backend::mvsep::MvsepSeparator;
    use audio_separator_core::backend::local::model_manage::ModelManager;

    let local = LocalSeparator::new(cfg).map_err(|e| e.to_string())?;
    let mvsep = if cfg.mvsep.api_key.is_some() {
        let manifest = ModelManager::load(&cfg.models, cfg.network.proxy.as_deref())
            .map_err(|e| e.to_string())?;
        Some(MvsepSeparator::new(cfg, Default::default(), None, Some(manifest.list().clone()))
            .map_err(|e| e.to_string())?)
    } else {
        None
    };
    Ok((Arc::new(local), mvsep.map(|s| Arc::new(s) as Arc<dyn Separator>)))
}
