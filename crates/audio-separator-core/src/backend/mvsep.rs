//! MVSEP 云后端客户端（M3）。
//!
//! 平台：`POST /api/separation/create`（multipart：api_token + audiofile|url + sep_type + add_opt* + output_format + webhook_url）
//! → `GET /api/separation/get?hash=` 轮询（waiting → processing → done）→ 下载 `data.files`。
//! 支持取消 / 删除、算法目录拉取（`/api/app/algorithms`）、用户信息（配额提示）。
//! 联网走用户 HTTP 代理（显式配置 > ALL_PROXY/HTTPS_PROXY/HTTP_PROXY），非 Premium 默认单并发。

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::backend::local::model_manage::resolve_proxy;
use crate::backend::{Input, SeparationRequest, SeparationResult, Separator};
use crate::config::{BackendKind, Config, MvsepConfig};
use crate::error::{Error, Result};
use crate::job::ProgressEvent;
use crate::model::{ModelList, ModelRef, OutputFormat};

const USER_AGENT: &str = "audio-separator-rs/0.1 (mvsep)";

// ---------------- API 响应类型 ----------------

#[derive(Deserialize)]
struct ApiEnvelope<T> {
    success: bool,
    data: Option<T>,
}

#[derive(Deserialize)]
struct CreateData {
    #[allow(dead_code)] // link 展示用（job-status 可打印）
    link: Option<String>,
    hash: Option<String>,
    message: Option<String>,
}

/// 输出文件（`data.files` 项）：`type` 为分轨名（如 Vocals / Other / Bass…）。
#[derive(Debug, Clone, Deserialize)]
pub struct MvsepFile {
    #[serde(rename = "type")]
    pub stem: String,
    pub url: String,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub bytes: Option<u64>,
    #[serde(default)]
    pub download: Option<String>,
}

#[derive(Deserialize, Default)]
struct GetData {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    queue_count: Option<u64>,
    #[serde(default)]
    current_order: Option<u64>,
    #[serde(default)]
    algorithm: Option<String>,
    #[serde(default)]
    files: Vec<MvsepFile>,
}

#[derive(Deserialize)]
struct GetResponse {
    success: bool,
    status: Option<String>,
    data: Option<GetData>,
}

/// 算法目录项（`/api/app/algorithms`）。
#[derive(Debug, Clone, Deserialize)]
pub struct MvsepAlgorithm {
    #[serde(rename = "render_id")]
    pub render_id: u64,
    pub name: String,
    #[serde(rename = "algorithm_group", default)]
    pub group: Option<AlgorithmGroup>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AlgorithmGroup {
    #[serde(default)]
    pub name: Option<String>,
}

/// 任务状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvsepStatus {
    Waiting,
    Processing,
    Distributing,
    Merging,
    Done,
    Failed,
    NotFound,
}

/// 一次查询到的任务状态快照。
#[derive(Debug, Clone)]
pub struct SeparationState {
    pub status: MvsepStatus,
    pub files: Vec<MvsepFile>,
    pub message: Option<String>,
    pub queue_count: Option<u64>,
    pub current_order: Option<u64>,
    pub algorithm: Option<String>,
}

// ---------------- 客户端 ----------------

/// MVSEP REST 客户端（create / get / cancel / delete / algorithms / user / download）。
pub struct MvsepClient {
    http: reqwest::Client,
    base: String,
    /// API Key；`algorithms`（公开目录）不需要，create/cancel/delete/user 需要。
    api_token: Option<String>,
    poll_interval: Duration,
    poll_timeout: Duration,
}

impl MvsepClient {
    /// 构建客户端；API Key 取 `cfg.api_key`，缺省回退环境变量 `ASEP_MVSEP_API_KEY`。
    pub fn new(cfg: &MvsepConfig, proxy: Option<&str>) -> Result<Self> {
        let api_token = cfg
            .api_key
            .clone()
            .or_else(|| std::env::var("ASEP_MVSEP_API_KEY").ok());
        let mut builder = reqwest::Client::builder().user_agent(USER_AGENT);
        if let Some(p) = resolve_proxy(proxy) {
            let proxy = reqwest::Proxy::all(&p)
                .map_err(|e| Error::Config(format!("invalid proxy config {p}: {e}")))?;
            builder = builder.proxy(proxy);
        }
        let http = builder
            .build()
            .map_err(|e| Error::Network(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            http,
            base: cfg.region.base_url().to_string(),
            api_token,
            poll_interval: Duration::from_secs(cfg.poll_interval_secs.max(1)),
            poll_timeout: Duration::from_secs(cfg.poll_timeout_secs),
        })
    }

    /// 返回 API Key（create/cancel/delete/user 需要）。
    fn token(&self) -> Result<&str> {
        self.api_token
            .as_deref()
            .ok_or_else(|| {
                Error::Config(
                    "MVSEP backend requires an API key: --api-key argument or ASEP_MVSEP_API_KEY env var"
                        .to_string(),
                )
            })
    }

    /// 创建分离任务（multipart 表单），返回任务 hash。
    pub async fn create(&self, form: Form) -> Result<String> {
        let url = format!("{}/separation/create", self.base);
        let resp = self
            .http
            .post(url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| Error::Network(format!("MVSEP create request failed: {e}")))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Network(format!("MVSEP create response read failed: {e}")))?;
        if !status.is_success() {
            return Err(match status.as_u16() {
                401 => Error::Config("invalid MVSEP API key (HTTP 401)".to_string()),
                400 => Error::Backend(format!("MVSEP bad request (HTTP 400): {body}")),
                _ => Error::Network(format!("MVSEP create HTTP {status}: {body}")),
            });
        }
        let env: ApiEnvelope<CreateData> = serde_json::from_str(&body).map_err(Error::Json)?;
        match env.data {
            Some(d) if env.success => d
                .hash
                .ok_or_else(|| Error::Backend("MVSEP create response missing hash".to_string())),
            Some(d) => Err(Error::Backend(format!(
                "MVSEP create failed: {}",
                d.message.unwrap_or_default()
            ))),
            None => Err(Error::Backend("MVSEP create response missing data".to_string())),
        }
    }

    /// 查询任务状态。
    pub async fn get(&self, hash: &str) -> Result<SeparationState> {
        let url = format!("{}/separation/get", self.base);
        let resp = self
            .http
            .get(url)
            .query(&[("hash", hash)])
            .send()
            .await
            .map_err(|e| Error::Network(format!("MVSEP get request failed: {e}")))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Network(format!("MVSEP get response read failed: {e}")))?;
        if !status.is_success() {
            return Err(match status.as_u16() {
                401 => Error::Config("invalid MVSEP API key (HTTP 401)".to_string()),
                _ => Error::Network(format!("MVSEP get HTTP {status}: {body}")),
            });
        }
        let env: GetResponse = serde_json::from_str(&body).map_err(Error::Json)?;
        if !env.success {
            return Err(Error::Backend(format!(
                "MVSEP query failed: {}",
                env.data
                    .as_ref()
                    .and_then(|d| d.message.clone())
                    .unwrap_or_default()
            )));
        }
        let data = env.data.unwrap_or_default();
        let status = match env.status.as_deref() {
            Some("waiting") => MvsepStatus::Waiting,
            Some("processing") => MvsepStatus::Processing,
            Some("distributing") => MvsepStatus::Distributing,
            Some("merging") => MvsepStatus::Merging,
            Some("done") => MvsepStatus::Done,
            Some("failed") => MvsepStatus::Failed,
            _ => MvsepStatus::NotFound,
        };
        Ok(SeparationState {
            status,
            files: data.files,
            message: data.message,
            queue_count: data.queue_count,
            current_order: data.current_order,
            algorithm: data.algorithm,
        })
    }

    /// 取消分离任务（尚未开始处理时退回积分）。
    pub async fn cancel(&self, hash: &str) -> Result<()> {
        let url = format!("{}/separation/cancel", self.base);
        let resp = self
            .http
            .post(url)
            .form(&[("api_token", self.token()?), ("hash", hash)])
            .send()
            .await
            .map_err(|e| Error::Network(format!("MVSEP cancel request failed: {e}")))?;
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Network(format!("MVSEP cancel response read failed: {e}")))?;
        let env: ApiEnvelope<CreateData> = serde_json::from_str(&body).map_err(Error::Json)?;
        if !env.success {
            return Err(Error::Backend(format!(
                "MVSEP cancel failed: {}",
                env.data
                    .and_then(|d| d.message)
                    .unwrap_or_else(|| body)
            )));
        }
        Ok(())
    }

    /// 删除分离任务及其输出文件。
    pub async fn delete(&self, hash: &str) -> Result<()> {
        let url = format!("{}/separation/delete", self.base);
        let resp = self
            .http
            .post(url)
            .form(&[("api_token", self.token()?), ("hash", hash)])
            .send()
            .await
            .map_err(|e| Error::Network(format!("MVSEP delete request failed: {e}")))?;
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Network(format!("MVSEP delete response read failed: {e}")))?;
        let env: ApiEnvelope<CreateData> = serde_json::from_str(&body).map_err(Error::Json)?;
        if !env.success {
            return Err(Error::Backend(format!(
                "MVSEP delete failed: {}",
                env.data
                    .and_then(|d| d.message)
                    .unwrap_or_else(|| body)
            )));
        }
        Ok(())
    }

    /// 拉取分离类型目录（单文件输入模型）。
    pub async fn algorithms(&self) -> Result<Vec<MvsepAlgorithm>> {
        let url = format!("{}/app/algorithms", self.base);
        let resp = self
            .http
            .get(url)
            .query(&[("scopes", "single_upload")])
            .send()
            .await
            .map_err(|e| Error::Network(format!("MVSEP algorithms request failed: {e}")))?;
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Network(format!("MVSEP algorithms response read failed: {e}")))?;
        serde_json::from_str(&body).map_err(Error::Json)
    }

    /// 获取用户信息（配额 / 余额提示用）。
    pub async fn user(&self) -> Result<serde_json::Value> {
        let url = format!("{}/app/user", self.base);
        let resp = self
            .http
            .get(url)
            .query(&[("api_token", self.token()?)])
            .send()
            .await
            .map_err(|e| Error::Network(format!("MVSEP user request failed: {e}")))?;
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Network(format!("MVSEP user response read failed: {e}")))?;
        serde_json::from_str(&body).map_err(Error::Json)
    }

    /// 下载输出文件到目标路径（进度 + 取消）。
    pub async fn download(
        &self,
        url: &str,
        dest: &Path,
        stem: &str,
        progress: Option<&mpsc::Sender<ProgressEvent>>,
        cancel: Option<&CancellationToken>,
    ) -> Result<()> {
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = dest.with_extension("part");
        let mut resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Network(format!("stem download failed {url}: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Error::Network(format!(
                "stem download failed {url}: HTTP {status}"
            )));
        }
        let total = resp.content_length();
        let mut file = tokio::fs::File::create(&tmp).await?;
        let mut downloaded: u64 = 0;
        loop {
            if let Some(c) = cancel {
                if c.is_cancelled() {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(Error::Cancelled);
                }
            }
            let chunk = match resp.chunk().await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(e) => {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(Error::Network(format!("stem download failed {url}: {e}")));
                }
            };
            use tokio::io::AsyncWriteExt;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            if let Some(pr) = progress {
                let percent = total
                    .filter(|t| *t > 0)
                    .map(|t| downloaded as f32 / t as f32)
                    .unwrap_or(0.0)
                    .min(1.0);
                let _ = pr.try_send(ProgressEvent::Writing {
                    stem: stem.to_string(),
                    percent,
                });
            }
        }
        use tokio::io::AsyncWriteExt;
        file.flush().await?;
        drop(file);
        tokio::fs::rename(&tmp, dest).await?;
        Ok(())
    }
}

// ---------------- Separator 实现 ----------------

/// MVSEP 云后端：按 [`Separator`] trait 执行 create → poll → download。
pub struct MvsepSeparator {
    client: MvsepClient,
    semaphore: Arc<Semaphore>,
    add_opts: BTreeMap<String, String>,
    webhook: Option<String>,
    manifest: Option<ModelList>,
    algorithms: RwLock<Option<Vec<MvsepAlgorithm>>>,
}

impl MvsepSeparator {
    /// 构建云后端。`manifest` 为可选本地模型清单（按名引用时查 mvsep 映射）；
    /// `add_opts` 为 CLI 显式附加选项（`add_opt1/2/3`），覆盖条目映射同名键。
    pub fn new(
        cfg: &Config,
        add_opts: BTreeMap<String, String>,
        webhook: Option<String>,
        manifest: Option<ModelList>,
    ) -> Result<Self> {
        let client = MvsepClient::new(&cfg.mvsep, cfg.network.proxy.as_deref())?;
        let semaphore = Arc::new(Semaphore::new(cfg.mvsep.concurrency.max(1)));
        let webhook = webhook.or_else(|| cfg.mvsep.webhook_url.clone());
        Ok(Self {
            client,
            semaphore,
            add_opts,
            webhook,
            manifest,
            algorithms: RwLock::new(None),
        })
    }

    /// 解析模型引用 → (sep_type, 附加选项)。
    /// 优先级：manifest 条目 mvsep 映射 > 纯数字 sep_type > 算法目录名字匹配。
    async fn resolve_model(&self, model: &ModelRef) -> Result<(u64, BTreeMap<String, String>)> {
        let name = match model {
            ModelRef::Name(n) => n,
            ModelRef::Url { .. } | ModelRef::LocalPath { .. } => {
                return Err(Error::Config(
                    "for the MVSEP backend, specify a model by name (manifest mvsep mapping / platform algorithm name) or a numeric sep_type; URL / local path forms are only supported by the local backend"
                        .to_string(),
                ));
            }
        };
        if let Some(ml) = &self.manifest {
            if let Some(entry) = ml.get(name) {
                if let Some(mv) = &entry.mvsep {
                    let mut opts = serde_json::from_value(mv.add_opts.clone())
                        .unwrap_or_else(|_| BTreeMap::new());
                    // CLI 显式选项覆盖条目映射
                    for (k, v) in &self.add_opts {
                        opts.insert(k.clone(), v.clone());
                    }
                    return Ok((mv.sep_type, opts));
                }
            }
        }
        if let Ok(n) = name.trim().parse::<u64>() {
            return Ok((n, self.add_opts.clone()));
        }
        let algos = self.algorithms_cached().await?;
        let hit = algos
            .iter()
            .find(|a| a.name.trim().eq_ignore_ascii_case(name))
            .or_else(|| {
                let lower = name.to_lowercase();
                algos.iter().find(|a| {
                    let n = a.name.to_lowercase();
                    n.contains(&lower) || lower.contains(&n)
                })
            });
        match hit {
            Some(a) => Ok((a.render_id, self.add_opts.clone())),
            None => {
                let samples: Vec<String> =
                    algos.iter().take(12).map(|a| a.name.clone()).collect();
                Err(Error::Model(format!(
                    "MVSEP model \"{name}\" not found; the platform algorithm catalog has {} entries - run `asep models --backend mvsep` to list them. Similar entries: {}",
                    algos.len(),
                    samples.join(" / ")
                )))
            }
        }
    }

    /// 算法目录缓存（进程生命周期内一次拉取）。
    async fn algorithms_cached(&self) -> Result<Vec<MvsepAlgorithm>> {
        if let Some(c) = self.algorithms.read().await.as_ref() {
            return Ok(c.clone());
        }
        let list = self.client.algorithms().await?;
        *self.algorithms.write().await = Some(list.clone());
        Ok(list)
    }

    /// 构建 create 表单：token + sep_type + output_format + add_opt* + webhook + 输入。
    async fn build_form(
        &self,
        input: &Input,
        sep_type: u64,
        add_opts: &BTreeMap<String, String>,
        output_format: OutputFormat,
    ) -> Result<Form> {
        let mut form = Form::new()
            .text("api_token", self.client.token()?.to_string())
            .text("sep_type", sep_type.to_string())
            .text("output_format", output_format.as_mvsep_code().to_string());
        for (k, v) in add_opts {
            form = form.text(k.clone(), v.clone());
        }
        if let Some(w) = &self.webhook {
            form = form.text("webhook_url", w.clone());
        }
        match input {
            Input::Path(p) => {
                let part = Part::file(p)
                    .await
                    .map_err(|e| Error::Network(format!("failed to read upload file {}: {e}", p.display())))?;
                form = form.part("audiofile", part);
            }
            Input::Url(u) => {
                // 远程输入：透传 url 让 MVSEP 自行拉取（remote_type 默认 direct）
                form = form.text("url", u.clone());
            }
            Input::Bytes(b) => {
                form = form.part("audiofile", Part::bytes(b.clone()).file_name("upload.wav"));
            }
        }
        Ok(form)
    }
}

#[async_trait]
impl Separator for MvsepSeparator {
    async fn separate(
        &self,
        req: SeparationRequest,
        progress: Option<mpsc::Sender<ProgressEvent>>,
        cancel: Option<CancellationToken>,
    ) -> Result<SeparationResult> {
        let started = Instant::now();
        // 非 Premium 单并发：信号量按配置限流（默认 1）。
        let permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| Error::Other("MVSEP concurrency semaphore is closed".to_string()))?;

        emit(&progress, ProgressEvent::Stage("resolve_model".to_string()));
        let (sep_type, add_opts) = self.resolve_model(&req.model).await?;

        emit(&progress, ProgressEvent::Stage("create".to_string()));
        let form = self
            .build_form(&req.input, sep_type, &add_opts, req.output_format)
            .await?;
        let hash = self.client.create(form).await?;
        emit(
            &progress,
            ProgressEvent::Stage(format!("polling {hash}")),
        );

        // 轮询：waiting/processing/distributing/merging → done/failed/not_found
        let deadline = Instant::now() + self.client.poll_timeout;
        let files = loop {
            if let Some(c) = &cancel {
                if c.is_cancelled() {
                    let _ = self.client.cancel(&hash).await;
                    return Err(Error::Cancelled);
                }
            }
            let st = self.client.get(&hash).await?;
            match st.status {
                MvsepStatus::Done => break st.files,
                MvsepStatus::Failed => {
                    return Err(Error::Backend(format!(
                        "MVSEP task failed: {}",
                        st.message.unwrap_or_else(|| "unknown reason".to_string())
                    )));
                }
                MvsepStatus::NotFound => {
                    return Err(Error::Backend(format!("MVSEP task not found or expired: {hash}")));
                }
                MvsepStatus::Waiting | MvsepStatus::Processing | MvsepStatus::Distributing
                | MvsepStatus::Merging => {
                    let msg = match (st.queue_count, st.current_order) {
                        (Some(q), Some(o)) => {
                            format!("in queue at position {o} (of {q} pending)")
                        }
                        (Some(q), None) => format!("queued (ahead of {q} tasks)"),
                        _ => format!("{:?}", st.status),
                    };
                    let pct = match st.status {
                        MvsepStatus::Waiting => 0.05,
                        MvsepStatus::Distributing => 0.3,
                        _ => 0.5,
                    };
                    emit(
                        &progress,
                        ProgressEvent::Process {
                            percent: pct,
                            message: Some(msg),
                        },
                    );
                    tokio::time::sleep(self.client.poll_interval).await;
                }
            }
            if Instant::now() > deadline {
                let _ = self.client.cancel(&hash).await;
                return Err(Error::Quota(format!(
                    "MVSEP poll timed out ({:.0}s); task {hash} cancelled",
                    self.client.poll_timeout.as_secs_f64()
                )));
            }
        };
        drop(permit);

        // 下载分轨到输出目录：文件名 {stem}.{ext}，stem = 平台 type 小写。
        std::fs::create_dir_all(&req.output_dir)?;
        let ext = req.output_format.as_extension();
        let mut stems = BTreeMap::new();
        for file in &files {
            let stem = file.stem.to_lowercase();
            if let Some(sel) = &req.select_stems {
                if !sel.iter().any(|s| s.eq_ignore_ascii_case(&stem)) {
                    continue;
                }
            }
            let dest = req.output_dir.join(format!("{stem}.{ext}"));
            emit(&progress, ProgressEvent::Stage(format!("download {stem}")));
            self.client
                .download(&file.url, &dest, &stem, progress.as_ref(), cancel.as_ref())
                .await?;
            stems.insert(stem, dest);
        }
        emit(&progress, ProgressEvent::Finished);

        Ok(SeparationResult {
            stems,
            backend: BackendKind::Mvsep,
            elapsed: started.elapsed(),
        })
    }
}

fn emit(sender: &Option<mpsc::Sender<ProgressEvent>>, ev: ProgressEvent) {
    if let Some(s) = sender {
        let _ = s.try_send(ev);
    }
}
