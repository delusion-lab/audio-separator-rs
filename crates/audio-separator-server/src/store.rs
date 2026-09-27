//! 任务存储（D3 决策：内存任务表 + 预留持久化 trait）。
//!
//! [`TaskStore`] 是唯一抽象：M4 用 [`InMemoryTaskStore`]；M5 可新增文件/数据库实现
//! （如 `FileTaskStore`，按 JSON 行或 SQLite 落盘），无需改动 API 层与 worker。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::Serialize;
use tokio_util::sync::CancellationToken;

use audio_separator_core::config::BackendKind;
use audio_separator_core::model::OutputFormat;

/// 任务状态机：queued → running → done | failed；queued/running 可被取消为 cancelled。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl TaskStatus {
    /// 是否处于终态。
    pub fn is_terminal(self) -> bool {
        matches!(self, TaskStatus::Done | TaskStatus::Failed | TaskStatus::Cancelled)
    }
}

/// 已完成任务的输出文件条目。
#[derive(Debug, Clone, Serialize)]
pub struct TaskFile {
    /// 分轨名（如 `vocals`）。
    pub name: String,
    /// 输出文件绝对路径。
    pub path: PathBuf,
    /// 文件大小（字节）。
    pub size: u64,
}

/// 一条分离任务记录。
#[derive(Debug, Clone, Serialize)]
pub struct TaskRecord {
    /// 任务 ID（API 标识）。
    pub id: String,
    /// 执行后端。
    pub backend: BackendKind,
    /// 模型引用原文（展示用）。
    pub model: String,
    /// 当前状态。
    pub status: TaskStatus,
    /// 进度 0.0–1.0（进行中按阶段估算）。
    pub progress: f32,
    /// 阶段 / 错误信息。
    pub message: String,
    /// 创建时间（Unix 秒）。
    pub created_at: u64,
    /// 最近更新（Unix 秒）。
    pub updated_at: u64,
    /// 输入文件名（展示用）。
    pub input_name: String,
    /// 输出目录（任务专属）。
    pub out_dir: PathBuf,
    /// 输出格式。
    pub format: OutputFormat,
    /// 只输出这些分轨；空 = 模型全部分轨。
    pub select_stems: Vec<String>,
    /// 输出文件（done 后填充）。
    pub files: Vec<TaskFile>,
    /// 失败原因。
    pub error: Option<String>,
    /// 取消令牌（内存态，不参与序列化；DELETE / 取消时触发）。
    #[serde(skip)]
    pub cancel: Option<CancellationToken>,
}

/// 任务存储抽象（预留持久化实现：M5 新增 `FileTaskStore`）。
pub trait TaskStore: Send + Sync {
    /// 插入任务（幂等：同 id 已存在则覆盖）。
    fn insert(&self, task: TaskRecord) -> Result<(), String>;
    /// 按 id 取任务。
    fn get(&self, id: &str) -> Option<TaskRecord>;
    /// 按 id 更新（闭包内修改；任务不存在时报错）。
    /// 使用 `Box<dyn FnOnce>` 保持 trait dyn-compatible（可替换持久化实现）。
    fn update(&self, id: &str, f: Box<dyn FnOnce(&mut TaskRecord) + Send>) -> Result<(), String>;
    /// 列出全部任务（时间升序）。
    fn list(&self) -> Vec<TaskRecord>;
    /// 删除任务（不清理磁盘产物；清理由调用方负责）。
    fn remove(&self, id: &str) -> Result<(), String>;
}

/// 内存任务表：全局互斥 + HashMap。
///
/// 任务量级为单机并发（workers 默认 1），内存表满足需求；持久化见 trait 注释。
pub struct InMemoryTaskStore {
    inner: Mutex<HashMap<String, TaskRecord>>,
}

impl InMemoryTaskStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryTaskStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskStore for InMemoryTaskStore {
    fn insert(&self, task: TaskRecord) -> Result<(), String> {
        self.inner.lock().unwrap().insert(task.id.clone(), task);
        Ok(())
    }

    fn get(&self, id: &str) -> Option<TaskRecord> {
        self.inner.lock().unwrap().get(id).cloned()
    }

    fn update(&self, id: &str, f: Box<dyn FnOnce(&mut TaskRecord) + Send>) -> Result<(), String> {
        let mut guard = self.inner.lock().unwrap();
        let task = guard
            .get_mut(id)
            .ok_or_else(|| format!("任务不存在: {id}"))?;
        f(task);
        task.updated_at = now();
        Ok(())
    }

    fn list(&self) -> Vec<TaskRecord> {
        let mut v: Vec<TaskRecord> = self.inner.lock().unwrap().values().cloned().collect();
        v.sort_by_key(|t| t.created_at);
        v
    }

    fn remove(&self, id: &str) -> Result<(), String> {
        self.inner.lock().unwrap().remove(id).map(|_| ()).ok_or_else(|| format!("任务不存在: {id}"))
    }
}

/// 当前 Unix 秒。
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
