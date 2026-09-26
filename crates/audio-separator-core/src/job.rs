//! 作业状态机与进度事件。

use serde::{Deserialize, Serialize};

/// 作业状态（服务端 API 响应与 CLI 查询共用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    /// 排队中。
    Queued,
    /// 执行中。
    Running,
    /// 完成。
    Done,
    /// 失败。
    Failed,
    /// 已取消。
    Cancelled,
}

/// 进度事件（`Separator::separate` 经 `mpsc` channel 上报）。
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// 阶段切换（如 "resolve_model" / "read_audio" / "infer"）。
    Stage(String),
    /// 模型文件下载进度。
    Download { downloaded: u64, total: Option<u64> },
    /// 处理进度（百分比 0.0-1.0）。
    Process { percent: f32, message: Option<String> },
    /// 写分轨进度。
    Writing { stem: String, percent: f32 },
    /// 全部完成。
    Finished,
}
