//! 后端抽象：`Separator` trait 与统一请求/结果类型。
//!
//! CLI 与服务端只依赖本 trait；本地多架构引擎（M1/M2）与 MVSEP 客户端（M3）
//! 都是它的实现，通过 `BackendFactory`（M1 起）按配置实例化。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::BackendKind;
use crate::error::Result;
use crate::job::ProgressEvent;
use crate::model::{ModelRef, OutputFormat};

/// 输入音频来源。
#[derive(Debug, Clone)]
pub enum Input {
    /// 本地文件路径。
    Path(PathBuf),
    /// 远程 URL（CLI 输入为 URL 时，先下载到临时/工作目录）。
    Url(String),
    /// 内存字节（服务端上传路径）。
    Bytes(Vec<u8>),
}

/// 一次分离请求。
#[derive(Debug, Clone)]
pub struct SeparationRequest {
    /// 输入音频。
    pub input: Input,
    /// 模型引用（名字 / 下载 URL / 本地路径）。
    pub model: ModelRef,
    /// 输出格式。
    pub output_format: OutputFormat,
    /// 输出目录。
    pub output_dir: PathBuf,
    /// 可选：只输出部分分轨；为空输出模型定义的全部分轨。
    pub select_stems: Option<Vec<String>>,
}

/// 分离结果。
#[derive(Debug, Clone)]
pub struct SeparationResult {
    /// 分轨名 → 输出文件路径。
    pub stems: BTreeMap<String, PathBuf>,
    /// 实际执行后端。
    pub backend: BackendKind,
    /// 总耗时。
    pub elapsed: Duration,
}

/// 统一后端接口。
#[async_trait]
pub trait Separator: Send + Sync {
    /// 执行一次分离。
    ///
    /// - `progress`：进度事件发送端（可选）。
    /// - `cancel`：取消令牌（可选）；触发后应尽快中止并返回 [`crate::Error::Cancelled`]。
    async fn separate(
        &self,
        req: SeparationRequest,
        progress: Option<mpsc::Sender<ProgressEvent>>,
        cancel: Option<CancellationToken>,
    ) -> Result<SeparationResult>;
}

// M0 骨架：本地引擎与 MVSEP 客户端在 M1-M3 实现。
// 预留模块路径：
//   backend/local/  —— 模型管理（model_manage）+ 引擎（onnx / candle）
//   backend/mvsep.rs —— MVSEP REST 客户端
