//! 统一错误类型。

use thiserror::Error;

/// 全 crate 统一错误类型。
#[derive(Debug, Error)]
pub enum Error {
    /// 配置错误（字段缺失、非法取值等）。
    #[error("配置错误: {0}")]
    Config(String),

    /// IO 错误。
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),

    /// JSON 解析错误。
    #[error("JSON 解析错误: {0}")]
    Json(#[from] serde_json::Error),

    /// TOML 解析错误。
    #[error("TOML 解析错误: {0}")]
    Toml(#[from] toml::de::Error),

    /// 模型相关错误（清单缺失、校验失败、下载失败等）。
    #[error("模型错误: {0}")]
    Model(String),

    /// 后端执行错误（推理失败、引擎不可用等）。
    #[error("后端错误: {0}")]
    Backend(String),

    /// candle 引擎错误（bs_roformer 等本地 torch 移植架构）。
    #[error("candle 错误: {0}")]
    Candle(#[from] candle_core::Error),

    /// 网络错误。
    #[error("网络错误: {0}")]
    Network(String),

    /// 配额/限流错误（如 MVSEP 非 Premium 单并发、积分不足）。
    #[error("配额错误: {0}")]
    Quota(String),

    /// 格式不支持。
    #[error("格式不支持: {0}")]
    Format(String),

    /// 任务被取消。
    #[error("任务被取消")]
    Cancelled,

    /// 其他错误。
    #[error("{0}")]
    Other(String),
}

/// 便捷别名。
pub type Result<T> = std::result::Result<T, Error>;
