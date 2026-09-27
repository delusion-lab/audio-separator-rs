//! 配置：TOML 文件 + 默认值；CLI 与环境变量覆盖在后续里程碑接入。

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::model::OutputFormat;

/// 后端类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// 本地推理（多架构 onnx / candle 引擎）。
    Local,
    /// MVSEP 云平台。
    Mvsep,
}

/// MVSEP 区域端点。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MvsepRegion {
    /// 主域名，按地理位置自动分流。
    Auto,
    /// 德国。
    De,
    /// 德国 2。
    De2,
    /// 新加坡。
    Sg,
}

impl MvsepRegion {
    /// API 基础地址（MVSEP 官方区域端点）。
    pub fn base_url(self) -> &'static str {
        match self {
            MvsepRegion::Auto => "https://mvsep.com/api",
            MvsepRegion::De => "https://de.mvsep.com/api",
            MvsepRegion::De2 => "https://de2.mvsep.com/api",
            MvsepRegion::Sg => "https://sg.mvsep.com/api",
        }
    }
}

/// 模型清单来源：本地 JSON 路径或远程 URL。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelListSource {
    Path(PathBuf),
    Url(String),
}

/// 模型相关配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    /// 模型清单源（本地 JSON 或 URL）。为空时使用内置默认清单（M1 起生效）。
    pub list: Option<ModelListSource>,
    /// 模型缓存目录。为空时使用系统缓存目录下 `audio-separator-rs/models`。
    pub cache_dir: Option<PathBuf>,
}

/// MVSEP 云后端配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MvsepConfig {
    /// API Key（也可用环境变量 `ASEP_MVSEP_API_KEY`）。
    pub api_key: Option<String>,
    /// 区域。
    pub region: MvsepRegion,
    /// 并发任务数（非 Premium 仅允许 1）。
    pub concurrency: usize,
    /// 轮询间隔（秒）。
    pub poll_interval_secs: u64,
    /// 轮询超时（秒）。
    pub poll_timeout_secs: u64,
    /// 处理完成回调 URL（可选，透传平台 webhook）。
    pub webhook_url: Option<String>,
}

impl Default for MvsepConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            region: MvsepRegion::Auto,
            concurrency: 1,
            poll_interval_secs: 5,
            poll_timeout_secs: 3600,
            webhook_url: None,
        }
    }
}

/// 网络配置（统一走用户 HTTP 代理）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NetworkConfig {
    /// HTTP(S) 代理地址（如 `http://127.0.0.1:12355`）。
    /// 为空时回退环境变量 `ALL_PROXY` / `HTTPS_PROXY` / `HTTP_PROXY`。
    pub proxy: Option<String>,
}

/// 服务端配置（CLI `serve` 与 server crate 共用）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// 监听地址。
    pub addr: String,
    /// 可选 Bearer Token（为空则不鉴权，D4 决策）。
    pub auth_token: Option<String>,
    /// 上传文件大小上限（字节）。
    pub max_upload_bytes: u64,
    /// 并行处理任务数（本地推理 / MVSEP 提交共用，MVSEP 非 Premium 平台侧仅允许 1）。
    pub workers: usize,
    /// 任务终态回调 URL（可选）：任务 done/failed/cancelled 时服务端向该 URL POST JSON。
    pub webhook_url: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080".to_string(),
            auth_token: None,
            max_upload_bytes: 1_000_000_000,
            workers: 1,
            webhook_url: None,
        }
    }
}

/// 输出相关配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// 默认输出格式。
    pub format: OutputFormat,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Wav16,
        }
    }
}

/// 顶层配置。所有字段均有默认值，`#[serde(default)]` 允许 TOML 只写需要覆盖的字段。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// 默认后端。
    pub backend: BackendKind,
    /// 模型相关配置。
    pub models: ModelsConfig,
    /// MVSEP 配置。
    pub mvsep: MvsepConfig,
    /// 服务端配置。
    pub server: ServerConfig,
    /// 网络配置（代理）。
    pub network: NetworkConfig,
    /// 输出配置。
    pub output: OutputConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: BackendKind::Local,
            models: ModelsConfig::default(),
            mvsep: MvsepConfig::default(),
            server: ServerConfig::default(),
            network: NetworkConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl Config {
    /// 从 TOML 文件加载配置；文件不存在或解析失败时返回错误。
    /// 仅需覆盖默认值的文件可省略任意字段。
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&text)?;
        Ok(cfg)
    }
}
