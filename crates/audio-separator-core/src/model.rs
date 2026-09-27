//! 模型清单（manifest）与模型引用类型。
//!
//! 设计原则（PLAN.md §5.3）：
//! - 模型是独立资产（文件 + 清单条目），架构是推理代码（引擎 + 参数 schema），两者分离。
//! - 分轨数由模型条目声明；参数挂在架构上，绝不跨架构混用。
//! - 模型清单可来自本地 JSON 文件或 URL（如 GitHub raw），默认指向独立模型仓库。

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// 常见分轨名（统一小写字符串；任意模型的分轨全集由 manifest 的 `stems` 声明）。
pub const STEM_VOCALS: &str = "vocals";
pub const STEM_INSTRUMENTAL: &str = "instrumental";
pub const STEM_DRUMS: &str = "drums";
pub const STEM_BASS: &str = "bass";
pub const STEM_GUITAR: &str = "guitar";
pub const STEM_PIANO: &str = "piano";
pub const STEM_OTHER: &str = "other";

/// 推理引擎标识。
pub const ENGINE_ONNX: &str = "onnx";
pub const ENGINE_CANDLE: &str = "candle";

/// 模型引用三态：按名（命中 manifest 才懒下载）/ 直接下载 URL / 本地模型文件。
///
/// `arch` 仅对 URL / 本地路径形态可选指定；按名时架构由 manifest 条目决定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    /// 按模型名引用，命中 manifest 条目；首次使用该名字才按 `source_url` 懒下载。
    Name(String),
    /// 直接指定下载链接。
    Url { url: String, arch: Option<String> },
    /// 直接使用用户已有模型文件。
    LocalPath { path: PathBuf, arch: Option<String> },
}

impl ModelRef {
    /// 人类可读描述（CLI / 日志用）。
    pub fn describe(&self) -> String {
        match self {
            ModelRef::Name(name) => format!("模型名「{name}」"),
            ModelRef::Url { url, arch } => match arch {
                Some(a) => format!("下载链接 {url}（架构 {a}）"),
                None => format!("下载链接 {url}"),
            },
            ModelRef::LocalPath { path, arch } => match arch {
                Some(a) => format!("本地文件 {}（架构 {a}）", path.display()),
                None => format!("本地文件 {}", path.display()),
            },
        }
    }
}

/// 输出格式：本地编码与 MVSEP `output_format` 统一映射。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// WAV 16-bit（MVSEP code 1）。
    #[serde(alias = "wav")]
    Wav16,
    /// WAV 32-bit float（MVSEP code 4）。
    Wav32,
    /// FLAC 16-bit（MVSEP code 2）。
    #[serde(alias = "flac")]
    Flac16,
    /// FLAC 24-bit（MVSEP code 5）。
    Flac24,
    /// MP3 320kbps（MVSEP code 0）。
    Mp3,
    /// M4A lossy（MVSEP code 3）。
    M4a,
}

impl OutputFormat {
    /// MVSEP API `output_format` 编码（0=mp3,1=wav16,2=flac16,3=m4a,4=wav32,5=flac24）。
    pub fn as_mvsep_code(self) -> i32 {
        match self {
            OutputFormat::Mp3 => 0,
            OutputFormat::Wav16 => 1,
            OutputFormat::Flac16 => 2,
            OutputFormat::M4a => 3,
            OutputFormat::Wav32 => 4,
            OutputFormat::Flac24 => 5,
        }
    }

    /// 输出文件扩展名。
    pub fn as_extension(self) -> &'static str {
        match self {
            OutputFormat::Wav16 | OutputFormat::Wav32 => "wav",
            OutputFormat::Flac16 | OutputFormat::Flac24 => "flac",
            OutputFormat::Mp3 => "mp3",
            OutputFormat::M4a => "m4a",
        }
    }
}

/// 模型清单条目。
///
/// `params` 为架构专属参数（JSON 对象），其反序列化与校验由架构注册表完成（M1+ 实现），
/// 此处保持开放值以兼容任意架构；`mdx` 与 `bs_roformer` 等架构的参数集互不混用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// 模型名（`--model` 按此匹配）。
    pub name: String,
    /// 架构标识（如 `mdx` / `vr` / `demucs` / `bs_roformer` / `mel_band_roformer` / `bs_polarformer`）。
    pub architecture: String,
    /// 推理引擎（`onnx` / `candle`）。
    #[serde(default)]
    pub engine: String,
    /// 模型文件下载地址（按名懒下载时使用）。
    #[serde(default)]
    pub source_url: Option<String>,
    /// 模型文件 SHA-256（远程清单强制要求，下载后校验）。
    #[serde(default)]
    pub sha256: Option<String>,
    /// 模型参数配置（yaml/json）下载地址：开源模型通常与权重同仓库发布 config
    /// （如 HuggingFace `config.yaml` / MSST 训练 yaml）。按名懒下载解析，
    /// 结果合并进 `params`（config_url 存在时以其解析结果为准）。也支持本地路径。
    #[serde(default)]
    pub config_url: Option<String>,
    /// 用户已有的本地模型路径（存在则优先直接使用，不下载）。
    #[serde(default)]
    pub local_path: Option<PathBuf>,
    /// 该模型输出的分轨全集（分轨数由模型定义，D2 决策）。
    #[serde(default)]
    pub stems: Vec<String>,
    /// 模型许可（随条目展示）。
    #[serde(default)]
    pub license: Option<String>,
    /// 架构专属参数（schema 校验在架构注册表完成）。
    #[serde(default)]
    pub params: serde_json::Value,
    /// MVSEP 云后端映射（可选）：按名引用时，MVSEP 后端据此提交 sep_type 与附加选项。
    #[serde(default)]
    pub mvsep: Option<MvsepEntry>,
}

/// MVSEP 云后端条目映射：平台分离类型 ID 与附加选项。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvsepEntry {
    /// 平台分离类型 ID（`/api/app/algorithms` 的 `render_id`，如 40=BS Roformer 2-stem）。
    pub sep_type: u64,
    /// 平台附加选项（`add_opt1/2/3` 键值，如 `{"add_opt1": "81"}`）；未列出的用平台默认。
    #[serde(default)]
    pub add_opts: serde_json::Value,
}

/// 模型清单：本地 JSON 文件或远程 URL 均可，条目集中维护。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    /// 清单格式版本。
    pub version: u32,
    /// 模型条目。
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

impl ModelList {
    /// 按名查找模型条目。
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.name == name)
    }
}
