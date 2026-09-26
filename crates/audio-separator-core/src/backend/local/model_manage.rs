//! 模型解析与获取：manifest（本地 / URL）、三态 [`ModelRef`]、懒下载、SHA-256 校验、缓存。
//!
//! 对应 PLAN.md §5.3.3 的解析优先级：
//! 1. `Name` → 查 manifest → `local_path` 优先，否则按 `source_url` 懒下载（首次使用该名字才下载）；
//! 2. `Url` → 直接下载到缓存（M1 默认架构 `mdx`，M2 起支持显式指定）；
//! 3. `LocalPath` → 直接使用。

use std::path::{Path, PathBuf};

use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::{ModelListSource, ModelsConfig};
use crate::error::{Error, Result};
use crate::job::ProgressEvent;
use crate::model::{ModelEntry, ModelList, ModelRef};

/// 解析完成的模型：本地文件路径 + 架构 + 可选的清单条目。
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// 命中清单的条目（Name 形态；URL/本地路径形态为 None）。
    pub entry: Option<ModelEntry>,
    /// 可直接加载的模型文件路径。
    pub local_path: PathBuf,
    /// 架构标识（如 `mdx`）。
    pub architecture: String,
}

/// 模型管理器：清单加载、模型解析、懒下载与缓存校验。
///
/// 注意：不持有 `reqwest::blocking::Client`（其内部自带 tokio runtime，跨越异步上下文
/// 创建/丢弃会 panic）。下载与清单获取均在阻塞线程内临时创建 Client，使用即弃。
pub struct ModelManager {
    list: ModelList,
    cache_dir: PathBuf,
}

impl ModelManager {
    /// 按配置加载模型清单并确定缓存目录。
    pub fn load(models: &ModelsConfig) -> Result<Self> {
        let cache_dir = match &models.cache_dir {
            Some(p) => p.clone(),
            None => dirs::cache_dir()
                .ok_or_else(|| {
                    Error::Config("无法确定系统缓存目录，请在配置中设置 models.cache_dir".to_string())
                })?
                .join("audio-separator-rs")
                .join("models"),
        };
        std::fs::create_dir_all(&cache_dir)?;

        let list = match &models.list {
            Some(ModelListSource::Url(url)) => fetch_json(url)?,
            Some(ModelListSource::Path(p)) => load_json_file(p)?,
            None => ModelList {
                version: 0,
                models: Vec::new(),
            },
        };
        Ok(Self { list, cache_dir })
    }

    /// 当前加载的模型清单（查询用）。
    pub fn list(&self) -> &ModelList {
        &self.list
    }

    /// 解析模型引用到本地文件与架构（同步执行；供 `spawn_blocking` 上下文调用）。
    pub fn resolve(
        &self,
        model: &ModelRef,
        progress: Option<&mpsc::Sender<ProgressEvent>>,
        cancel: Option<&CancellationToken>,
    ) -> Result<ResolvedModel> {
        match model {
            ModelRef::Name(name) => {
                let entry = self.list.get(name).ok_or_else(|| {
                    Error::Model(format!(
                        "模型「{name}」不在清单中；可用 --models-file/--models-url 指定清单，或直接提供模型 URL / 本地路径"
                    ))
                })?;
                if let Some(lp) = &entry.local_path {
                    if !lp.exists() {
                        return Err(Error::Model(format!(
                            "模型条目声明的本地路径不存在: {}",
                            lp.display()
                        )));
                    }
                    return Ok(ResolvedModel {
                        entry: Some(entry.clone()),
                        local_path: lp.clone(),
                        architecture: entry.architecture.clone(),
                    });
                }
                let url = entry.source_url.as_ref().ok_or_else(|| {
                    Error::Model(format!("模型「{name}」缺少 source_url 且未配置 local_path"))
                })?;
                let dest = self.cache_dir.join(sanitize_name(name));
                self.download_if_missing(url, &dest, entry.sha256.as_deref(), progress, cancel)?;
                Ok(ResolvedModel {
                    entry: Some(entry.clone()),
                    local_path: dest,
                    architecture: entry.architecture.clone(),
                })
            }
            ModelRef::Url { url, arch } => {
                // M1 仅 mdx 架构；M2 起 URL / 本地路径形态支持显式架构（语法待定）。
                let architecture = arch.clone().unwrap_or_else(|| "mdx".to_string());
                let dest = self.cache_dir.join(url_file_name(url));
                self.download_if_missing(url, &dest, None, progress, cancel)?;
                Ok(ResolvedModel {
                    entry: None,
                    local_path: dest,
                    architecture,
                })
            }
            ModelRef::LocalPath { path, arch } => {
                if !path.exists() {
                    return Err(Error::Model(format!("模型文件不存在: {}", path.display())));
                }
                let architecture = arch.clone().unwrap_or_else(|| "mdx".to_string());
                Ok(ResolvedModel {
                    entry: None,
                    local_path: path.clone(),
                    architecture,
                })
            }
        }
    }

    /// 若缓存缺失（或 sha256 不匹配）则下载；下载后校验并原子改名。
    fn download_if_missing(
        &self,
        url: &str,
        dest: &Path,
        expected_sha: Option<&str>,
        progress: Option<&mpsc::Sender<ProgressEvent>>,
        cancel: Option<&CancellationToken>,
    ) -> Result<()> {
        if dest.exists() {
            if let Some(sha) = expected_sha {
                if file_sha256(dest)? == sha {
                    return Ok(());
                }
                // 缓存文件哈希不匹配：删除并重新下载。
                std::fs::remove_file(dest)?;
            } else {
                return Ok(());
            }
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = dest.with_extension("part");

        let mut resp = Client::new()
            .get(url)
            .send()
            .map_err(|e| Error::Network(format!("下载模型失败 {url}: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Error::Network(format!(
                "下载模型失败 {url}: HTTP {status}"
            )));
        }
        let total = resp.content_length();
        let mut file = std::fs::File::create(&tmp)?;
        let mut downloaded: u64 = 0;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            if let Some(c) = cancel {
                if c.is_cancelled() {
                    let _ = std::fs::remove_file(&tmp);
                    return Err(Error::Cancelled);
                }
            }
            use std::io::Read;
            let n = resp
                .read(&mut buf)
                .map_err(|e| Error::Network(format!("下载模型失败 {url}: {e}")))?;
            if n == 0 {
                break;
            }
            use std::io::Write;
            file.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            downloaded += n as u64;
            if let Some(pr) = progress {
                let _ = pr.try_send(ProgressEvent::Download { downloaded, total });
            }
        }
        use std::io::Write;
        file.flush()?;
        drop(file);

        if let Some(sha) = expected_sha {
            let actual = hex(&hasher.finalize());
            if actual != sha {
                let _ = std::fs::remove_file(&tmp);
                return Err(Error::Model(format!(
                    "模型 SHA-256 校验失败: 期望 {sha}，实际 {actual}"
                )));
            }
        }
        std::fs::rename(&tmp, dest)?;
        Ok(())
    }
}

/// 从远程 URL 获取模型清单（临时 blocking Client）。
fn fetch_json(url: &str) -> Result<ModelList> {
    let resp = Client::new()
        .get(url)
        .send()
        .map_err(|e| Error::Network(format!("获取模型清单失败 {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(Error::Network(format!(
            "获取模型清单失败 {url}: HTTP {status}"
        )));
    }
    let text = resp
        .text()
        .map_err(|e| Error::Network(format!("读取模型清单失败 {url}: {e}")))?;
    serde_json::from_str(&text).map_err(Error::Json)
}

/// 从本地 JSON 文件加载模型清单。
fn load_json_file(path: &Path) -> Result<ModelList> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(Error::Json)
}

/// 计算文件 SHA-256（hex 小写）。
pub fn file_sha256(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// 模型名 → 安全文件名（保留字母数字与 .-_）。
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// URL → 文件名（取最后一段并 sanitize）。
fn url_file_name(url: &str) -> String {
    url.rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .map(sanitize_name)
        .unwrap_or_else(|| "model.bin".to_string())
}
