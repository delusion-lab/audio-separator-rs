//! 模型解析与获取：manifest（本地 / URL）、三态 [`ModelRef`]、懒下载、SHA-256 校验、缓存。
//!
//! 对应 PLAN.md §5.3.3 的解析优先级：
//! 1. `Name` → 查 manifest → `local_path` 优先，否则按 `source_url` 懒下载（首次使用该名字才下载）；
//! 2. `Url` → 直接下载到缓存（M1 默认架构 `mdx`，M2 起支持显式指定）；
//! 3. `LocalPath` → 直接使用。

use std::path::{Path, PathBuf};

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
    /// 模型缓存目录（懒转换 ckpt→safetensors 的输出位置）。
    pub cache_dir: PathBuf,
}

/// 模型管理器：清单加载、模型解析、懒下载与缓存校验。
///
/// 注意：不持有 `reqwest::blocking::Client`（其内部自带 tokio runtime，跨越异步上下文
/// 创建/丢弃会 panic）。下载与清单获取均在阻塞线程内临时创建 Client，使用即弃。
pub struct ModelManager {
    list: ModelList,
    cache_dir: PathBuf,
    proxy: Option<String>,
}

impl ModelManager {
    /// URL / 本地路径形态 + 显式模型配置（yaml/json URL 或本地路径）时，
    /// 构造临时清单条目（params 来自配置文件解析结果），使架构参数直接可用；
    /// 未提供 config 时返回 `None`（架构参数走默认值，与原先行为一致）。
    fn config_entry(
        &self,
        name: &str,
        architecture: &str,
        config: Option<&str>,
        progress: Option<&mpsc::Sender<ProgressEvent>>,
        cancel: Option<&CancellationToken>,
    ) -> Result<Option<ModelEntry>> {
        let Some(cfg) = config else {
            return Ok(None);
        };
        let params = self.load_config(cfg, name, progress, cancel)?;
        Ok(Some(ModelEntry {
            name: name.to_string(),
            architecture: architecture.to_string(),
            engine: String::new(),
            source_url: None,
            sha256: None,
            config_url: Some(cfg.to_string()),
            local_path: None,
            stems: Vec::new(),
            license: None,
            params,
            mvsep: None,
        }))
    }

    /// 按配置加载模型清单并确定缓存目录。
    pub fn load(models: &ModelsConfig, proxy: Option<&str>) -> Result<Self> {
        let cache_dir = match &models.cache_dir {
            Some(p) => p.clone(),
            None => dirs::cache_dir()
                .ok_or_else(|| {
                    Error::Config("cannot determine system cache directory; set models.cache_dir in config".to_string())
                })?
                .join("audio-separator-rs")
                .join("models"),
        };
        std::fs::create_dir_all(&cache_dir)?;

        let list = match &models.list {
            Some(ModelListSource::Url(url)) => fetch_json(url, proxy)?,
            Some(ModelListSource::Path(p)) => load_json_file(p)?,
            None => ModelList {
                version: 0,
                models: Vec::new(),
            },
        };
        Ok(Self {
            list,
            cache_dir,
            proxy: proxy.map(|s| s.to_string()),
        })
    }

    /// 当前加载的模型清单（查询用）。
    pub fn list(&self) -> &ModelList {
        &self.list
    }

    /// 解析模型引用到本地文件与架构（同步执行；供 `spawn_blocking` 上下文调用）。
    pub fn resolve(
        &self,
        model: &ModelRef,
        config: Option<&str>,
        progress: Option<&mpsc::Sender<ProgressEvent>>,
        cancel: Option<&CancellationToken>,
    ) -> Result<ResolvedModel> {
        match model {
            ModelRef::Name(name) => {
                let mut entry = self.list.get(name).ok_or_else(|| {
                    Error::Model(format!(
                        "model \"{name}\" not in the manifest; use --models-file/--models-url to point at a list, or provide a model URL / local path directly"
                    ))
                })?
                .clone();
                // 显式传入的配置（CLI --config-url / server config_url 字段）优先于条目声明。
                if let Some(cfg) = config {
                    entry.config_url = Some(cfg.to_string());
                }
                // 模型参数配置（yaml/json，与权重同仓库发布）：config_url 存在时
                // 懒下载/读取并解析，解析结果作为架构参数的权威来源（替换内嵌 params）。
                if let Some(cfg) = entry.config_url.clone() {
                    entry.params = self.load_config(&cfg, name, progress, cancel)?;
                }
                let local_path = entry.local_path.clone();
                if let Some(lp) = &local_path {
                    if !lp.exists() {
                        return Err(Error::Model(format!(
                            "model entry declared local path does not exist: {}",
                            lp.display()
                        )));
                    }
                    let architecture = entry.architecture.clone();
                    return Ok(ResolvedModel {
                        entry: Some(entry),
                        local_path: lp.clone(),
                        architecture,
                        cache_dir: self.cache_dir.clone(),
                    });
                }
                let url = entry.source_url.clone().ok_or_else(|| {
                    Error::Model(format!("model \"{name}\" has no source_url and no local_path configured"))
                })?;
                let architecture = entry.architecture.clone();
                // 下载目标沿用 URL 文件名（保留扩展名，供架构按扩展名判断格式）
                let dest = self.cache_dir.join(url_file_name(&url));
                self.download_if_missing(&url, &dest, entry.sha256.as_deref(), progress, cancel)?;
                Ok(ResolvedModel {
                    entry: Some(entry),
                    local_path: dest,
                    architecture,
                    cache_dir: self.cache_dir.clone(),
                })
            }
            ModelRef::Url { url, arch } => {
                // M1 仅 mdx 架构；M2 起 URL / 本地路径形态支持显式架构（语法待定）。
                let architecture = arch.clone().unwrap_or_else(|| "mdx".to_string());
                let dest = self.cache_dir.join(url_file_name(url));
                self.download_if_missing(url, &dest, None, progress, cancel)?;
                let entry = self.config_entry(&url_file_name(url), &architecture, config, progress, cancel)?;
                Ok(ResolvedModel {
                    entry,
                    local_path: dest,
                    architecture,
                    cache_dir: self.cache_dir.clone(),
                })
            }
            ModelRef::LocalPath { path, arch } => {
                if !path.exists() {
                    return Err(Error::Model(format!("model file does not exist: {}", path.display())));
                }
                let architecture = arch.clone().unwrap_or_else(|| "mdx".to_string());
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "model".to_string());
                let entry = self.config_entry(&name, &architecture, config, progress, cancel)?;
                Ok(ResolvedModel {
                    entry,
                    local_path: path.clone(),
                    architecture,
                    cache_dir: self.cache_dir.clone(),
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

        let mut resp = blocking_client(self.proxy.as_deref())?
            .get(url)
            .send()
            .map_err(|e| Error::Network(format!("model download failed {url}: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Error::Network(format!(
                "model download failed {url}: HTTP {status}"
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
                .map_err(|e| Error::Network(format!("model download failed {url}: {e}")))?;
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
                    "model SHA-256 verification failed: expected {sha}, got {actual}"
                )));
            }
        }
        std::fs::rename(&tmp, dest)?;
        Ok(())
    }

    /// 加载并解析模型参数配置（yaml/json，URL 或本地路径）：URL 按模型名缓存到
    /// `cache_dir/configs/`，解析结果（audio/model 段拍平）作为架构参数返回。
    fn load_config(
        &self,
        config: &str,
        name: &str,
        progress: Option<&mpsc::Sender<ProgressEvent>>,
        cancel: Option<&CancellationToken>,
    ) -> Result<serde_json::Value> {
        let path = if config.starts_with("http://") || config.starts_with("https://") {
            let file = url_file_name(config);
            let ext = file
                .rsplit('.')
                .next()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "yaml".to_string());
            let dest = self
                .cache_dir
                .join("configs")
                .join(format!("{}.{}", sanitize_name(name), ext));
            self.download_if_missing(config, &dest, None, progress, cancel)?;
            dest
        } else {
            let p = PathBuf::from(config);
            if !p.exists() {
                return Err(Error::Model(format!(
                    "model config path does not exist: {}",
                    p.display()
                )));
            }
            p
        };
        let text = std::fs::read_to_string(&path)?;
        crate::model_config::parse_config(&text)
    }
}

/// 从远程 URL 获取模型清单（临时 blocking Client，走代理）。
fn fetch_json(url: &str, proxy: Option<&str>) -> Result<ModelList> {
    let resp = blocking_client(proxy)?
        .get(url)
        .send()
        .map_err(|e| Error::Network(format!("model list fetch failed {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(Error::Network(format!(
            "model list fetch failed {url}: HTTP {status}"
        )));
    }
    let text = resp
        .text()
        .map_err(|e| Error::Network(format!("model list read failed {url}: {e}")))?;
    serde_json::from_str(&text).map_err(Error::Json)
}

/// 构建 blocking Client：显式代理优先，其次环境变量 ALL_PROXY/HTTPS_PROXY/HTTP_PROXY。
fn blocking_client(proxy: Option<&str>) -> Result<reqwest::blocking::Client> {
    let mut builder = reqwest::blocking::Client::builder()
        .user_agent("audio-separator-rs/0.1");
    if let Some(p) = resolve_proxy(proxy) {
        let proxy = reqwest::Proxy::all(&p)
            .map_err(|e| Error::Config(format!("invalid proxy config {p}: {e}")))?;
        builder = builder.proxy(proxy);
    }
    builder.build().map_err(|e| Error::Network(format!("failed to build HTTP client: {e}")))
}

/// 代理解析：显式配置 > ALL_PROXY > HTTPS_PROXY > HTTP_PROXY。
pub(crate) fn resolve_proxy(explicit: Option<&str>) -> Option<String> {
    if let Some(p) = explicit {
        if !p.trim().is_empty() {
            return Some(p.to_string());
        }
    }
    for var in ["ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"] {
        if let Ok(v) = std::env::var(var) {
            if !v.trim().is_empty() {
                return Some(v);
            }
        }
    }
    None
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
pub(crate) fn sanitize_name(name: &str) -> String {
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
pub(crate) fn url_file_name(url: &str) -> String {
    let path = url.split('?').next().unwrap_or(url);
    path.rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .map(sanitize_name)
        .unwrap_or_else(|| "model.bin".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelsConfig;

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("asep-mgr-{tag}-{}", std::process::id()))
    }

    fn write_temp(path: &Path, content: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn manager(cache: &Path) -> ModelManager {
        ModelManager::load(
            &ModelsConfig {
                list: None,
                cache_dir: Some(cache.to_path_buf()),
            },
            None,
        )
        .unwrap()
    }

    #[test]
    fn resolve_local_path_with_config_parses_params() {
        let d = temp_dir("withcfg");
        let ckpt = d.join("m.ckpt");
        write_temp(&ckpt, "fake-weights");
        let cfg = d.join("m.yaml");
        write_temp(&cfg, "audio:
  sample_rate: 44100
model:
  dim: 384
  depth: 12
  heads: 8
");
        let mgr = manager(&d.join("cache"));
        let resolved = mgr
            .resolve(
                &ModelRef::LocalPath {
                    path: ckpt,
                    arch: Some("bs_roformer".to_string()),
                },
                Some(cfg.to_str().unwrap()),
                None,
                None,
            )
            .unwrap();
        let entry = resolved.entry.as_ref().expect("should build a temp entry");
        assert_eq!(entry.architecture, "bs_roformer");
        assert_eq!(entry.config_url.as_deref(), Some(cfg.to_str().unwrap()));
        assert_eq!(entry.params["dim"], 384);
        assert_eq!(entry.params["depth"], 12);
        assert_eq!(entry.params["heads"], 8);
        assert_eq!(entry.params["sample_rate"], 44100);
        assert_eq!(resolved.architecture, "bs_roformer");
    }

    #[test]
    fn resolve_local_path_without_config_keeps_none() {
        let d = temp_dir("nocfg");
        let ckpt = d.join("m.ckpt");
        write_temp(&ckpt, "fake");
        let mgr = manager(&d.join("cache"));
        let resolved = mgr
            .resolve(
                &ModelRef::LocalPath {
                    path: ckpt,
                    arch: Some("mdx".to_string()),
                },
                None,
                None,
                None,
            )
            .unwrap();
        assert!(resolved.entry.is_none());
        assert_eq!(resolved.architecture, "mdx");
    }

    #[test]
    fn resolve_name_explicit_config_overrides_entry() {
        let d = temp_dir("namecfg");
        let manifest = d.join("models.json");
        let lp = d.join("m1.onnx");
        write_temp(&lp, "fake-onnx");
        let lp_json = lp.to_str().unwrap().replace("\\", "\\\\");
        let manifest_json = format!(
            r#"{{"version":1,"models":[{{"name":"m1","architecture":"mdx","local_path":"{lp_json}","params":{{"default_k":1}}}}]}}"#
        );
        write_temp(&manifest, &manifest_json);
        let cfg = d.join("override.yaml");
        write_temp(&cfg, "overlap: 0.5
batch_size: 1
");
        let mgr = ModelManager::load(
            &ModelsConfig {
                list: Some(ModelListSource::Path(manifest)),
                cache_dir: Some(d.join("cache")),
            },
            None,
        )
        .unwrap();
        let resolved = mgr
            .resolve(&ModelRef::Name("m1".to_string()), Some(cfg.to_str().unwrap()), None, None)
            .unwrap();
        let entry = resolved.entry.as_ref().expect("should match entry");
        // 显式 config 替换条目 config_url 并解析（拍平后 overlap/batch_size 可见，
        // 且条目内嵌 default_k 保留）
        assert_eq!(entry.config_url.as_deref(), Some(cfg.to_str().unwrap()));
        assert_eq!(entry.params["overlap"], 0.5);
        assert_eq!(entry.params["batch_size"], 1);
    }
}
