//! 本地推理后端：模型管理 + 引擎（onnx / candle）+ 各架构推理实现。
//!
//! M1 交付：model_manage + onnx 引擎 + mdx 架构端到端（`LocalSeparator`）。

pub mod arch;
pub mod engine;
pub mod model_manage;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::task::spawn_blocking;
use tokio_util::sync::CancellationToken;

use crate::backend::{Input, SeparationRequest, SeparationResult, Separator};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io;
use crate::job::ProgressEvent;
use crate::model::OutputFormat;

use self::arch::bs_polarformer::BsPolarformer;
use self::arch::bs_roformer::{BsRoformer, RoformerOptions as BsRoformerOptions};
use self::arch::mdx::{separate_mdx, MdxParams};
use self::arch::mel_band_roformer::{MelBandRoformer, RoformerOptions as MelRoformerOptions};
use self::engine::OnnxSession;
use self::model_manage::{ModelManager, ResolvedModel};
use crate::model::ModelEntry;
use crate::weights::convert_ckpt_to_safetensors;

/// Roformer 家族在 candle 下统一输出 (vocals, instrumental)。
const ROFORMER_SAMPLE_RATE: u32 = 44_100;

/// 本地分离后端（M1：mdx 架构端到端；M2：三款 Roformer 接入）。
pub struct LocalSeparator {
    manager: Arc<ModelManager>,
}

impl LocalSeparator {
    /// 按配置构建本地后端（加载模型清单、确定缓存目录）。
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            manager: Arc::new(ModelManager::load(
                &config.models,
                config.network.proxy.as_deref(),
            )?),
        })
    }
}

#[async_trait]
impl Separator for LocalSeparator {
    async fn separate(
        &self,
        req: SeparationRequest,
        progress: Option<mpsc::Sender<ProgressEvent>>,
        cancel: Option<CancellationToken>,
    ) -> Result<SeparationResult> {
        // 本地执行（解析模型、解码、推理、写盘）为阻塞密集操作，包进 spawn_blocking。
        let manager = Arc::clone(&self.manager);
        let model = req.model.clone();
        let config_url = req.config_url.clone();
        let input = req.input.clone();
        let output_format = req.output_format;
        let output_dir = req.output_dir.clone();
        let select_stems = req.select_stems.clone();

        let job = spawn_blocking(move || {
            emit(
                progress.as_ref(),
                ProgressEvent::Stage("resolve_model".to_string()),
            );
            let resolved =
                manager.resolve(&model, config_url.as_deref(), progress.as_ref(), cancel.as_ref())?;
            run_local(
                resolved,
                &input,
                output_format,
                &output_dir,
                select_stems,
                progress.as_ref(),
                cancel.as_ref(),
            )
        });
        job.await
            .map_err(|e| Error::Other(format!("本地任务异常: {e}")))?
    }
}

/// 本地执行流水线：解码 → 重采样 → 架构推理 → 写分轨。
#[allow(clippy::too_many_arguments)]
fn run_local(
    resolved: ResolvedModel,
    input: &Input,
    output_format: OutputFormat,
    output_dir: &PathBuf,
    select_stems: Option<Vec<String>>,
    progress: Option<&mpsc::Sender<ProgressEvent>>,
    cancel: Option<&CancellationToken>,
) -> Result<SeparationResult> {
    let started = Instant::now();

    // 输入：M1 支持本地路径；URL / 内存字节在后续里程碑交付。
    let input_path = match input {
        Input::Path(p) => p.clone(),
        Input::Url(_) => {
            return Err(Error::Backend(
                "输入 URL 支持将在后续里程碑交付".to_string(),
            ))
        }
        Input::Bytes(_) => {
            return Err(Error::Backend(
                "内存输入支持将在服务端里程碑交付".to_string(),
            ))
        }
    };

    // 架构分发（M1 mdx；M2 起三款 Roformer）
    let (stems, model_sample_rate): (Vec<(String, Vec<f32>)>, u32) =
        match resolved.architecture.as_str() {
            "mdx" => {
                let mut params = MdxParams::from_entry(resolved.entry.as_ref())?;
                emit(progress, ProgressEvent::Stage("load_model".to_string()));
                let mut session = OnnxSession::load(&resolved.local_path)?;
                // 模型 metadata 为参数权威来源（UVR 转换模型自描述 n_fft/hop/dim_f/dim_t）
                params.apply_metadata(&|k| session.metadata_value(k))?;
                emit(progress, ProgressEvent::Stage("read_audio".to_string()));
                let audio = io::decode(&input_path)?;
                let samples = io::resample_to(
                    &audio.samples,
                    audio.channels,
                    audio.sample_rate,
                    params.sample_rate,
                )?;
                emit(progress, ProgressEvent::Stage("infer".to_string()));
                let stems = separate_mdx(&mut session, &samples, &params, progress, cancel)?;
                (stems, params.sample_rate)
            }
            "bs_roformer" | "mel_band_roformer" | "bs_polarformer" => {
                // ckpt/pt 权重先懒转换为 safetensors（缓存；权重形状校验在架构 load 内强制）
                let weights = ensure_safetensors(
                    &resolved.local_path,
                    &resolved.cache_dir,
                    &resolved,
                    progress,
                )?;
                emit(progress, ProgressEvent::Stage("load_model".to_string()));
                // 按架构构建专属 RoformerOptions（字段名同义但类型独立，不跨架构共用）
                let (tb, fb) = batch_params(resolved.entry.as_ref());
                emit(progress, ProgressEvent::Stage("read_audio".to_string()));
                let audio = io::decode(&input_path)?;
                let samples = io::resample_to(
                    &audio.samples,
                    audio.channels,
                    audio.sample_rate,
                    ROFORMER_SAMPLE_RATE,
                )?;
                emit(progress, ProgressEvent::Stage("infer".to_string()));
                let (vocals, instrumental) = match resolved.architecture.as_str() {
                    "bs_roformer" => BsRoformer::load(
                        &weights,
                        BsRoformerOptions {
                            time_batch: tb.unwrap_or(62),
                            frequency_batch: fb.unwrap_or(301),
                        },
                    )?
                    .separate(&samples, ROFORMER_SAMPLE_RATE)?,
                    "mel_band_roformer" => MelBandRoformer::load(
                        &weights,
                        MelRoformerOptions {
                            time_batch: tb.unwrap_or(60),
                            frequency_batch: fb.unwrap_or(301),
                        },
                    )?
                    .separate(&samples, ROFORMER_SAMPLE_RATE)?,
                    _ => BsPolarformer::load(
                        &weights,
                        self::arch::bs_polarformer::RoformerOptions {
                            time_batch: tb.unwrap_or(62),
                            frequency_batch: fb.unwrap_or(301),
                        },
                    )?
                    .separate(&samples, ROFORMER_SAMPLE_RATE)?,
                };
                (
                    vec![
                        ("vocals".to_string(), vocals),
                        ("instrumental".to_string(), instrumental),
                    ],
                    ROFORMER_SAMPLE_RATE,
                )
            }
            other => {
                return Err(Error::Backend(format!(
                    "架构「{other}」尚未实现（M2 交付 Roformer 家族）"
                )));
            }
        };

    // 写盘（M1 仅 WAV16；其他格式 M5 交付）
    std::fs::create_dir_all(output_dir)?;
    let ext = output_format.as_extension();
    let mut result_stems = BTreeMap::new();
    let total = stems.len();
    let mut written = 0usize;
    for (name, mut samples) in stems {
        if let Some(sel) = &select_stems {
            if !sel.iter().any(|s| s == &name) {
                continue;
            }
        }
        io::normalize(&mut samples);
        let path = output_dir.join(format!("{name}.{ext}"));
        io::write_audio(&path, &samples, model_sample_rate, 2, output_format)?;
        result_stems.insert(name.clone(), path);
        written += 1;
        if let Some(pr) = progress {
            let _ = pr.try_send(ProgressEvent::Writing {
                stem: name,
                percent: if total > 0 {
                    written as f32 / total as f32
                } else {
                    1.0
                },
            });
        }
    }
    if let Some(pr) = progress {
        let _ = pr.try_send(ProgressEvent::Finished);
    }

    Ok(SeparationResult {
        stems: result_stems,
        backend: crate::config::BackendKind::Local,
        elapsed: started.elapsed(),
    })
}

fn emit(sender: Option<&mpsc::Sender<ProgressEvent>>, ev: ProgressEvent) {
    if let Some(s) = sender {
        let _ = s.try_send(ev);
    }
}

/// 从清单条目 params 读取 batch 参数（各架构自己的字段，缺失时用默认值）。
fn batch_params(entry: Option<&ModelEntry>) -> (Option<usize>, Option<usize>) {
    let obj = entry.and_then(|e| e.params.as_object());
    let get = |k: &str| {
        obj.and_then(|o| o.get(k))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    };
    (get("time_batch"), get("frequency_batch"))
}

/// ckpt/pt → safetensors 懒转换：已 是 safetensors 直接用；
/// 缓存缺失（或比源文件旧）时调用转换器生成，返回可加载路径。
fn ensure_safetensors(
    local_path: &std::path::Path,
    cache_dir: &std::path::Path,
    resolved: &ResolvedModel,
    progress: Option<&mpsc::Sender<ProgressEvent>>,
) -> Result<std::path::PathBuf> {
    let ext = local_path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if ext == "safetensors" {
        return Ok(local_path.to_path_buf());
    }
    if !matches!(ext.as_str(), "ckpt" | "pt" | "pth") {
        return Err(Error::Model(format!(
            "架构「{}」不支持的权重格式 .{ext}（期望 .ckpt/.pt/.pth 或 .safetensors）",
            resolved.architecture
        )));
    }
    let name = resolved
        .entry
        .as_ref()
        .map(|e| e.name.clone())
        .unwrap_or_else(|| {
            local_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".to_string())
        });
    let out = cache_dir.join(format!("{}.safetensors", model_manage::sanitize_name(&name)));
    if out.exists() {
        let cached = std::fs::metadata(&out)
            .map(|m| m.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH))
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let source = std::fs::metadata(local_path)
            .map(|m| m.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH))
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if cached >= source {
            return Ok(out);
        }
    }
    emit(progress, ProgressEvent::Stage("convert".to_string()));
    convert_ckpt_to_safetensors(local_path, &out)?;
    Ok(out)
}
