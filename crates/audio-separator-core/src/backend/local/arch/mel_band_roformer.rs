//! Mel-Band-RoFormer（candle CPU FP32 前向），权重 = ep_3005_sdr_11.4360。
//!
//! 数学与权重名映射逐项对照 MSST `models/bs_roformer/mel_band_roformer.py`：
//! - librosa Slaney mel 滤波器组（60 频带）→ 每 band 覆盖频率索引（stereo 展开）
//! - 每层 time/freq Transformer **带输出 norm**（`layers.{i}.{t}.norm.gamma`，bs_roformer 无）
//! - **无 final_norm**（bs_roformer 有）；mask 估计为 3 层 Linear + 双 Tanh + GLU
//! - mask 应用为 **scatter_add + 按覆盖数平均**（mel 频带重叠），且 **zero_dc**
//! 权重 828 个 tensor（band_split 180 + layers 288 + mask_estimators 360），全部消费校验。
//!
//! 输入：立体声窗口 `2*samples`（平面布局），输出同 bs_roformer。

use candle_core::{Device, Tensor};
use candle_nn::Linear;
use rustfft::num_complex::Complex32;

use crate::error::{Error, Result};

use super::bs_roformer::{linear_forward, rms_norm, rotate, Safetensors};
use super::mel::{build_mel_bands, N_MELS};
use super::roformer_stft::{FFT, HOP, PreciseStft, Stft};

pub const DIM: usize = 384;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const DEPTH: usize = 12;
pub const SAMPLE_RATE: u32 = 44_100;
pub const CHUNK: usize = 352_800;
/// mel_band 默认 hop 441（与权重训练一致；torch.istft 输出对齐 floor(samples/HOP)*HOP）。
pub const HOP_MEL: usize = 441;

pub struct RoformerOptions {
    pub time_batch: usize,
    pub frequency_batch: usize,
}

impl Default for RoformerOptions {
    fn default() -> Self {
        Self {
            time_batch: N_MELS,
            frequency_batch: 301,
        }
    }
}

// ---- 模型结构 ----

struct Attention {
    norm_gamma: Vec<f32>,
    qkv: Linear,
    gates: Linear,
    out: Linear,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Attention {
    fn load(st: &Safetensors, prefix: &str, sequence: usize) -> Result<Self> {
        let freqs = st.vec1(&format!("{prefix}.rotary_embed.freqs"), HEAD_DIM / 2)?;
        let mut cos = Vec::with_capacity(sequence * HEAD_DIM / 2);
        let mut sin = Vec::with_capacity(sequence * HEAD_DIM / 2);
        for index in 0..sequence {
            for &frequency in &freqs {
                let angle = index as f32 * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        // MSST 版 qkv 投影到 dim_inner = heads*dim_head = 512（非 dim=384）
        let dim_inner = HEADS * HEAD_DIM;
        Ok(Self {
            norm_gamma: st.vec1(&format!("{prefix}.norm.gamma"), DIM)?,
            qkv: st.linear(&format!("{prefix}.to_qkv"), DIM, 3 * dim_inner, false)?,
            gates: st.linear(&format!("{prefix}.to_gates"), DIM, HEADS, true)?,
            out: st.linear(&format!("{prefix}.to_out.0"), dim_inner, DIM, false)?,
            cos,
            sin,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dims()[0];
        let seq = x.dims()[1];
        let xn = rms_norm(x, &self.norm_gamma, DIM)?;
        let projected = linear_forward(&self.qkv, &xn)?;
        let dim_inner = HEADS * HEAD_DIM;
        let head = |start: usize| -> Result<Tensor> {
            projected
                .narrow(2, start, dim_inner)
                .map_err(|e| Error::Model(format!("qkv narrow: {e}")))?
                .reshape((batch, seq, HEADS, HEAD_DIM))
                .map_err(|e| Error::Model(format!("qkv reshape: {e}")))?
                .permute((0, 2, 1, 3))
                .map_err(|e| Error::Model(format!("qkv permute: {e}")))
        };
        let rope = |t: Tensor| -> Result<Tensor> {
            rotate(
                &t,
                &self.cos[..seq * HEAD_DIM / 2],
                &self.sin[..seq * HEAD_DIM / 2],
                seq,
            )
        };
        let q = rope(head(0)?)?;
        let k = rope(head(dim_inner)?)?;
        let v = head(2 * dim_inner)?.contiguous().map_err(|e| Error::Model(format!("v contig: {e}")))?;
        let k_t = k
            .transpose(2, 3)
            .map_err(|e| Error::Model(format!("k transpose: {e}")))?
            .contiguous()
            .map_err(|e| Error::Model(format!("k contig: {e}")))?;
        let scores = q
            .matmul(&k_t)
            .map_err(|e| Error::Model(format!("scores matmul: {e}")))?
            .affine((HEAD_DIM as f64).powf(-0.5), 0.0)
            .map_err(|e| Error::Model(format!("scores scale: {e}")))?;
        let probs = candle_nn::ops::softmax(&scores, 3)
            .map_err(|e| Error::Model(format!("softmax: {e}")))?;
        let gates = candle_nn::ops::sigmoid(&linear_forward(&self.gates, &xn)?)
            .map_err(|e| Error::Model(format!("gates sigmoid: {e}")))?
            .reshape((batch, seq, HEADS, 1))
            .map_err(|e| Error::Model(format!("gates reshape: {e}")))?
            .permute((0, 2, 1, 3))
            .map_err(|e| Error::Model(format!("gates permute: {e}")))?;
        let values = probs
            .matmul(&v)
            .map_err(|e| Error::Model(format!("values matmul: {e}")))?
            .broadcast_mul(&gates)
            .map_err(|e| Error::Model(format!("values gate mul: {e}")))?;
        let values = values
            .permute((0, 2, 1, 3))
            .map_err(|e| Error::Model(format!("out permute: {e}")))?
            .reshape((batch, seq, dim_inner))
            .map_err(|e| Error::Model(format!("out reshape: {e}")))?;
        linear_forward(&self.out, &values)
    }
}

/// MSST 版 Transformer：attn + ff（内部各含 RMSNorm），**末尾带输出 RMSNorm**。
struct Transformer {
    attention: Attention,
    ffn_norm_gamma: Vec<f32>,
    first: Linear,
    last: Linear,
    out_norm_gamma: Vec<f32>,
}

impl Transformer {
    /// prefix = `layers.{i}.{t}.layers.0`（attn/ff），norm_prefix = `layers.{i}.{t}`（输出 norm）。
    fn load(st: &Safetensors, prefix: &str, norm_prefix: &str, sequence: usize) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(st, &format!("{prefix}.0"), sequence)?,
            ffn_norm_gamma: st.vec1(&format!("{prefix}.1.net.0.gamma"), DIM)?,
            first: st.linear(&format!("{prefix}.1.net.1"), DIM, 4 * DIM, true)?,
            last: st.linear(&format!("{prefix}.1.net.4"), 4 * DIM, DIM, true)?,
            out_norm_gamma: st.vec1(&format!("{norm_prefix}.norm.gamma"), DIM)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self
            .attention
            .forward(x)
            .map_err(|e| Error::Model(format!("attention: {e}")))?;
        let x = attn
            .add(x)
            .map_err(|e| Error::Model(format!("attn residual: {e}")))?;
        let norm = rms_norm(&x, &self.ffn_norm_gamma, DIM)?;
        let first = linear_forward(&self.first, &norm)?;
        let gelu = first
            .gelu_erf()
            .map_err(|e| Error::Model(format!("gelu: {e}")))?;
        let last = linear_forward(&self.last, &gelu)?;
        let x = last
            .add(&x)
            .map_err(|e| Error::Model(format!("ffn residual: {e}")))?;
        rms_norm(&x, &self.out_norm_gamma, DIM)
    }

    /// x: [1, batch, seq, DIM]，按 dim=1 分批前向后拼接。
    #[doc(hidden)]
    pub fn forward_batches(&self, x: &Tensor, size: usize) -> Result<Tensor> {
        let batches = x.dims()[1];
        if size >= batches {
            let inner = x.squeeze(0).map_err(|e| Error::Model(format!("squeeze: {e}")))?;
            return self
                .forward(&inner)
                .and_then(|out| out.unsqueeze(0).map_err(|e| Error::Model(format!("unsqueeze: {e}"))));
        }
        let mut outputs = Vec::with_capacity(batches.div_ceil(size));
        for start in (0..batches).step_by(size) {
            let end = (start + size).min(batches);
            let slice = x
                .narrow(1, start, end - start)
                .map_err(|e| Error::Model(format!("narrow: {e}")))?;
            let inner = slice
                .squeeze(0)
                .map_err(|e| Error::Model(format!("squeeze: {e}")))?;
            let out = self
                .forward(&inner)
                .and_then(|o| o.unsqueeze(0).map_err(|e| Error::Model(format!("unsqueeze: {e}"))))?;
            outputs.push(out);
        }
        Tensor::cat(&outputs, 1).map_err(|e| Error::Model(format!("cat: {e}")))
    }
}

/// MSST 版 MaskEstimator：Linear→Tanh→Linear→Tanh→Linear→GLU（3 层，双 Tanh）。
struct MaskEstimator {
    first: Linear,
    second: Linear,
    last: Linear,
}

impl MaskEstimator {
    fn load(st: &Safetensors, prefix: &str, input: usize) -> Result<Self> {
        Ok(Self {
            first: st.linear(&format!("{prefix}.0.0"), DIM, 4 * DIM, true)?,
            second: st.linear(&format!("{prefix}.0.2"), 4 * DIM, 4 * DIM, true)?,
            last: st.linear(&format!("{prefix}.0.4"), 4 * DIM, 2 * input, true)?,
        })
    }

    /// 输入 [1,1,frames,DIM]，输出 [frames, 2*input]（复值对折叠）。
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x
            .squeeze(0)
            .map_err(|e| Error::Model(format!("mask squeeze: {e}")))?
            .squeeze(0)
            .map_err(|e| Error::Model(format!("mask squeeze: {e}")))?; // [frames, DIM]
        let x = linear_forward(&self.first, &x)?
            .tanh()
            .map_err(|e| Error::Model(format!("mask tanh: {e}")))?;
        let x = linear_forward(&self.second, &x)?
            .tanh()
            .map_err(|e| Error::Model(format!("mask tanh: {e}")))?;
        let x = linear_forward(&self.last, &x)?;
        let half = x.dims()[1] / 2;
        let a = x
            .narrow(1, 0, half)
            .map_err(|e| Error::Model(format!("mask narrow: {e}")))?;
        let b = candle_nn::ops::sigmoid(
            &x.narrow(1, half, half)
                .map_err(|e| Error::Model(format!("mask narrow: {e}")))?,
        )
        .map_err(|e| Error::Model(format!("mask sigmoid: {e}")))?;
        a.mul(&b).map_err(|e| Error::Model(format!("mask mul: {e}")))
    }
}

// ---- 模型 ----

pub struct MelBandRoformer {
    /// (norm gamma, projection linear)，input = 2 * num_freqs * 2（stereo），各 band 不同。
    bands: Vec<(Vec<f32>, Linear)>,
    /// 每层 [time, freq] transformer。
    layers: Vec<[Transformer; 2]>,
    masks: Vec<MaskEstimator>,
    /// mel 频带结构（freq_indices 为单声道版本）。
    mel: super::mel::MelBands,
    /// stereo 展平索引：freq*2+ch，band 升序。
    freq_indices_stereo: Vec<usize>,
    options: RoformerOptions,
}

impl MelBandRoformer {
    pub fn load(path: &std::path::Path, options: RoformerOptions) -> Result<Self> {
        let device = Device::Cpu;
        let st = Safetensors::load(path, &device)?;
        let mel = build_mel_bands(SAMPLE_RATE, FFT);
        let mut consumed = 0usize;

        let mut bands = Vec::with_capacity(N_MELS);
        let mut masks = Vec::with_capacity(N_MELS);
        let mut freq_indices_stereo = Vec::new();
        for (index, &num_freqs) in mel.num_freqs_per_band.iter().enumerate() {
            let input = 4 * num_freqs; // 2 * num_freqs * 2(stereo) 复值对
            let prefix = format!("band_split.to_features.{index}");
            let gamma = st.vec1(&format!("{prefix}.0.gamma"), input)?;
            let linear = st.linear(&format!("{prefix}.1"), input, DIM, true)?;
            consumed += 3;
            bands.push((gamma, linear));
            masks.push(MaskEstimator::load(
                &st,
                &format!("mask_estimators.0.to_freqs.{index}"),
                input,
            )?);
            consumed += 6; // 3 Linear × (weight+bias)
            // stereo 展开：该 band 的 freq 段内 freq*2+0 / freq*2+1
            let start: usize = mel
                .num_freqs_per_band
                .iter()
                .take(index)
                .sum();
            for &f in &mel.freq_indices[start..start + num_freqs] {
                freq_indices_stereo.push(2 * f);
                freq_indices_stereo.push(2 * f + 1);
            }
        }
        let mut layers = Vec::with_capacity(DEPTH);
        for layer in 0..DEPTH {
            layers.push([
                Transformer::load(
                    &st,
                    &format!("layers.{layer}.0.layers.0"),
                    &format!("layers.{layer}.0"),
                    CHUNK / HOP + 1,
                )?,
                Transformer::load(
                    &st,
                    &format!("layers.{layer}.1.layers.0"),
                    &format!("layers.{layer}.1"),
                    N_MELS,
                )?,
            ]);
            consumed += 2 * 12; // attention 6 + ffn 5 + out norm 1
        }

        let total = st.len();
        if consumed != total {
            return Err(Error::Model(format!(
                "权重消费数不符: 期望 {total}, 已消费 {consumed}"
            )));
        }
        Ok(Self {
            bands,
            layers,
            masks,
            mel,
            freq_indices_stereo,
            options,
        })
    }

    pub fn batch_sizes(&self) -> (usize, usize) {
        (self.options.time_batch, self.options.frequency_batch)
    }

    /// 立体声窗口（平面布局），samples ∈ (FFT/2, CHUNK]。
    /// 输出 2*(samples/HOP*HOP)，平面布局。
    pub fn predict_window(&self, audio: &[f32], samples: usize) -> Result<Vec<f32>> {
        if samples <= FFT / 2 || samples > CHUNK || audio.len() != 2 * samples {
            return Err(Error::Model(format!(
                "mel_band_roformer 期望立体声窗口 1025..={samples}<=352800, 实际 {}",
                audio.len()
            )));
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(Error::Model("音频含非有限值".into()));
        }
        let frames = samples / HOP + 1;
        let (time_batch, frequency_batch) = self.batch_sizes();

        // 1) f64 STFT 每声道
        let mut prec = PreciseStft::new()?;
        let mut spectra = Vec::with_capacity(2);
        for channel in audio.chunks_exact(samples) {
            let spec = prec.forward(channel)?;
            if spec.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
                return Err(Error::Model("Roformer 谱含非有限值".into()));
            }
            spectra.push(spec);
        }

        // 2) mel band 投影（按 freq_indices_stereo 取谱，band 内 (freq*2+ch) 交错）
        let device = Device::Cpu;
        let mut features: Vec<Tensor> = Vec::with_capacity(N_MELS);
        let mut seg_start = 0usize;
        for ((gamma, linear), &num_freqs) in self.bands.iter().zip(&self.mel.num_freqs_per_band) {
            let input = 4 * num_freqs;
            let stereo_slice = &self.freq_indices_stereo[seg_start..seg_start + 2 * num_freqs];
            let mut values = Vec::with_capacity(frames * input);
            for frame in 0..frames {
                for &idx in stereo_slice {
                    let freq = idx / 2;
                    let ch = idx % 2;
                    let v = spectra[ch][freq * frames + frame];
                    values.push(v.re);
                    values.push(v.im);
                }
            }
            let x = Tensor::from_vec(values, (1, 1, frames, input), &device)
                .map_err(|e| Error::Model(format!("band input: {e}")))?;
            let x = rms_norm(&x, gamma, input)?;
            features.push(linear_forward(linear, &x)?);
            seg_start += 2 * num_freqs;
        }

        // 3) 12 层 time/freq transformer（每层带输出 norm）
        let mut x = Tensor::cat(&features, 1).map_err(|e| Error::Model(format!("band cat: {e}")))?;
        for [time, frequency] in &self.layers {
            x = time.forward_batches(&x, time_batch)?;
            let swapped = x
                .permute((0, 2, 1, 3))
                .map_err(|e| Error::Model(format!("swap dims: {e}")))?;
            let freq_out = frequency.forward_batches(&swapped, frequency_batch)?;
            x = freq_out
                .permute((0, 2, 1, 3))
                .map_err(|e| Error::Model(format!("swap dims: {e}")))?;
        }
        // 4) mask 估计 + scatter 平均 + zero_dc（无 final_norm）
        let bins = FFT / 2 + 1;
        let mut masked: Vec<Vec<Complex32>> = (0..2)
            .map(|_| vec![Complex32::default(); bins * frames])
            .collect();
        seg_start = 0;
        for (index, (_, estimator)) in self.bands.iter().zip(&self.masks).enumerate() {
            let band_slice = x
                .narrow(1, index, 1)
                .map_err(|e| Error::Model(format!("band narrow: {e}")))?; // [1,1,frames,DIM]
            let mask = estimator.forward(&band_slice)?; // [frames, 4*num_freqs]
            let mask = mask
                .flatten_all()
                .map_err(|e| Error::Model(format!("mask flatten: {e}")))?
                .to_vec1::<f32>()
                .map_err(|e| Error::Model(format!("mask read: {e}")))?;
            if mask.iter().any(|v| !v.is_finite()) {
                return Err(Error::Model("Roformer mask 含非有限值".into()));
            }
            if std::env::var("ASEP_DEBUG").is_ok() {
                let sum_abs: f32 = mask.iter().map(|v| v.abs()).sum();
                let nonzero = mask.iter().filter(|v| **v != 0.0).count();
                eprintln!(
                    "DBG band {index}: mask len={} sum_abs={sum_abs:.4} nonzero={nonzero}",
                    mask.len()
                );
            }
            let num_freqs = self.mel.num_freqs_per_band[index];
            let stereo_slice = &self.freq_indices_stereo[seg_start..seg_start + 2 * num_freqs];
            for frame in 0..frames {
                for (j, &idx) in stereo_slice.iter().enumerate() {
                    let freq = idx / 2;
                    let ch = idx % 2;
                    let i = frame * 2 * num_freqs * 2 + j * 2;
                    let position = freq * frames + frame;
                    masked[ch][position] += Complex32::new(mask[i], mask[i + 1]);
                }
            }
            seg_start += 2 * num_freqs;
        }
        // 平均（每 freq 被覆盖的 band 数）+ 乘原谱 + zero_dc
        let denom = &self.mel.num_bands_per_freq;
        let mut stft = Stft::new()?;
        let mut out_masked: Vec<Vec<Complex32>> = (0..2)
            .map(|_| vec![Complex32::default(); bins * frames])
            .collect();
        for ch in 0..2 {
            for freq in 0..bins {
                for frame in 0..frames {
                    let position = freq * frames + frame;
                    let d = denom[freq].max(1e-8);
                    let m = masked[ch][position] / d;
                    out_masked[ch][position] = spectra[ch][position] * m;
                }
            }
        }
        // zero_dc：DC bin 全帧置 0
        for ch in 0..2 {
            for frame in 0..frames {
                out_masked[ch][frame] = Complex32::default();
            }
        }
        if std::env::var("ASEP_DEBUG").is_ok() {
            for ch in 0..2 {
                let e: f64 = out_masked[ch].iter().map(|v| v.norm_sqr() as f64).sum();
                eprintln!("DBG masked spectrum ch{ch} energy={e:.6}");
            }
            // 输入谱能量 + 440Hz(bin20)/880Hz(bin41) 位置的 masked 值
            for ch in 0..2 {
                let e_in: f64 = spectra[ch].iter().map(|v| v.norm_sqr() as f64).sum();
                eprintln!(
                    "DBG in spectrum ch{ch} energy={e_in:.3}  masked[bin20]={:?}  masked[bin41]={:?}",
                    out_masked[ch][20 * frames],
                    out_masked[ch][41 * frames]
                );
            }
        }

        // 5) iSTFT 每声道
        let output_samples = samples / HOP * HOP;
        let mut output = Vec::with_capacity(2 * output_samples);
        for spectrum in &out_masked {
            output.extend(stft.inverse(spectrum, frames)?);
        }
        if output.len() != 2 * output_samples || output.iter().any(|v| !v.is_finite()) {
            return Err(Error::Model("Roformer 输出波形无效".into()));
        }
        Ok(output)
    }

    /// 完整分离：任意采样率交错音频 → (vocals, instrumental)。
    pub fn separate(&self, samples: &[f32], sample_rate: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        if sample_rate != SAMPLE_RATE {
            let resampled = crate::io::resample_to(samples, 2, sample_rate, SAMPLE_RATE)
                .map_err(|e| Error::Backend(format!("重采样失败: {e}")))?;
            self.separate_44100(&resampled)
        } else {
            self.separate_44100(samples)
        }
    }

    fn separate_44100(&self, samples: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if samples.len() % 2 != 0 {
            return Err(Error::Model("交错音频长度必须为偶数".into()));
        }
        let len = samples.len() / 2;
        let mut audio: Vec<&[f32]> = Vec::with_capacity(2);
        for ch in 0..2 {
            audio.push(&samples[ch * len..(ch + 1) * len]);
        }
        let samples_len = audio[0].len();
        let padded = samples_len.max(FFT / 2 + 1).div_ceil(HOP) * HOP;
        let tail = padded.saturating_sub(CHUNK);
        let starts: Vec<usize> = (0..tail).step_by(CHUNK / 4).chain([tail]).collect();
        let silent = audio.iter().copied().flatten().all(|v| *v == 0.0);
        let starts = if silent { Vec::new() } else { starts };

        let mut vocals = vec![0.0f32; 2 * samples_len];
        let mut counter = vec![0.0f32; samples_len];
        let window: Vec<f32> = (0..CHUNK)
            .map(|i| {
                (0.54 - 0.46 * (std::f64::consts::TAU * i as f64 / (CHUNK - 1) as f64).cos())
                    as f32
            })
            .collect();
        for &offset in &starts {
            let length = CHUNK.min(padded - offset);
            let keep = length.min(samples_len - offset);
            let mut input = vec![0.0f32; 2 * length];
            for (channel, source) in audio.iter().enumerate() {
                input[channel * length..channel * length + keep]
                    .copy_from_slice(&source[offset..offset + keep]);
            }
            let predicted = self.predict_window(&input, length)?;
            debug_assert_eq!(predicted.len(), 2 * length);
            for (channel, _source) in audio.iter().enumerate() {
                for i in 0..keep {
                    vocals[channel * samples_len + offset + i] +=
                        predicted[channel * length + i] * window[i];
                }
                for i in keep..length {
                    if offset + i < samples_len {
                        vocals[channel * samples_len + offset + i] +=
                            predicted[channel * length + i] * window[i];
                    }
                }
            }
            for i in 0..length {
                let pos = offset + i;
                if pos < samples_len {
                    counter[pos] += window[i];
                }
            }
        }
        let mut out_vocals = vec![0.0f32; 2 * samples_len];
        for channel in 0..2 {
            for i in 0..samples_len {
                if !silent && counter[i] <= 0.0 {
                    return Err(Error::Model("Roformer 调度窗未覆盖样本".into()));
                }
                let weight = if silent { 1.0 } else { counter[i] };
                out_vocals[channel * samples_len + i] =
                    vocals[channel * samples_len + i] / weight;
            }
        }
        let mut instrumental = Vec::with_capacity(2 * samples_len);
        for i in 0..2 * samples_len {
            instrumental.push(samples[i] - out_vocals[i]);
        }
        Ok((out_vocals, instrumental))
    }
}
