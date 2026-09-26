//! BS-RoFormer 1296（candle CPU FP32 前向）。
//!
//! 数学与权重名映射逐项对照 uvr_roformer（IronHpc/UVR-rs，Burn 参考实现）：
//! 62 频带、DIM=512 / HEADS=8 / HEAD_DIM=64 / DEPTH=12、FFT=2048 / HOP=441、
//! 相邻对 RoPE（非 rotate_half）、RMSNorm 变体（÷sqrt(sum(x²))·√dim·γ）、erf GELU。
//! 权重来自 M2-A 转换器产出的 safetensors（ckpt → safetensors，699 个 tensor）。
//!
//! 输入：立体声窗口 `2*samples`（交错），samples ∈ (1024, 352800]。
//! 输出：`2 * (samples/HOP*HOP)`，前一半左声道、后一半右声道（与参考一致）。

use std::collections::HashMap;

use rayon::prelude::*;

use candle_core::{Device, Tensor};
use candle_nn::Linear;
use rustfft::num_complex::Complex32;

use crate::error::{Error, Result};

use super::roformer_stft::{FFT, HOP, PreciseStft, Stft};

pub const DIM: usize = 512;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const DEPTH: usize = 12;
pub const SAMPLE_RATE: u32 = 44_100;
pub const CHUNK: usize = 352_800;

/// 62 个频带宽度，总和 = 1025 = FFT/2+1。
pub const BANDS: [usize; 62] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48,
    48, 48, 48, 128, 129,
];

pub struct RoformerOptions {
    /// 时间注意力每批处理的频带数（uvr 默认 62 全量）。
    pub time_batch: usize,
    /// 频率注意力每批处理的帧数（uvr 默认 301）。
    pub frequency_batch: usize,
}

impl Default for RoformerOptions {
    fn default() -> Self {
        Self {
            time_batch: BANDS.len(),
            frequency_batch: 301,
        }
    }
}

/// 简易 safetensors 只读加载（f32 数据）。
pub struct Safetensors {
    values: HashMap<String, Tensor>,
}

impl Safetensors {
    pub fn load(path: &std::path::Path, device: &Device) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Model(format!("读取 safetensors 失败: {e}")))?;
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])
            .map_err(|e| Error::Model(format!("safetensors header 解析失败: {e}")))?;
        let data = &bytes[8 + header_len..];
        let mut values = HashMap::with_capacity(header.as_object().map_or(0, |o| o.len()));
        for (name, meta) in header
            .as_object()
            .ok_or_else(|| Error::Model("header 非对象".into()))?
        {
            let dtype = meta["dtype"]
                .as_str()
                .ok_or_else(|| Error::Model("缺 dtype".into()))?;
            if dtype != "F32" {
                return Err(Error::Model(format!("{name}: 仅支持 F32, 实际 {dtype}")));
            }
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .ok_or_else(|| Error::Model("缺 shape".into()))?
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let offsets = meta["data_offsets"]
                .as_array()
                .ok_or_else(|| Error::Model("缺 offsets".into()))?;
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            let raw = &data[start..end];
            if raw.len() != shape.iter().product::<usize>() * 4 {
                return Err(Error::Model(format!("{name}: 数据长度不符")));
            }
            let vec: Vec<f32> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let t = Tensor::from_vec(vec, shape, device)
                .map_err(|e| Error::Model(format!("{name}: {e}")))?;
            values.insert(name.clone(), t);
        }
        Ok(Self { values })
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.values
            .get(name)
            .ok_or_else(|| Error::Model(format!("缺少权重: {name}")))
    }

    /// 读取并校验形状，转为 f32 向量（用于 norm/rope 等标量参数）。
    fn vec1(&self, name: &str, expected: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dims() != [expected] {
            return Err(Error::Model(format!("{name}: 形状不符 {:?}", t.dims())));
        }
        let v = t.to_vec1::<f32>()?;
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::Model(format!("{name}: 权重含非有限值")));
        }
        Ok(v)
    }

    fn linear(&self, prefix: &str, input: usize, output: usize, bias: bool) -> Result<Linear> {
        let weight = self.tensor(&format!("{prefix}.weight"))?;
        if weight.dims() != [output, input] {
            return Err(Error::Model(format!(
                "{prefix}.weight: 形状不符 {:?}",
                weight.dims()
            )));
        }
        let bias_t = if bias {
            let b = self.tensor(&format!("{prefix}.bias"))?;
            if b.dims() != [output] {
                return Err(Error::Model(format!("{prefix}.bias: 形状不符")));
            }
            Some(b.clone())
        } else {
            None
        };
        Ok(Linear::new(weight.clone(), bias_t))
    }
}

/// candle_nn 0.9 的 Linear 无 forward，手动实现：x @ W^T + b。
/// 采用 uvr 默认 Flattened 布局：x 压成 2D [N, in] @ [in, out] 再还原，
/// 规避 candle 对高维 matmul 的 batch 广播限制。
/// 性能注：candle CPU matmul（rayon 并行）与 Intel MKL sgemm 实测同级别
/// （约 85 GFLOPS，本机 16 核），BS-RoFormer 1296 CPU 单窗 ~100s 属模型固有量级，
/// 参考实现 UVR-rs 同机同配置约 82-88s。故不依赖外部 BLAS。
#[doc(hidden)]
pub fn linear_forward(l: &Linear, x: &Tensor) -> Result<Tensor> {
    let w = l.weight().t()?; // [in, out]
    let shape = x.shape().clone();
    let input = shape.dims().last().copied().unwrap_or(0);
    let n = shape.elem_count() / input;
    let flat = x
        .reshape((n, input))
        .map_err(|e| Error::Model(format!("linear flatten: {e}")))?;
    let y = flat
        .matmul(&w)
        .map_err(|e| Error::Model(format!("linear matmul: {e}")))?;
    let mut out_shape = shape.dims().to_vec();
    *out_shape.last_mut().unwrap() = w.dims()[1];
    let y = y
        .reshape(out_shape)
        .map_err(|e| Error::Model(format!("linear reshape: {e}")))?;
    match l.bias() {
        Some(b) => y
            .broadcast_add(b)
            .map_err(|e| Error::Model(format!("linear bias: {e}"))),
        None => Ok(y),
    }
}

// ---- 基础算子（对照 uvr roformer/cpu.rs）----

/// RMSNorm 变体：x / sqrt(sum(x²)).clamp_min(1e-12) * sqrt(dim) * γ，作用于最后一维。
#[doc(hidden)]
pub fn rms_norm(x: &Tensor, gamma: &[f32], dim: usize) -> Result<Tensor> {
    let shape = x.shape().clone();
    let data = x
        .flatten_all()
        .map_err(|e| Error::Model(format!("rms_norm flatten: {e}")))?
        .to_vec1::<f32>()
        .map_err(|e| Error::Model(format!("rms_norm read: {e}")))?;
    let rows = data.len() / dim;
    debug_assert_eq!(rows * dim, data.len());
    let scale = (dim as f32).sqrt();
    let mut out = vec![0.0f32; data.len()];
    // 逐行并行（rayon）：行内归约顺序不变，数值与串行一致
    out.par_chunks_mut(dim).enumerate().for_each(|(row, chunk)| {
        let src = &data[row * dim..(row + 1) * dim];
        let mut sum = 0.0f32;
        for &v in src {
            sum += v * v;
        }
        let length = sum.sqrt().max(1e-12);
        for ((o, &v), &g) in chunk.iter_mut().zip(src).zip(gamma) {
            *o = v / length * scale * g;
        }
    });
    Tensor::from_vec(out, shape, x.device()).map_err(|e| Error::Model(format!("rms_norm: {e}")))
}

/// 相邻对 RoPE：x 布局 [b, heads, seq, dim]，dim 偶数；cos/sin 表 [seq, dim/2]。
fn rotate(x: &Tensor, cos: &[f32], sin: &[f32], seq: usize) -> Result<Tensor> {
    let shape = x.shape().clone();
    let dim = shape.dims()[3];
    let data = x
        .flatten_all()
        .map_err(|e| Error::Model(format!("rotate flatten: {e}")))?
        .to_vec1::<f32>()
        .map_err(|e| Error::Model(format!("rotate read: {e}")))?;
    let half = dim / 2;
    let mut out = data.clone();
    // 逐行并行（rayon），cos/sin 表按行只读
    out.par_chunks_mut(dim).enumerate().for_each(|(row_index, row)| {
        let offset = row_index % seq * half;
        for (pair_index, pair) in row.chunks_exact_mut(2).enumerate() {
            let c = cos[offset + pair_index];
            let s = sin[offset + pair_index];
            let even = pair[0];
            let odd = pair[1];
            pair[0] = even * c - odd * s;
            pair[1] = odd * c + even * s;
        }
    });
    Tensor::from_vec(out, shape, x.device()).map_err(|e| Error::Model(format!("rotate: {e}")))
}

// ---- 模型结构（对照 uvr roformer.rs）----

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
        Ok(Self {
            norm_gamma: st.vec1(&format!("{prefix}.norm.gamma"), DIM)?,
            qkv: st.linear(&format!("{prefix}.to_qkv"), DIM, 3 * DIM, false)?,
            gates: st.linear(&format!("{prefix}.to_gates"), DIM, HEADS, true)?,
            out: st.linear(&format!("{prefix}.to_out.0"), DIM, DIM, false)?,
            cos,
            sin,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [b, seq, DIM]（seq = 帧数或频带数）
        let batch = x.dims()[0];
        let seq = x.dims()[1];
        let xn = rms_norm(x, &self.norm_gamma, DIM)?;
        let projected = linear_forward(&self.qkv, &xn)?; // [b, seq, 3*DIM]
        let mut head = |start: usize| -> Result<Tensor> {
            projected
                .narrow(2, start, DIM)
                .map_err(|e| Error::Model(format!("qkv narrow: {e}")))?
                .reshape((batch, seq, HEADS, HEAD_DIM))
                .map_err(|e| Error::Model(format!("qkv reshape: {e}")))?
                .permute((0, 2, 1, 3)) // [b, heads, seq, head_dim]
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
        let k = rope(head(DIM)?)?;
        let v = head(2 * DIM)?.contiguous().map_err(|e| Error::Model(format!("v contig: {e}")))?;
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
            .reshape((batch, seq, DIM))
            .map_err(|e| Error::Model(format!("out reshape: {e}")))?;
        linear_forward(&self.out, &values)
    }
}

struct Transformer {
    attention: Attention,
    norm_gamma: Vec<f32>,
    first: Linear,
    last: Linear,
}

impl Transformer {
    fn load(st: &Safetensors, prefix: &str, sequence: usize) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(st, &format!("{prefix}.0"), sequence)?,
            norm_gamma: st.vec1(&format!("{prefix}.1.net.0.gamma"), DIM)?,
            first: st.linear(&format!("{prefix}.1.net.1"), DIM, 4 * DIM, true)?,
            last: st.linear(&format!("{prefix}.1.net.4"), 4 * DIM, DIM, true)?,
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
        let norm = rms_norm(&x, &self.norm_gamma, DIM)?;
        let first = linear_forward(&self.first, &norm)?;
        let gelu = first
            .gelu_erf()
            .map_err(|e| Error::Model(format!("gelu: {e}")))?;
        let last = linear_forward(&self.last, &gelu)?;
        last.add(&x).map_err(|e| Error::Model(format!("ffn residual: {e}")))
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

struct MaskEstimator {
    first: Linear,
    last: Linear,
}

impl MaskEstimator {
    fn load(st: &Safetensors, prefix: &str, input: usize) -> Result<Self> {
        Ok(Self {
            first: st.linear(&format!("{prefix}.0.0"), DIM, 4 * DIM, true)?,
            last: st.linear(&format!("{prefix}.0.2"), 4 * DIM, 2 * input, true)?,
        })
    }

    /// 输入 [1,1,frames,DIM]，输出 [frames, 2*input]。
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x
            .squeeze(0)
            .map_err(|e| Error::Model(format!("mask squeeze: {e}")))?
            .squeeze(0)
            .map_err(|e| Error::Model(format!("mask squeeze: {e}")))?; // [frames, DIM]
        let x = linear_forward(&self.first, &x)?
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

pub struct BsRoformer {
    bands: Vec<(Vec<f32>, Linear)>, // (norm gamma, projection linear)，input = 4*bins
    layers: Vec<[Transformer; 2]>,
    final_norm: Vec<f32>,
    masks: Vec<MaskEstimator>,
    options: RoformerOptions,
}

impl BsRoformer {
    pub fn load(path: &std::path::Path, options: RoformerOptions) -> Result<Self> {
        let device = Device::Cpu;
        let st = Safetensors::load(path, &device)?;
        let mut consumed = 0usize;

        let mut bands = Vec::with_capacity(BANDS.len());
        let mut masks = Vec::with_capacity(BANDS.len());
        for (index, &bins) in BANDS.iter().enumerate() {
            let input = 4 * bins;
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
            consumed += 4; // first weight/bias + last weight/bias
        }
        let mut layers = Vec::with_capacity(DEPTH);
        for layer in 0..DEPTH {
            layers.push([
                Transformer::load(
                    &st,
                    &format!("layers.{layer}.0.layers.0"),
                    CHUNK / HOP + 1,
                )?,
                Transformer::load(&st, &format!("layers.{layer}.1.layers.0"), BANDS.len())?,
            ]);
            consumed += 2 * 11; // attention 6 + ffn 5
        }
        let final_norm = st.vec1("final_norm.gamma", DIM)?;
        consumed += 1;

        let total = st.len();
        if consumed != total {
            return Err(Error::Model(format!(
                "权重消费数不符: 期望 {total}, 已消费 {consumed}"
            )));
        }
        Ok(Self {
            bands,
            layers,
            final_norm,
            masks,
            options,
        })
    }

    pub fn consumed_tensors(&self) -> usize {
        3 * BANDS.len() + 4 * BANDS.len() + 2 * DEPTH * 11 + 1
    }

    // 剖析/测试访问器
    #[doc(hidden)]
    pub fn band_gamma(&self, i: usize) -> &[f32] {
        &self.bands[i].0
    }
    #[doc(hidden)]
    pub fn band_linear(&self, i: usize) -> &Linear {
        &self.bands[i].1
    }
    #[doc(hidden)]
    pub fn final_norm(&self) -> &[f32] {
        &self.final_norm
    }
    #[doc(hidden)]
    pub fn mask_forward(&self, i: usize, x: &Tensor) -> Result<Tensor> {
        self.masks[i].forward(x)
    }

    /// 分阶段计时剖析（8 秒窗口），返回 (stft, band, layers, mask) 各阶段秒数。
    #[doc(hidden)]
    pub fn profile(&self, audio: &[f32], samples: usize) -> Result<Vec<f64>> {
        use std::time::Instant;
        let frames = samples / HOP + 1;
        let mut timings = Vec::new();

        let t0 = Instant::now();
        let mut prec = PreciseStft::new()?;
        let mut spectra = Vec::with_capacity(2);
        for channel in audio.chunks_exact(samples) {
            spectra.push(prec.forward(channel)?);
        }
        timings.push(t0.elapsed().as_secs_f64());

        let t1 = Instant::now();
        let device = Device::Cpu;
        let mut features: Vec<Tensor> = Vec::with_capacity(BANDS.len());
        let mut first_bin = 0;
        for ((gamma, linear), &bins) in self.bands.iter().zip(&BANDS) {
            let input = 4 * bins;
            let mut values = Vec::with_capacity(frames * input);
            for frame in 0..frames {
                for bin in first_bin..first_bin + bins {
                    for spectrum in &spectra {
                        let v = spectrum[bin * frames + frame];
                        values.push(v.re);
                        values.push(v.im);
                    }
                }
            }
            let x = Tensor::from_vec(values, (1, 1, frames, input), &device)
                .map_err(|e| Error::Model(format!("band input: {e}")))?;
            let x = rms_norm(&x, gamma, input)?;
            features.push(linear_forward(linear, &x)?);
            first_bin += bins;
        }
        timings.push(t1.elapsed().as_secs_f64());

        let t2 = Instant::now();
        let (time_batch, frequency_batch) = self.batch_sizes();
        let mut x = Tensor::cat(&features, 1).map_err(|e| Error::Model(format!("band cat: {e}")))?;
        for [time, frequency] in &self.layers {
            x = time.forward_batches(&x, time_batch)?;
            let swapped = x.permute((0, 2, 1, 3))?;
            x = frequency.forward_batches(&swapped, frequency_batch)?.permute((0, 2, 1, 3))?;
        }
        timings.push(t2.elapsed().as_secs_f64());

        let t3 = Instant::now();
        x = rms_norm(&x, &self.final_norm, DIM)?;
        for (index, estimator) in self.masks.iter().enumerate() {
            let band_slice = x.narrow(1, index, 1)?;
            let mask = estimator.forward(&band_slice)?;
            mask.flatten_all()?.to_vec1::<f32>()?;
        }
        timings.push(t3.elapsed().as_secs_f64());
        Ok(timings)
    }

    pub fn batch_sizes(&self) -> (usize, usize) {
        (self.options.time_batch, self.options.frequency_batch)
    }

    /// 立体声窗口（**平面布局**：前 samples 个为左声道，后 samples 个为右声道），
    /// samples ∈ (FFT/2, CHUNK]。
    /// 输出 2*(samples/HOP*HOP)，同样平面布局（前一半左声道、后一半右声道）。
    pub fn predict_window(&self, audio: &[f32], samples: usize) -> Result<Vec<f32>> {
        if samples <= FFT / 2 || samples > CHUNK || audio.len() != 2 * samples {
            return Err(Error::Model(format!(
                "bs_roformer 期望立体声窗口 1025..={samples}<=352800, 实际 {}",
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

        // 2) 频带投影
        let device = Device::Cpu;
        let mut features: Vec<Tensor> = Vec::with_capacity(BANDS.len());
        let mut first_bin = 0;
        for ((gamma, linear), &bins) in self.bands.iter().zip(&BANDS) {
            let input = 4 * bins;
            let mut values = Vec::with_capacity(frames * input);
            for frame in 0..frames {
                for bin in first_bin..first_bin + bins {
                    for spectrum in &spectra {
                        let v = spectrum[bin * frames + frame];
                        values.push(v.re);
                        values.push(v.im);
                    }
                }
            }
            let x = Tensor::from_vec(values, (1, 1, frames, input), &device)
                .map_err(|e| Error::Model(format!("band input: {e}")))?;
            let x = rms_norm(&x, gamma, input)?;
            features.push(linear_forward(linear, &x)?);
            first_bin += bins;
        }
        // 3) 12 层 time/freq transformer
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
        // 4) final norm
        x = rms_norm(&x, &self.final_norm, DIM)?;

        // 5) mask 估计与应用
        let bins = FFT / 2 + 1;
        let mut masked: Vec<Vec<Complex32>> = (0..2)
            .map(|_| vec![Complex32::default(); bins * frames])
            .collect();
        first_bin = 0;
        for (index, (&bins_band, estimator)) in BANDS.iter().zip(&self.masks).enumerate() {
            let band_slice = x
                .narrow(1, index, 1)
                .map_err(|e| Error::Model(format!("band narrow: {e}")))?; // [1, 1, frames, DIM]
            let mask = estimator.forward(&band_slice)?; // [frames, 2*input]
            let mask = mask
                .flatten_all()
                .map_err(|e| Error::Model(format!("mask flatten: {e}")))?
                .to_vec1::<f32>()
                .map_err(|e| Error::Model(format!("mask read: {e}")))?;
            if mask.iter().any(|v| !v.is_finite()) {
                return Err(Error::Model("Roformer mask 含非有限值".into()));
            }
            for frame in 0..frames {
                for bin in 0..bins_band {
                    for channel in 0..2 {
                        let i = frame * bins_band * 4 + bin * 4 + channel * 2;
                        let position = (first_bin + bin) * frames + frame;
                        masked[channel][position] =
                            spectra[channel][position] * Complex32::new(mask[i], mask[i + 1]);
                    }
                }
            }
            first_bin += bins_band;
        }

        // 6) iSTFT 每声道
        let output_samples = samples / HOP * HOP;
        let mut stft = Stft::new()?;
        let mut output = Vec::with_capacity(2 * output_samples);
        for spectrum in &masked {
            output.extend(stft.inverse(spectrum, frames)?);
        }
        if output.len() != 2 * output_samples || output.iter().any(|v| !v.is_finite()) {
            return Err(Error::Model("Roformer 输出波形无效".into()));
        }
        Ok(output)
    }

    /// 完整分离：任意采样率交错音频 → (vocals, instrumental)，均为交织立体声。
    /// 调度同参考实现：重采样到 44.1k → 8s 窗口（步长 2s）→ Hamming 窗 overlap-add。
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
        // 平面声道视图
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
            for (channel, source) in audio.iter().enumerate() {
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
