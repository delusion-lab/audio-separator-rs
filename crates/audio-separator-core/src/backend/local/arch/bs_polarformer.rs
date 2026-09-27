//! BS-PolarFormer（candle CPU FP32 前向）。
//!
//! 数学与权重名映射逐项对照 hunterFormsBS BandSplitRotator 0.3.1 +
//! PoPE-pytorch（lucidrains）：62 频带、DIM=256 / HEADS=8 / HEAD_DIM=64 / DEPTH=12、
//! FFT=2048 / HOP=512、**PoPE（Polar Coordinate Positional Embedding）替代 RoPE**：
//! softplus 幅度 → 极坐标旋转（q 用频相位、k 用频相位+可学习 key bias，clamp[-2π,0]）
//! → dim 翻倍拼接；mask_estimator_depth 等效 1（两层 Linear + Tanh + GLU，
//! 与权重 `to_freqs.{b}.0.{0,2}` 一致——hunterFormsBS 文档说明早期权重
//! "mask_estimator_depth: 2" 实际存储两层）；无每层输出 norm（norm_output=False）；
//! 有 final_norm。权重来自 M2-D 转换器（float16 ckpt → f32 safetensors，723 个 tensor）。
//!
//! 输入：立体声窗口 `2*samples`（交错），samples ∈ (1024, 352768] 且被 512 整除。
//! 输出：`2 * (samples/HOP*HOP)`，前一半左声道、后一半右声道（与参考一致）。

use std::collections::HashMap;

use rayon::prelude::*;

use candle_core::{Device, Tensor};
use candle_nn::Linear;
use rustfft::num_complex::Complex32;

use crate::error::{Error, Result};

use super::roformer_stft::{FFT, PreciseStft, Stft};

pub const DIM: usize = 256;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const DEPTH: usize = 12;
pub const HOP: usize = 512;
pub const SAMPLE_RATE: u32 = 44_100;
/// 预测窗口 = 512 * 689（8s，帧数 690，与 bs/mel 同量级；保证 samples/HOP 整除）。
pub const CHUNK: usize = 352_768;

/// 62 个频带宽度，总和 = 1025 = FFT/2+1（与 bs_roformer 相同布局）。
pub const BANDS: [usize; 62] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48,
    48, 48, 48, 128, 129,
];

pub struct RoformerOptions {
    /// 时间注意力每批处理的频带数（默认 62 全量）。
    pub time_batch: usize,
    /// 频率注意力每批处理的帧数（默认 301）。
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
            .map_err(|e| Error::Model(format!("failed to read safetensors: {e}")))?;
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])
            .map_err(|e| Error::Model(format!("failed to parse safetensors header: {e}")))?;
        let data = &bytes[8 + header_len..];
        let mut values = HashMap::with_capacity(header.as_object().map_or(0, |o| o.len()));
        for (name, meta) in header
            .as_object()
            .ok_or_else(|| Error::Model("header is not an object".into()))?
        {
            let dtype = meta["dtype"]
                .as_str()
                .ok_or_else(|| Error::Model("missing dtype".into()))?;
            if dtype != "F32" {
                return Err(Error::Model(format!("{name}: only F32 supported, got {dtype}")));
            }
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .ok_or_else(|| Error::Model("missing shape".into()))?
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let offsets = meta["data_offsets"]
                .as_array()
                .ok_or_else(|| Error::Model("missing offsets".into()))?;
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            let raw = &data[start..end];
            if raw.len() != shape.iter().product::<usize>() * 4 {
                return Err(Error::Model(format!("{name}: data length mismatch")));
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

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.values
            .get(name)
            .ok_or_else(|| Error::Model(format!("missing weight: {name}")))
    }

    /// 读取并校验形状，转为 f32 向量（用于 norm/pope 等标量参数）。
    pub(crate) fn vec1(&self, name: &str, expected: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dims() != [expected] {
            return Err(Error::Model(format!("{name}: shape mismatch {:?}", t.dims())));
        }
        let v = t.to_vec1::<f32>()?;
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::Model(format!("{name}: weight contains non-finite values")));
        }
        Ok(v)
    }

    /// 读取二维矩阵参数并转为 f32 向量（pope bias [heads, dim]）。
    pub(crate) fn vec2(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dims() != [rows, cols] {
            return Err(Error::Model(format!("{name}: shape mismatch {:?}", t.dims())));
        }
        let v = t.flatten_all()?.to_vec1::<f32>()?;
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::Model(format!("{name}: weight contains non-finite values")));
        }
        Ok(v)
    }

    pub(crate) fn linear(&self, prefix: &str, input: usize, output: usize, bias: bool) -> Result<Linear> {
        let weight = self.tensor(&format!("{prefix}.weight"))?;
        if weight.dims() != [output, input] {
            return Err(Error::Model(format!(
                "{prefix}.weight: shape mismatch {:?}",
                weight.dims()
            )));
        }
        let bias_t = if bias {
            let b = self.tensor(&format!("{prefix}.bias"))?;
            if b.dims() != [output] {
                return Err(Error::Model(format!("{prefix}.bias: shape mismatch")));
            }
            Some(b.clone())
        } else {
            None
        };
        Ok(Linear::new(weight.clone(), bias_t))
    }
}

/// candle_nn 0.9 的 Linear 无 forward，手动实现：x @ W^T + b。
/// 与 bs_roformer.rs 的 linear_forward 相同约定（Flattened 布局）。
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

/// RMSNorm 变体：x / sqrt(sum(x²)).clamp_min(1e-12) * sqrt(dim) * γ，作用于最后一维。
/// 与 bs_roformer.rs 相同约定。
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

/// softplus（数值稳定）。
fn softplus(v: f32) -> f32 {
    if v <= 0.0 {
        (v.exp()).ln_1p()
    } else {
        v + (-v).exp().ln_1p()
    }
}

/// PoPE 极坐标旋转：softplus(x) 乘 cos/sin 表 → 拼接 dim 翻倍。
/// `t` 布局 [b, 1, seq, dim]，`cos/sin` 为 [seq*dim] 平铺（每行 seq 个、每行 dim 列）。
/// 输出 [b, 1, seq, 2*dim]（与 PoPE_pytorch `apply_pope_to_qk` 的
/// `[q*qcos, q*qsin] -> (d two)` 拼接一致）。
fn pope_rotate(t: &Tensor, cos: &[f32], sin: &[f32], seq: usize, dim: usize) -> Result<Tensor> {
    let shape = t.shape().clone();
    let data = t
        .flatten_all()
        .map_err(|e| Error::Model(format!("pope flatten: {e}")))?
        .to_vec1::<f32>()
        .map_err(|e| Error::Model(format!("pope read: {e}")))?;
    let rows = data.len() / dim;
    debug_assert_eq!(rows * dim, data.len());
    let mut out = vec![0.0f32; rows * dim * 2];
    out.par_chunks_mut(dim * 2)
        .enumerate()
        .for_each(|(row, chunk)| {
            let src = &data[row * dim..(row + 1) * dim];
            let base = (row % seq) * dim;
            let (cos_row, sin_row) = (&cos[base..base + dim], &sin[base..base + dim]);
            for (j, &v) in src.iter().enumerate() {
                let m = softplus(v);
                chunk[j] = m * cos_row[j];
                chunk[dim + j] = m * sin_row[j];
            }
        });
    // 输出维度翻倍：shape 尾维必须改为 2*dim（否则 candle 只取前 shape 元素，sin 部分丢失）
    let mut out_shape = shape.dims().to_vec();
    *out_shape.last_mut().unwrap() = dim * 2;
    Tensor::from_vec(out, out_shape, t.device()).map_err(|e| Error::Model(format!("pope: {e}")))
}

// ---- 模型结构（对照 hunterFormsBS bandSplitRotator）----

#[doc(hidden)]
pub struct Attention {
    norm_gamma: Vec<f32>,
    qkv: Linear, // [DIM → 3*dim_inner]，dim_inner = HEADS*HEAD_DIM = 512
    gates: Linear,
    out: Linear, // [dim_inner → DIM]
    /// PoPE 逆频率 θ^(-j/dim)（每层每轴独立权重），长度 HEAD_DIM。
    inv_freqs: Vec<f32>,
    /// PoPE key 可学习相位偏置 [HEADS, HEAD_DIM]，前向时 clamp[-2π, 0]。
    bias: Vec<f32>,
}

impl Attention {
    fn load(st: &Safetensors, prefix: &str) -> Result<Self> {
        let dim_inner = HEADS * HEAD_DIM;
        Ok(Self {
            norm_gamma: st.vec1(&format!("{prefix}.norm.gamma"), DIM)?,
            qkv: st.linear(&format!("{prefix}.to_qkv"), DIM, 3 * dim_inner, false)?,
            gates: st.linear(&format!("{prefix}.to_gates"), DIM, HEADS, true)?,
            out: st.linear(&format!("{prefix}.to_out.0"), dim_inner, DIM, false)?,
            inv_freqs: st.vec1(&format!("{prefix}.pope_embed.inv_freqs"), HEAD_DIM)?,
            bias: st.vec2(&format!("{prefix}.pope_embed.bias"), HEADS, HEAD_DIM)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [b, seq, DIM]
        let batch = x.dims()[0];
        let seq = x.dims()[1];
        let dim_inner = HEADS * HEAD_DIM;
        let xn = rms_norm(x, &self.norm_gamma, DIM)?;
        let projected = linear_forward(&self.qkv, &xn)?; // [b, seq, 3*dim_inner]
        let head = |start: usize| -> Result<Tensor> {
            projected
                .narrow(2, start, dim_inner)
                .map_err(|e| Error::Model(format!("qkv narrow: {e}")))?
                .reshape((batch, seq, HEADS, HEAD_DIM))
                .map_err(|e| Error::Model(format!("qkv reshape: {e}")))?
                .permute((0, 2, 1, 3)) // [b, heads, seq, head_dim]
                .map_err(|e| Error::Model(format!("qkv permute: {e}")))
        };
        let q = head(0)?;
        let k = head(dim_inner)?;
        let v = head(2 * dim_inner)?
            .contiguous()
            .map_err(|e| Error::Model(format!("v contig: {e}")))?;
        #[allow(unused_mut)]
        let dump_enabled = std::env::var("ASEP_DUMP_ATTN").is_ok();
        let dump_id = if dump_enabled {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static ID: AtomicUsize = AtomicUsize::new(0);
            let id = ID.fetch_add(1, Ordering::SeqCst);
            let dn = move |name: &str, t: &Tensor| {
                let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let mut bin = Vec::new();
                for x in v {
                    bin.extend_from_slice(&x.to_le_bytes());
                }
                std::fs::write(format!("test-assets/candle_attn_{id}_{name}.npy"), bin).unwrap();
            };
            let _ = dn("xn", &xn);
            let _ = dn("q", &q);
            let _ = dn("k", &k);
            let _ = dn("v", &v);
            Some(dn)
        } else {
            None
        };

        // PoPE：freqs[i][j] = i * inv_freqs[j]；q 旋转用纯相位，
        // k 旋转用相位 + 每头可学习 bias（clamp[-2π, 0]）。
        let mut cos = Vec::with_capacity(seq * HEAD_DIM);
        let mut sin = Vec::with_capacity(seq * HEAD_DIM);
        for index in 0..seq {
            for &f in &self.inv_freqs {
                let angle = index as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let mut out_heads: Vec<Tensor> = Vec::with_capacity(HEADS);
        for h in 0..HEADS {
            let qh = q.narrow(1, h, 1)?;
            let kh = k.narrow(1, h, 1)?;
            let vh = v.narrow(1, h, 1)?;
            let qp = pope_rotate(&qh, &cos, &sin, seq, HEAD_DIM)?; // [b,1,seq,128]
            // k 侧相位 = freqs + bias[h]（clamped）
            let mut cos_b = Vec::with_capacity(seq * HEAD_DIM);
            let mut sin_b = Vec::with_capacity(seq * HEAD_DIM);
            for index in 0..seq {
                for j in 0..HEAD_DIM {
                    let phase = index as f32 * self.inv_freqs[j] + self.bias[h * HEAD_DIM + j].clamp(-2.0 * std::f32::consts::PI, 0.0);
                    cos_b.push(phase.cos());
                    sin_b.push(phase.sin());
                }
            }
            let kp = pope_rotate(&kh, &cos_b, &sin_b, seq, HEAD_DIM)?;
            let kp_t = kp
                .transpose(2, 3)
                .map_err(|e| Error::Model(format!("k transpose: {e}")))?
                .contiguous()
                .map_err(|e| Error::Model(format!("k contig: {e}")))?;
            let scores = qp
                .matmul(&kp_t)
                .map_err(|e| Error::Model(format!("scores matmul: {e}")))?
                .affine((HEAD_DIM as f64).powf(-0.5), 0.0)
                .map_err(|e| Error::Model(format!("scores scale: {e}")))?;
            let probs = candle_nn::ops::softmax(&scores, 3)
                .map_err(|e| Error::Model(format!("softmax: {e}")))?;
            if let Some(dn) = &dump_id {
                let _ = dn(&format!("qp_{h}"), &qp);
                let _ = dn(&format!("kp_{h}"), &kp);
                let _ = dn(&format!("scores_{h}"), &scores);
                let _ = dn(&format!("probs_{h}"), &probs);
            }
            let out_h = probs
                .matmul(&vh)
                .map_err(|e| Error::Model(format!("values matmul: {e}")))?;
            out_heads.push(out_h);
        }
        let values = Tensor::cat(&out_heads, 1).map_err(|e| Error::Model(format!("cat: {e}")))?;
        let gates = candle_nn::ops::sigmoid(&linear_forward(&self.gates, &xn)?)
            .map_err(|e| Error::Model(format!("gates sigmoid: {e}")))?
            .reshape((batch, seq, HEADS, 1))
            .map_err(|e| Error::Model(format!("gates reshape: {e}")))?
            .permute((0, 2, 1, 3))
            .map_err(|e| Error::Model(format!("gates permute: {e}")))?;
        let values = values
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

#[doc(hidden)]
pub struct Transformer {
    attention: Attention,
    norm_gamma: Vec<f32>,
    first: Linear,
    last: Linear,
}

impl Transformer {
    fn load(st: &Safetensors, prefix: &str) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(st, &format!("{prefix}.0"))?,
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

#[doc(hidden)]
pub struct MaskEstimator {
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

pub struct BsPolarformer {
    bands: Vec<(Vec<f32>, Linear)>, // (norm gamma, projection linear)，input = 4*bins
    layers: Vec<[Transformer; 2]>,
    final_norm: Vec<f32>,
    masks: Vec<MaskEstimator>,
    options: RoformerOptions,
}

impl BsPolarformer {
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
                Transformer::load(&st, &format!("layers.{layer}.0.layers.0"))?,
                Transformer::load(&st, &format!("layers.{layer}.1.layers.0"))?,
            ]);
            consumed += 2 * 12; // attention 7（含 pope 2）+ ffn 5
        }
        let final_norm = st.vec1("final_norm.gamma", DIM)?;
        consumed += 1;

        let total = st.len();
        if consumed != total {
            return Err(Error::Model(format!(
                "weight consumption mismatch: expected {total}, consumed {consumed}"
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
        3 * BANDS.len() + 4 * BANDS.len() + 2 * DEPTH * 12 + 1
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
    #[doc(hidden)]
    pub fn layers(&self) -> &Vec<[Transformer; 2]> {
        &self.layers
    }
    #[doc(hidden)]
    pub fn bands(&self) -> &Vec<(Vec<f32>, Linear)> {
        &self.bands
    }

    pub fn batch_sizes(&self) -> (usize, usize) {
        (self.options.time_batch, self.options.frequency_batch)
    }

    /// 立体声窗口（**平面布局**：前 samples 个为左声道，后 samples 个为右声道），
    /// samples ∈ (FFT/2, CHUNK] 且被 HOP 整除。
    /// 输出 2*(samples/HOP*HOP)，同样平面布局。
    pub fn predict_window(&self, audio: &[f32], samples: usize) -> Result<Vec<f32>> {
        if samples <= FFT / 2 || samples > CHUNK || samples % HOP != 0 || audio.len() != 2 * samples {
            return Err(Error::Model(format!(
                "bs_polarformer expects stereo window (samples divisible by {HOP}) 1025..={samples}<={CHUNK}, got {}",
                audio.len()
            )));
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(Error::Model("audio contains non-finite values".into()));
        }
        let frames = samples / HOP + 1;
        let (time_batch, frequency_batch) = self.batch_sizes();

        // 1) f64 STFT 每声道
        let mut prec = PreciseStft::new(HOP)?;
        let mut spectra = Vec::with_capacity(2);
        for channel in audio.chunks_exact(samples) {
            let spec = prec.forward(channel)?;
            if spec.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
                return Err(Error::Model("Roformer spectrum contains non-finite values".into()));
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
        // features cat 后布局为 [1, bands, frames, DIM]（band 前、frame 后），与参考一致：
        // time 注意力在 frame 轴（batch=bands、seq=frames，直接用该布局）；
        // freq 注意力在 band 轴（batch=frames、seq=bands，需转置后恢复）。
        let mut x = Tensor::cat(&features, 1).map_err(|e| Error::Model(format!("band cat: {e}")))?;
        for [time, frequency] in &self.layers {
            x = time.forward_batches(&x, time_batch)?;
            let freq_in = x
                .permute((0, 2, 1, 3))
                .map_err(|e| Error::Model(format!("swap dims failed: {e}")))?; // [1, frames, bands, DIM]
            x = frequency.forward_batches(&freq_in, frequency_batch)?;
            x = x
                .permute((0, 2, 1, 3))
                .map_err(|e| Error::Model(format!("swap dims failed: {e}")))?; // 回 [1, bands, frames, DIM]
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
                return Err(Error::Model("Roformer mask contains non-finite values".into()));
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
        let mut stft = Stft::new(HOP)?;
        let mut output = Vec::with_capacity(2 * output_samples);
        for spectrum in &masked {
            output.extend(stft.inverse(spectrum, frames)?);
        }
        if output.len() != 2 * output_samples || output.iter().any(|v| !v.is_finite()) {
            return Err(Error::Model("Roformer output waveform invalid".into()));
        }
        Ok(output)
    }

    /// 完整分离：任意采样率交错音频 → (vocals, instrumental)，均为交织立体声。
    /// 调度同参考实现：重采样到 44.1k → 8s 窗口（步长 2s）→ Hamming 窗 overlap-add。
    pub fn separate(&self, samples: &[f32], sample_rate: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        if sample_rate != SAMPLE_RATE {
            let resampled = crate::io::resample_to(samples, 2, sample_rate, SAMPLE_RATE)
                .map_err(|e| Error::Backend(format!("resampling failed: {e}")))?;
            self.separate_44100(&resampled)
        } else {
            self.separate_44100(samples)
        }
    }

    fn separate_44100(&self, samples: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if samples.len() % 2 != 0 {
            return Err(Error::Model("interleaved audio length must be even".into()));
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
                    return Err(Error::Model("Roformer schedule window did not cover the samples".into()));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumed_matches_723() {
        assert_eq!(
            3 * BANDS.len() + 4 * BANDS.len() + 2 * DEPTH * 12 + 1,
            723
        );
    }

    #[test]
    fn softplus_numeric() {
        assert_eq!(softplus(0.0), (1.0f32).ln_1p());
        assert_eq!(softplus(10.0), 10.0 + (-10.0f32).exp().ln_1p());
        assert_eq!(softplus(-5.0), (-5.0f32).exp().ln_1p());
        assert!(softplus(100.0).is_finite());
    }
}
