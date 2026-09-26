//! MDX-Net 架构推理（UVR-MDX-NET 系列，2 分轨：vocals / instrumental）。
//!
//! 管线：立体声 f32 → 分 segment（首尾交叉淡化）→ 每段 STFT（realfft，hann 窗，center pad）
//! → 输入 `[1,4,freq,frames]`（布局 r0,r1,i0,i1）→ ONNX 输出复数 mask
//! → 复数 mask 乘原频谱 → iSTFT（WOLA，除以 ∑win²）→ 裁剪 center pad → 叠加 → 权重归一化。
//!
//! 移植参考：UVR / audio-separator（MIT）的 MDX 推理逻辑；输出形状与参数口径在真机验证后校准。

use ndarray::Array4;
use num_complex::Complex;
use realfft::RealFftPlanner;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{Error, Result};
use crate::job::ProgressEvent;
use crate::model::ModelEntry;

use super::super::engine::OnnxSession;

/// MDX 架构参数（audio-separator / UVR 兼容口径）。
///
/// 优先级：模型 metadata（UVR 转换模型自带 n_fft/hop_length/dim_f/dim_t/sample_rate）
/// > manifest params > 默认值。参数由模型自描述，manifest 仅作 fallback/覆盖。
#[derive(Debug, Clone)]
pub struct MdxParams {
    /// 模型采样率（输入重采样到此）。
    pub sample_rate: u32,
    /// segment 长度（秒，UVR 口径；当 dim_t>0 时以 dim_t 帧数为准）。
    pub segment_secs: usize,
    /// segment 间重叠比例（0-1）。
    pub overlap: f32,
    /// 批大小（当前实现恒 1，仅校验）。
    pub batch_size: usize,
    /// STFT hop。
    pub hop: usize,
    /// STFT FFT 大小。
    pub n_fft: usize,
    /// 模型输入频率 bins（= n_fft/2，不含 Nyquist bin）。
    pub dim_f: usize,
    /// 模型输入帧数（固定 256 的 UVR 模型；0=由 segment 决定）。
    pub dim_t: usize,
}

impl Default for MdxParams {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            segment_secs: 256,
            overlap: 0.25,
            batch_size: 1,
            hop: 1024,
            n_fft: 6144,
            dim_f: 3072,
            dim_t: 256,
        }
    }
}

impl MdxParams {
    /// 从模型条目参数解析；缺失字段用默认值。
    pub fn from_entry(entry: Option<&ModelEntry>) -> Result<Self> {
        let mut p = Self::default();
        let Some(entry) = entry else {
            return Ok(p);
        };
        let v = &entry.params;
        if let Some(x) = v.get("sample_rate").and_then(|x| x.as_u64()) {
            p.sample_rate = x as u32;
        }
        if let Some(x) = v.get("segment_size").and_then(|x| x.as_u64()) {
            p.segment_secs = x as usize;
        }
        if let Some(x) = v.get("overlap").and_then(|x| x.as_f64()) {
            p.overlap = x as f32;
        }
        if let Some(x) = v.get("batch_size").and_then(|x| x.as_u64()) {
            p.batch_size = x as usize;
        }
        if let Some(x) = v
            .get("hop")
            .or_else(|| v.get("hop_length"))
            .and_then(|x| x.as_u64())
        {
            p.hop = x as usize;
        }
        if let Some(x) = v.get("n_fft").and_then(|x| x.as_u64()) {
            p.n_fft = x as usize;
        }
        if let Some(x) = v.get("dim_f").and_then(|x| x.as_u64()) {
            p.dim_f = x as usize;
        }
        if let Some(x) = v.get("dim_t").and_then(|x| x.as_u64()) {
            p.dim_t = x as usize;
        }
        p.validate()?;
        Ok(p)
    }

    /// 用模型 metadata 覆盖 manifest/默认参数（metadata 为权威）。
    pub fn apply_metadata(&mut self, get: &dyn Fn(&str) -> Option<String>) -> Result<()> {
        if let Some(v) = get("sample_rate") {
            if let Ok(x) = v.parse::<u32>() {
                self.sample_rate = x;
            }
        }
        if let Some(v) = get("hop_length") {
            if let Ok(x) = v.parse::<usize>() {
                self.hop = x;
            }
        }
        if let Some(v) = get("n_fft") {
            if let Ok(x) = v.parse::<usize>() {
                self.n_fft = x;
            }
        }
        if let Some(v) = get("dim_f") {
            if let Ok(x) = v.parse::<usize>() {
                self.dim_f = x;
            }
        }
        if let Some(v) = get("dim_t") {
            if let Ok(x) = v.parse::<usize>() {
                self.dim_t = x;
            }
        }
        self.validate()
    }

    fn validate(&mut self) -> Result<()> {
        if self.overlap < 0.0 || self.overlap >= 1.0 {
            return Err(Error::Model(format!("mdx 参数 overlap 非法: {}", self.overlap)));
        }
        if self.n_fft < 8 || self.hop < 1 || self.hop >= self.n_fft {
            return Err(Error::Model(format!(
                "mdx 参数非法: n_fft={} hop={}",
                self.n_fft, self.hop
            )));
        }
        if self.dim_f == 0 || self.dim_f != self.n_fft / 2 {
            // UVR 系列模型输入不含 Nyquist bin：dim_f 应为 n_fft/2。
            self.dim_f = self.n_fft / 2;
        }
        Ok(())
    }

    /// segment 实际样本数：dim_t>0 时按 (dim_t-1)*hop 固定帧对齐。
    pub fn segment_samples(&self) -> usize {
        if self.dim_t > 0 {
            return (self.dim_t - 1) * self.hop;
        }
        let samples = self.segment_secs.saturating_mul(self.sample_rate as usize);
        ((samples + self.hop - 1) / self.hop) * self.hop
    }
}

/// 对立体声交织采样执行 MDX 分离，返回 `[(stem 名, 交织采样)]`。
pub fn separate_mdx(
    session: &mut OnnxSession,
    input: &[f32],
    params: &MdxParams,
    progress: Option<&mpsc::Sender<ProgressEvent>>,
    cancel: Option<&CancellationToken>,
) -> Result<Vec<(String, Vec<f32>)>> {
    const CH: usize = 2;
    let frames_total = input.len() / CH;
    if frames_total == 0 {
        return Err(Error::Format("输入音频为空".to_string()));
    }
    let seg = params.segment_samples();
    let hop_seg = ((seg as f32 * (1.0 - params.overlap)).max(1.0)) as usize;
    let n_segments = if frames_total <= seg {
        1
    } else {
        (frames_total - seg).div_ceil(hop_seg) + 1
    };
    let overlap_len = ((seg as f32 * params.overlap) as usize).min(seg / 2);

    let mut vocals = vec![0.0f32; frames_total * CH];
    let mut inst = vec![0.0f32; frames_total * CH];
    let mut weight = vec![0.0f32; frames_total];

    for s in 0..n_segments {
        let start = s * hop_seg;
        if start >= frames_total {
            break;
        }
        let end = (start + seg).min(frames_total);

        // segment 拷贝（不足补零）
        let mut seg_samples = vec![0.0f32; seg * CH];
        for f in start..end {
            seg_samples[(f - start) * CH] = input[f * CH];
            seg_samples[(f - start) * CH + 1] = input[f * CH + 1];
        }

        // STFT → [ch][frames][freq] 复数（freq = n_fft/2+1，含 Nyquist bin）
        let spec = stft(&seg_samples, CH, params.n_fft, params.hop)?;
        let freq = spec[0][0].len();
        let frames = spec[0].len();
        let dim_f = params.dim_f; // 模型输入频率 bins（= n_fft/2，不含 Nyquist）

        // 模型输入 [1,4,dim_f,frames]（布局 r0,r1,i0,i1；丢弃 Nyquist bin）
        let mut arr = Array4::<f32>::zeros((1, 4, dim_f, frames));
        for f in 0..frames {
            for k in 0..dim_f {
                let c0 = spec[0][f][k];
                let c1 = spec[1][f][k];
                arr[[0, 0, k, f]] = c0.re;
                arr[[0, 1, k, f]] = c1.re;
                arr[[0, 2, k, f]] = c0.im;
                arr[[0, 3, k, f]] = c1.im;
            }
        }

        let outputs = session.run(arr.into_dyn())?;
        let out = outputs
            .first()
            .ok_or_else(|| Error::Backend("模型无输出".to_string()))?;
        let shape = out.shape().to_vec();
        if shape.len() != 4 || shape[0] != 1 || shape[2] != dim_f || shape[3] != frames {
            return Err(Error::Backend(format!(
                "MDX 输出形状异常: {shape:?}（期望 [1, 2 或 4, {dim_f}, {frames}]）"
            )));
        }
        // 复数 mask：v = out[0,0]+j out[0,1]；i = out[0,2]+j out[0,3]
        // 输出仅 2 通道时：instrumental mask = 1 - vocals mask。
        let (v_re, v_im, i_re, i_im) = match shape[1] {
            4 => (
                mask_plane(out, 0),
                mask_plane(out, 1),
                mask_plane(out, 2),
                mask_plane(out, 3),
            ),
            2 => {
                let v_re = mask_plane(out, 0);
                let v_im = mask_plane(out, 1);
                (v_re.clone(), v_im.clone(), one_minus(&v_re), neg(&v_im))
            }
            n => return Err(Error::Backend(format!("MDX 输出通道数异常: {n}"))),
        };

        // masked 频谱：spec * mask（复数相乘，两声道共用 mask）
        // 输出 mask 覆盖 dim_f bins；iSTFT 需 n_fft/2+1 bins，Nyquist bin 置零。
        let full_freq = params.n_fft / 2 + 1;
        // [声道][帧][freq]
        let mut v_spec: Vec<Vec<Vec<Complex<f32>>>> = Vec::with_capacity(CH);
        let mut i_spec: Vec<Vec<Vec<Complex<f32>>>> = Vec::with_capacity(CH);
        for c in 0..CH {
            let mut v_planes = vec![vec![Complex::default(); full_freq]; frames];
            let mut i_planes = vec![vec![Complex::default(); full_freq]; frames];
            for f in 0..frames {
                for k in 0..dim_f {
                    let m_v = Complex::new(v_re[f * dim_f + k], v_im[f * dim_f + k]);
                    let m_i = Complex::new(i_re[f * dim_f + k], i_im[f * dim_f + k]);
                    v_planes[f][k] = spec[c][f][k] * m_v;
                    i_planes[f][k] = spec[c][f][k] * m_i;
                }
            }
            v_spec.push(v_planes);
            i_spec.push(i_planes);
        }

        // iSTFT（WOLA，每声道），交织输出
        let mut v_t = vec![0.0f32; (frames - 1) * params.hop * CH];
        let mut i_t = vec![0.0f32; (frames - 1) * params.hop * CH];
        for c in 0..CH {
            let v_mono = istft(&v_spec[c], params.n_fft, params.hop)?;
            let i_mono = istft(&i_spec[c], params.n_fft, params.hop)?;
            for (f, (a, b)) in v_mono.iter().zip(i_mono.iter()).enumerate() {
                v_t[f * CH + c] = *a;
                i_t[f * CH + c] = *b;
            }
        }

        // crossfade 累加（首尾 overlap 线性淡入/淡出）
        let out_len = v_t.len().min(end - start);
        for f in 0..out_len {
            let w = crossfade_weight(f, seg, overlap_len);
            let abs = start + f;
            if abs >= frames_total {
                break;
            }
            vocals[abs * CH] += v_t[f * CH] * w;
            vocals[abs * CH + 1] += v_t[f * CH + 1] * w;
            inst[abs * CH] += i_t[f * CH] * w;
            inst[abs * CH + 1] += i_t[f * CH + 1] * w;
            weight[abs] += w;
        }

        if let Some(c) = cancel {
            if c.is_cancelled() {
                return Err(Error::Cancelled);
            }
        }
        if let Some(pr) = progress {
            let _ = pr.try_send(ProgressEvent::Process {
                percent: (s + 1) as f32 / n_segments as f32,
                message: Some(format!("segment {}/{}", s + 1, n_segments)),
            });
        }
    }

    // 权重归一化
    for f in 0..frames_total {
        let w = weight[f];
        if w > 1e-6 {
            vocals[f * CH] /= w;
            vocals[f * CH + 1] /= w;
            inst[f * CH] /= w;
            inst[f * CH + 1] /= w;
        }
    }

    Ok(vec![
        (crate::model::STEM_VOCALS.to_string(), vocals),
        (crate::model::STEM_INSTRUMENTAL.to_string(), inst),
    ])
}

/// 从输出 `[1,C,freq,frames]` 提取第 `c` 通道平面（帧主序，与 spec 布局一致）。
fn mask_plane(out: &ndarray::ArrayD<f32>, c: usize) -> Vec<f32> {
    let shape = out.shape();
    let freq = shape[2];
    let frames = shape[3];
    let mut v = Vec::with_capacity(freq * frames);
    for f in 0..frames {
        for k in 0..freq {
            v.push(out[[0, c, k, f]]);
        }
    }
    v
}

fn one_minus(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| 1.0 - x).collect()
}

fn neg(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| -x).collect()
}

/// 立体声 STFT：返回 `[ch][frames][freq]` 复数频谱（hann 窗，center pad n_fft/2）。
fn stft(
    signal: &[f32],
    ch: usize,
    n_fft: usize,
    hop: usize,
) -> Result<Vec<Vec<Vec<Complex<f32>>>>> {
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let window = hann(n_fft);
    let pad = n_fft / 2;
    let frames_src = signal.len() / ch;
    let padded_len = frames_src + 2 * pad;
    let frames = padded_len.saturating_sub(n_fft) / hop + 1;

    let mut out: Vec<Vec<Vec<Complex<f32>>>> = Vec::with_capacity(ch);
    for c in 0..ch {
        let mut frames_out = Vec::with_capacity(frames);
        for t in 0..frames {
            let start = t * hop; // padded 索引
            let mut buf = vec![0.0f32; n_fft];
            for i in 0..n_fft {
                let orig = start + i;
                let orig = orig as isize - pad as isize;
                if orig >= 0 && orig < frames_src as isize {
                    buf[i] = signal[orig as usize * ch + c] * window[i];
                }
            }
            let mut out_buf = vec![Complex::new(0.0, 0.0); n_fft / 2 + 1];
            r2c.process(&mut buf, &mut out_buf)
                .map_err(|e| Error::Backend(format!("STFT 失败: {e}")))?;
            frames_out.push(out_buf);
        }
        out.push(frames_out);
    }
    Ok(out)
}

/// iSTFT（WOLA）：叠加加权帧并除以 ∑win²，再裁剪 center pad。
/// 输出长度 = hop × (frames-1)。
/// realfft 的逆变换不归一化（输出为 n_fft 倍），且要求 DC/Nyquist bin 虚部为零——
/// 二者在此处理（mask 相乘可能污染 DC 虚部，置零是实数信号物理约束下的合理近似）。
fn istft(frames_in: &[Vec<Complex<f32>>], n_fft: usize, hop: usize) -> Result<Vec<f32>> {
    let mut planner = RealFftPlanner::<f32>::new();
    let c2r = planner.plan_fft_inverse(n_fft);
    let frames = frames_in.len();
    let out_len = (frames.saturating_sub(1)) * hop + n_fft;
    let mut out = vec![0.0f32; out_len];
    let mut wsum = vec![0.0f32; out_len];
    let window = hann(n_fft);
    for t in 0..frames {
        let mut spec = frames_in[t].to_vec(); // 长度 n_fft/2+1
        spec[0].im = 0.0; // DC bin
        spec[n_fft / 2].im = 0.0; // Nyquist bin
        let mut time_buf = vec![0.0f32; n_fft];
        c2r.process(&mut spec, &mut time_buf)
            .map_err(|e| Error::Backend(format!("iSTFT 失败: {e}")))?;
        for i in 0..n_fft {
            let x = time_buf[i] / n_fft as f32; // realfft 逆变换无缩放，补 1/N
            out[t * hop + i] += x * window[i];
            wsum[t * hop + i] += window[i] * window[i];
        }
    }
    for i in 0..out_len {
        if wsum[i] > 1e-8 {
            out[i] /= wsum[i];
        }
    }
    // 裁剪 center pad（两侧各 n_fft/2）
    let start = n_fft / 2;
    let end = out_len - n_fft / 2;
    Ok(out[start..end].to_vec())
}

/// Hann 窗。
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// 交叉淡化权重：前 overlap 线性 0→1，后 overlap 线性 1→0，中间为 1。
fn crossfade_weight(i: usize, seg: usize, overlap: usize) -> f32 {
    if overlap == 0 {
        return 1.0;
    }
    if i < overlap {
        return i as f32 / overlap as f32;
    }
    if i >= seg - overlap {
        return (seg - i) as f32 / overlap as f32;
    }
    1.0
}
