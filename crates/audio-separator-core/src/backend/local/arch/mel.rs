//! librosa 式 Slaney mel 滤波器组（mel_band_roformer 专用）。
//!
//! 公式逐项对照 librosa 源码（core/audio.py 的 hz_to_mel / mel_to_hz Slaney 分支，
//! filters.py 的 mel()），并已与 ep_3005_sdr_11.4360 权重的 band_split 输入维度
//! 逐一核对（60 个 band 输入维度 28,24×14,28×3,36×3,… 完全吻合）。
//! 强制项与 MSST model_mel_band_roformer.py 一致：
//! `weights[0][0] = 1.`、`weights[-1][-1] = 1.`（补 DC 与 Nyquist 覆盖）。

pub const N_MELS: usize = 60;

/// librosa `hz_to_mel(htk=False)`：分段线性 + 对数。
fn hz_to_mel_slaney(f: f64) -> f64 {
    let f_min = 0.0f64;
    let f_sp = 200.0 / 3.0;
    if f >= 1000.0 {
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = 6.4f64.ln() / 27.0;
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        (f - f_min) / f_sp
    }
}

/// librosa `mel_to_hz(htk=False)`。
fn mel_to_hz_slaney(m: f64) -> f64 {
    let f_min = 0.0f64;
    let f_sp = 200.0 / 3.0;
    if m >= 15.0 {
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = 6.4f64.ln() / 27.0;
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_min + f_sp * m
    }
}

/// 60×1025 三角 mel 滤波器组（slaney 归一化）。
pub fn mel_filter_bank(sr: u32, n_fft: usize) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let n_mels = N_MELS;
    // linspace(0, sr/2, n_freqs)
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|j| sr as f64 / 2.0 * j as f64 / (n_freqs as f64 - 1.0))
        .collect();
    // linspace(hz_to_mel(0), hz_to_mel(sr/2), n_mels+2)
    let min_mel = hz_to_mel_slaney(0.0);
    let max_mel = hz_to_mel_slaney(sr as f64 / 2.0);
    let mels: Vec<f64> = (0..n_mels + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f64 / (n_mels as f64 + 1.0))
        .collect();
    let mel_f: Vec<f64> = mels.iter().map(|&m| mel_to_hz_slaney(m)).collect();
    let fdiff: Vec<f64> = (0..n_mels + 1)
        .map(|i| mel_f[i + 1] - mel_f[i])
        .collect();
    let mut weights = vec![vec![0.0f32; n_freqs]; n_mels];
    for i in 0..n_mels {
        for (j, &freq) in fft_freqs.iter().enumerate() {
            let lower = -(mel_f[i] - freq) / fdiff[i];
            let upper = (mel_f[i + 2] - freq) / fdiff[i + 1];
            weights[i][j] = lower.max(0.0).min(upper) as f32;
        }
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for w in weights[i].iter_mut() {
            *w *= enorm as f32;
        }
    }
    // 强制项（与 MSST 一致）
    weights[0][0] = 1.0;
    weights[n_mels - 1][n_freqs - 1] = 1.0;
    weights
}

/// mel 频带结构：按 band 升序展平的覆盖频率索引、每 band 频率数、每频率覆盖 band 数。
pub struct MelBands {
    /// 单声道覆盖频率索引（band 升序展平），长度 = Σ num_freqs_per_band。
    pub freq_indices: Vec<usize>,
    /// 每 band 频率数（60 个）。
    pub num_freqs_per_band: Vec<usize>,
    /// 每频率被多少 band 覆盖（1025 个），用于 mask 平均。
    pub num_bands_per_freq: Vec<f32>,
}

pub fn build_mel_bands(sr: u32, n_fft: usize) -> MelBands {
    let n_freqs = n_fft / 2 + 1;
    let weights = mel_filter_bank(sr, n_fft);
    let mut freq_indices = Vec::new();
    let mut num_freqs_per_band = Vec::with_capacity(N_MELS);
    let mut num_bands_per_freq = vec![0.0f32; n_freqs];
    for row in &weights {
        let mut count = 0;
        for (f, &w) in row.iter().enumerate() {
            if w > 0.0 {
                freq_indices.push(f);
                num_bands_per_freq[f] += 1.0;
                count += 1;
            }
        }
        num_freqs_per_band.push(count);
    }
    MelBands {
        freq_indices,
        num_freqs_per_band,
        num_bands_per_freq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_bank_matches_weight_dims() {
        // 与 ep_3005_sdr_11.4360 band_split 输入维度（stereo，2*num_freqs*2）对照
        let bands = build_mel_bands(44_100, 2048);
        let expect: [usize; 60] = [
            28, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 36, 36, 36, 40,
            40, 44, 52, 52, 52, 60, 64, 68, 76, 80, 80, 88, 96, 104, 112, 116, 124, 132, 144, 156,
            164, 176, 188, 200, 216, 228, 244, 264, 284, 304, 320, 344, 372, 396, 420, 452, 488, 520,
        ];
        for (i, (actual, &want)) in bands
            .num_freqs_per_band
            .iter()
            .map(|&n| 4 * n)
            .zip(expect.iter())
            .enumerate()
        {
            assert_eq!(actual, want, "band {i}");
        }
        assert_eq!(bands.freq_indices.len(), 1979);
        assert_eq!(bands.num_bands_per_freq.len(), 1025);
        assert!(bands.num_bands_per_freq.iter().all(|&v| v >= 1.0 && v <= 2.0));
    }
}
