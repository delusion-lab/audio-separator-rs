//! BS-RoFormer 家族专属 STFT/iSTFT（参考 uvr_roformer 约定）。
//!
//! - 前向：对称 Hann 窗、reflect padding、**f64 精度**（避免 FFT 舍入主导安静频带），
//!   输出 bin-major 布局 `values[bin * frames + frame]`（含 Nyquist）。
//! - 逆向：Hermitian 镜像填充 → 逆 FFT（÷N）→ 加窗 overlap-add → 按窗平方和归一化。
//!
//! HOP 按架构参数化：bs_roformer / mel_band_roformer 用 441，bs_polarformer 用 512。

use std::sync::Arc;

use crate::error::{Error, Result};
use rustfft::{Fft, FftPlanner, num_complex::Complex32, num_complex::Complex64};

pub const FFT: usize = 2048;

/// f64 精度前向变换（每声道）。
pub struct PreciseStft {
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    buffer: Vec<Complex64>,
    scratch: Vec<Complex64>,
    hop: usize,
}

impl PreciseStft {
    pub fn new(hop: usize) -> Result<Self> {
        let fft = FftPlanner::<f64>::new().plan_fft_forward(FFT);
        let scratch_len = fft.get_inplace_scratch_len();
        Ok(Self {
            fft,
            scratch: vec![Complex64::default(); scratch_len],
            buffer: vec![Complex64::default(); FFT],
            window: (0..FFT)
                .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FFT as f64).cos())
                .collect(),
            hop,
        })
    }

    /// 输入长度 `len >= hop`（frames = len/hop + 1）。返回 bin-major 复数谱。
    pub fn forward(&mut self, audio: &[f32]) -> Result<Vec<Complex32>> {
        let pad = FFT / 2;
        let frames = audio.len() / self.hop + 1;
        let bins = pad + 1;
        let mut values = vec![Complex32::default(); bins * frames];
        for frame in 0..frames {
            for i in 0..FFT {
                let padded = frame * self.hop + i;
                let index = if padded < pad {
                    pad - padded
                } else {
                    let index = padded - pad;
                    if index < audio.len() {
                        index
                    } else {
                        2 * audio.len() - 2 - index
                    }
                };
                self.buffer[i] = Complex64::new(f64::from(audio[index]) * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut self.buffer, &mut self.scratch);
            for bin in 0..bins {
                values[bin * frames + frame] =
                    Complex32::new(self.buffer[bin].re as f32, self.buffer[bin].im as f32);
            }
        }
        Ok(values)
    }
}

/// f32 逆向变换（能量归一化），输出长度为 `(frames - 1) * hop`。
pub struct Stft {
    inverse: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buffer: Vec<Complex32>,
    scratch: Vec<Complex32>,
    hop: usize,
}

impl Stft {
    pub fn new(hop: usize) -> Result<Self> {
        let inverse = FftPlanner::<f32>::new().plan_fft_inverse(FFT);
        let scratch_len = inverse.get_inplace_scratch_len();
        Ok(Self {
            inverse,
            scratch: vec![Complex32::default(); scratch_len],
            buffer: vec![Complex32::default(); FFT],
            window: (0..FFT)
                .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FFT as f64).cos()) as f32)
                .collect(),
            hop,
        })
    }

    /// `spectrum` 为 bin-major（bins × frames），bins 必须 = FFT/2+1。
    pub fn inverse(&mut self, spectrum: &[Complex32], frames: usize) -> Result<Vec<f32>> {
        let bins = FFT / 2 + 1;
        if spectrum.len() != bins * frames {
            return Err(Error::Model("Roformer spectrum shape mismatch".to_string()));
        }
        let hop = self.hop;
        let natural = (frames - 1) * hop;
        let covered = natural + FFT;
        let mut wave = vec![0.0f32; covered];
        let mut energy = vec![0.0f32; covered];
        for frame in 0..frames {
            for bin in 0..bins {
                self.buffer[bin] = spectrum[bin * frames + frame];
            }
            for bin in bins..FFT {
                self.buffer[bin] = self.buffer[FFT - bin].conj();
            }
            self.inverse.process_with_scratch(&mut self.buffer, &mut self.scratch);
            for i in 0..FFT {
                let index = frame * hop + i;
                wave[index] += self.buffer[i].re * (1.0 / FFT as f32) * self.window[i];
                energy[index] += self.window[i] * self.window[i];
            }
        }
        let pad = FFT / 2;
        let mut output = vec![0.0f32; natural];
        for i in 0..natural {
            if energy[i + pad] <= 1e-11 {
                return Err(Error::Model("Roformer window overlap leaves uncovered samples".to_string()));
            }
            output[i] = wave[i + pad] / energy[i + pad];
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_reconstructs_signal() {
        let mut forward = PreciseStft::new(441).unwrap();
        let mut inverse = Stft::new(441).unwrap();
        let len = 441 * 20; // 20 帧
        let audio: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.05).sin() * 0.5)
            .collect();
        let spec = forward.forward(&audio).unwrap();
        let frames = len / 441 + 1;
        let out = inverse.inverse(&spec, frames).unwrap();
        assert_eq!(out.len(), len / 441 * 441);
        // 重建质量：中间段相对误差 < 1%
        let start = FFT;
        let end = out.len() - FFT;
        let mut max_err = 0.0f32;
        for i in start..end {
            let err = (out[i] - audio[i]).abs() / (audio[i].abs() + 1e-6);
            max_err = max_err.max(err);
        }
        assert!(max_err < 0.01, "max relative error {max_err}");
    }
}
