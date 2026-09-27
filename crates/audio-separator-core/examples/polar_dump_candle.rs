//! 诊断：dump candle 侧 band_split 输出与 mask 输出（对照 torch_polar_band/mask.npy）。
//! 运行：cargo run --release -p audio-separator-core --example polar_dump_candle

use candle_core::{Device, Tensor};

use audio_separator_core::backend::local::arch::bs_polarformer::{
    BsPolarformer, RoformerOptions, linear_forward, rms_norm,
};
use audio_separator_core::backend::local::arch::roformer_stft::{FFT, PreciseStft};

fn write_npy(path: &str, data: &[f32], shape: &[usize]) {
    let mut bin = Vec::new();
    // 转 C 顺序（candle 是 row-major，torch numpy 默认 C 顺序）——candle 输出 [1,1,frames,input] row-major
    for v in data {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bin).unwrap();
    println!("saved {path} shape {shape:?}");
}

fn main() {
    let model = BsPolarformer::load(
        std::path::Path::new("test-assets/bs_polarformer.safetensors"),
        RoformerOptions::default(),
    )
    .unwrap();
    let samples = 512 * 43usize;
    let sr = 44100.0f32;
    let mut audio = vec![0.0f32; 2 * samples];
    for i in 0..samples {
        let t = i as f32 / sr;
        audio[i] = 0.3 * (2.0 * std::f32::consts::PI * 330.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * 550.0 * t).sin();
        audio[samples + i] = 0.3 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * 1100.0 * t).sin();
    }
    let frames = samples / 512 + 1; // 43+1 = 44

    // STFT
    let mut prec = PreciseStft::new(512).unwrap();
    let mut spectra = Vec::new();
    for channel in audio.chunks_exact(samples) {
        spectra.push(prec.forward(channel).unwrap());
    }
    // band_split 输出（band 0 与 band 61）
    let device = Device::Cpu;
    let mut first_bin = 0;
    let mut band_out0: Option<Tensor> = None;
    let mut band_out_last: Option<Tensor> = None;
    for (index, &bins) in super_bands().iter().enumerate() {
        let (gamma, linear) = (
            model.band_gamma(index),
            model.band_linear(index),
        );
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
        let x = Tensor::from_vec(values, (1, 1, frames, input), &device).unwrap();
        let x = rms_norm(&x, gamma, input).unwrap();
        let feat = linear_forward(linear, &x).unwrap();
        if index == 0 {
            band_out0 = Some(feat.clone());
        }
        if index == 61 {
            band_out_last = Some(feat);
        }
        first_bin += bins;
    }
    let b0 = band_out0
        .as_ref()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    write_npy("test-assets/candle_polar_band0.npy", &b0, &[1, 1, frames, 256]);
    let bl = band_out_last.unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    write_npy("test-assets/candle_polar_band61.npy", &bl, &[1, 1, frames, 256]);

    // mask0（band 0）输出：通过完整 predict_window 无法中途取——直接用 mask_forward
    let x0 = Tensor::from_vec(
        band_out0.unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        (1, 1, frames, 256),
        &device,
    )
    .unwrap();
    let mask = model.mask_forward(0, &x0).unwrap();
    let m = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    write_npy("test-assets/candle_polar_mask0.npy", &m, &[frames, 2 * 8]);
}

// 避免引入整个常量：直接引用 BANDS
use audio_separator_core::backend::local::arch::bs_polarformer::BANDS;
fn super_bands() -> &'static [usize; 62] {
    &BANDS
}
