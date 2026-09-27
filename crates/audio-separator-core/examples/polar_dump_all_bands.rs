//! 诊断：dump candle 全部 62 band 的 band_split 输出（npy [62, 44, 256]）。
//! 运行：cargo run --release -p audio-separator-core --example polar_dump_all_bands

use candle_core::{Device, Tensor};

use audio_separator_core::backend::local::arch::bs_polarformer::{
    BANDS, BsPolarformer, RoformerOptions, linear_forward, rms_norm,
};
use audio_separator_core::backend::local::arch::roformer_stft::PreciseStft;

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
    let frames = samples / 512 + 1;
    let mut prec = PreciseStft::new(512).unwrap();
    let mut spectra = Vec::new();
    for channel in audio.chunks_exact(samples) {
        spectra.push(prec.forward(channel).unwrap());
    }
    let device = Device::Cpu;
    let mut first_bin = 0;
    let mut all: Vec<f32> = Vec::with_capacity(62 * frames * 256);
    for (index, &bins) in BANDS.iter().enumerate() {
        let (gamma, linear) = (model.band_gamma(index), model.band_linear(index));
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
        all.extend_from_slice(&feat.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        first_bin += bins;
    }
    let mut bin = Vec::new();
    for v in all {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    let size = bin.len();
    std::fs::write("test-assets/candle_polar_all_bands.npy", bin).unwrap();
    println!("saved all bands: {} bytes ({} bands x 44 x 256)", size, 62);
}
