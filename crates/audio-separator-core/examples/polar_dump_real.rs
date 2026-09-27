//! 诊断：真实歌曲 band_split + vocals dump（对照 torch_real_band.npy 等）。
//! 运行：cargo run --release -p audio-separator-core --example polar_dump_real

use candle_core::{Device, Tensor};

use audio_separator_core::backend::local::arch::bs_polarformer::{
    BANDS, BsPolarformer, RoformerOptions, linear_forward, rms_norm,
};
use audio_separator_core::backend::local::arch::roformer_stft::PreciseStft;

fn write_npy(path: &str, data: &[f32], shape: &[usize]) {
    let mut bin = Vec::new();
    for v in data {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bin).unwrap();
    println!("saved {path} shape {shape:?}");
}

// 读 PCM16 stereo wav 前 n 样本 → [2, n] f32
fn read_wav_head(path: &str, n: usize) -> (usize, Vec<f32>) {
    let data = std::fs::read(path).unwrap();
    let sr = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
    let channels = u16::from_le_bytes(data[22..24].try_into().unwrap()) as usize;
    assert_eq!(channels, 2);
    let mut out = vec![0.0f32; 2 * n];
    for i in 0..n {
        let off = 44 + i * 2 * channels;
        for ch in 0..2 {
            let v = i16::from_le_bytes([data[off + ch * 2], data[off + ch * 2 + 1]]) as f32 / 32768.0;
            out[ch * n + i] = v;
        }
    }
    (sr, out)
}

fn main() {
    let model = BsPolarformer::load(
        std::path::Path::new("test-assets/bs_polarformer.safetensors"),
        RoformerOptions::default(),
    )
    .unwrap();
    let samples = 512 * 43usize;
    let (_sr, audio) = read_wav_head("test-assets/qi-feng-le-zh.wav", samples);
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
    for v in &all {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("test-assets/candle_real_band.npy", bin).unwrap();

    // 完整前向 → vocals（窗口 22016 → 输出 22016）
    let out = model.predict_window(&audio[..2 * samples], samples).unwrap();
    let mut ob = Vec::new();
    for v in &out {
        ob.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("test-assets/candle_real_vocals.npy", ob).unwrap();
    let e: f32 = out.iter().map(|x| x * x).sum();
    println!("saved real band + vocals; vocals energy {e:.3e}");
}
