//! candle 侧对照输出：0.5s 混合复音信号（与 polar_verify_torch.py 相同）→ npy 供对比。
//! 运行：cargo run --release -p audio-separator-core --example polar_candle_mix_out

use audio_separator_core::backend::local::arch::bs_polarformer::{BsPolarformer, RoformerOptions};

fn main() {
    let model = BsPolarformer::load(
        std::path::Path::new("test-assets/bs_polarformer.safetensors"),
        RoformerOptions::default(),
    )
    .unwrap();
    let samples = 512 * 43usize; // 22016, 0.499s
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
    let out = model.predict_window(&audio, samples).unwrap();
    let e: f32 = out.iter().map(|v| v * v).sum();
    println!("candle out energy: {e:.9}");
    let keep = samples.min(out.len());
    let mut bin = Vec::new();
    for ch in 0..2 {
        for i in 0..keep {
            bin.extend_from_slice(&out[ch * keep + i].to_le_bytes());
        }
    }
    std::fs::write("test-assets/candle_polar_out.npy", bin).unwrap();
    println!("saved candle_polar_out.npy (keep={keep})");
}
