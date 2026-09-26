//! 性能剖析：8 秒窗口分阶段计时，定位 bs_roformer 热点。
//! 运行：cargo run --release -p audio-separator-core --example profile_bs_roformer

use std::time::Instant;

use audio_separator_core::backend::local::arch::bs_roformer::{BsRoformer, RoformerOptions};

fn main() {
    let path = std::path::Path::new("test-assets/bs_roformer_1296.safetensors");
    let model = BsRoformer::load(path, RoformerOptions::default()).unwrap();

    let samples = 352_800usize; // 8 秒
    let mut audio = vec![0.0f32; 2 * samples];
    for i in 0..samples {
        let t = i as f32 / 44100.0;
        audio[i] = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.05;
        audio[samples + i] = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.05;
    }

    // 预热一次（加载/规划/缓存）
    let _ = model.profile(&audio, samples).unwrap();
    let t0 = Instant::now();
    let timings = model.profile(&audio, samples).unwrap();
    let total = t0.elapsed().as_secs_f64();
    println!(
        "STFT: {:.2}s | band 投影: {:.2}s | 12 层 transformer: {:.2}s | mask: {:.2}s | 总计: {:.2}s",
        timings[0], timings[1], timings[2], timings[3], total
    );
}
