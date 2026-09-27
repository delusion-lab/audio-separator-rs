//! mel_band_roformer 冒烟验证：加载转换后的 safetensors，跑窗口前向 + 完整调度。
//! 运行：cargo run -p audio-separator-core --example smoke_mel_band_roformer

use std::time::Instant;

use audio_separator_core::backend::local::arch::mel_band_roformer::{MelBandRoformer, RoformerOptions};

fn main() {
    let path = std::path::Path::new("test-assets/mel_band_roformer_3005.safetensors");
    let t0 = Instant::now();
    let model = MelBandRoformer::load(path, RoformerOptions::default()).unwrap();
    println!(
        "模型加载完成（{:.1}s），消费权重校验通过（期望 828）",
        t0.elapsed().as_secs_f32()
    );

    // 平面布局：左 440Hz、右 880Hz
    let samples = 4410usize; // 0.1s @ 44100
    let mut audio = vec![0.0f32; 2 * samples];
    for i in 0..samples {
        let t = i as f32 / 44100.0;
        audio[i] = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.1; // 左
        audio[samples + i] = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.1; // 右
    }
    let t1 = Instant::now();
    let out = model.predict_window(&audio, samples).unwrap();
    println!("窗口前向完成（{:.1}s）", t1.elapsed().as_secs_f32());

    let expected = samples / 441 * 441;
    println!(
        "窗口输出长度 {}（期望 {expected}），有限值: {}",
        out.len(),
        out.iter().all(|v| v.is_finite())
    );
    let energy: f32 = out.iter().map(|v| v * v).sum();
    println!("窗口输出能量: {energy:.6}");
    if out.len() != 2 * expected || !out.iter().all(|v| v.is_finite()) || !energy.is_finite() {
        std::process::exit(1);
    }

    // 完整调度：2 秒音频（一窗口），输出与输入同长
    let samples2 = 88200usize;
    let mut audio2 = vec![0.0f32; 2 * samples2];
    for i in 0..samples2 {
        let t = i as f32 / 44100.0;
        audio2[i] = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.1;
        audio2[samples2 + i] = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.1;
    }
    let t2 = Instant::now();
    let (vocals, instrumental) = model.separate(&audio2, 44100).unwrap();
    println!(
        "调度分离完成（{:.1}s），vocals={} instrumental={} 有限: {}",
        t2.elapsed().as_secs_f32(),
        vocals.len(),
        instrumental.len(),
        vocals.iter().chain(instrumental.iter()).all(|v| v.is_finite())
    );
    let v_energy: f32 = vocals.iter().map(|v| v * v).sum();
    let i_energy: f32 = instrumental.iter().map(|v| v * v).sum();
    println!("vocals 能量: {v_energy:.4}, instrumental 能量: {i_energy:.4}");
    if vocals.len() != 2 * samples2 || !vocals.iter().all(|v| v.is_finite()) {
        std::process::exit(1);
    }
    println!("OK: mel_band_roformer 冒烟通过");
}
