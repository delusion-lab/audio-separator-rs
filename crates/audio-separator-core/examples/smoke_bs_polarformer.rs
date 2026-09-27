//! bs_polarformer debug smoke：小窗口 predict_window（0.093s）+ 2s separate 调度。
//! 运行：cargo run -p audio-separator-core --example smoke_bs_polarformer

use std::time::Instant;

use audio_separator_core::backend::local::arch::bs_polarformer::{BsPolarformer, RoformerOptions};

fn main() {
    let model = BsPolarformer::load(
        std::path::Path::new("test-assets/bs_polarformer.safetensors"),
        RoformerOptions::default(),
    )
    .unwrap();
    println!("加载完成, 消费权重 {}", model.consumed_tensors());

    // 1) 小窗口 predict_window（512*8 = 4096 样本）
    let samples = 4096usize;
    let sr = 44100.0f32;
    let mut audio = vec![0.0f32; 2 * samples];
    for i in 0..samples {
        let t = i as f32 / sr;
        audio[i] = 0.3 * (2.0 * std::f32::consts::PI * 330.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        audio[samples + i] = 0.3 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin();
    }
    let t0 = Instant::now();
    let out = model.predict_window(&audio, samples).unwrap();
    let e: f32 = out.iter().map(|v| v * v).sum();
    println!(
        "窗口前向 {:.2}s 输出 {} 能量 {e:.3e} 有限={}",
        t0.elapsed().as_secs_f32(),
        out.len(),
        out.iter().all(|v| v.is_finite())
    );

    // 2) 2s 完整分离调度
    let total = 44100 * 2;
    let mut song = vec![0.0f32; 2 * total];
    for i in 0..total {
        let t = i as f32 / sr;
        song[i] = 0.2 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * 330.0 * t).sin();
        song[total + i] = 0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * 550.0 * t).sin();
    }
    let t1 = Instant::now();
    let (vocals, instrumental) = model.separate(&song, 44100).unwrap();
    println!("2s 分离 {:.1}s", t1.elapsed().as_secs_f32());
    let v_e: f32 = vocals.iter().map(|v| v * v).sum();
    let i_e: f32 = instrumental.iter().map(|v| v * v).sum();
    let in_e: f32 = song.iter().map(|v| v * v).sum();
    println!(
        "能量: 输入 {in_e:.1} vocals {v_e:.1} instrumental {i_e:.1} 有限={}",
        vocals.iter().chain(&instrumental).all(|v| v.is_finite())
    );
}
