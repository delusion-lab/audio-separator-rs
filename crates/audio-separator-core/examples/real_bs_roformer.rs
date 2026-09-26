//! 真实歌曲分离验证：qi-feng-le-zh.wav（26.1s）→ vocals/instrumental wav。
//! 运行（release）：cargo run --release -p audio-separator-core --example real_bs_roformer

use std::time::Instant;

use audio_separator_core::backend::local::arch::bs_roformer::{BsRoformer, RoformerOptions};
use audio_separator_core::io;

fn main() {
    let input = std::path::Path::new("test-assets/qi-feng-le-zh.wav");
    let decoded = io::decode(input).unwrap();
    println!(
        "输入: {} 声道 {}Hz {}s",
        decoded.channels,
        decoded.sample_rate,
        decoded.samples.len() as f32 / 2.0 / decoded.sample_rate as f32
    );

    let model_path = std::path::Path::new("test-assets/bs_roformer_1296.safetensors");
    let t0 = Instant::now();
    let model = BsRoformer::load(model_path, RoformerOptions::default()).unwrap();
    println!("加载完成 {:.1}s", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    let (vocals, instrumental) = model
        .separate(&decoded.samples, decoded.sample_rate)
        .unwrap();
    println!(
        "分离完成 {:.1}s（RTF {:.2}）",
        t1.elapsed().as_secs_f32(),
        t1.elapsed().as_secs_f64() / (decoded.samples.len() as f64 / 2.0 / decoded.sample_rate as f64)
    );
    let v_energy: f32 = vocals.iter().map(|v| v * v).sum();
    let i_energy: f32 = instrumental.iter().map(|v| v * v).sum();
    let in_energy: f32 = decoded.samples.iter().map(|v| v * v).sum();
    println!(
        "能量: 输入 {in_energy:.1} vocals {v_energy:.1} instrumental {i_energy:.1} (占比 {:.1}% / {:.1}%)",
        v_energy / in_energy * 100.0,
        i_energy / in_energy * 100.0
    );
    println!(
        "相关性(vocals, instrumental): {:.3}",
        correlation(&vocals, &instrumental)
    );

    io::write_wav(
        std::path::Path::new("test-assets/out_bs_vocals.wav"),
        &vocals,
        44100,
        2,
    )
    .unwrap();
    io::write_wav(
        std::path::Path::new("test-assets/out_bs_instrumental.wav"),
        &instrumental,
        44100,
        2,
    )
    .unwrap();
    println!("已写出 out_bs_vocals.wav / out_bs_instrumental.wav");
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va * vb).sqrt()
}
