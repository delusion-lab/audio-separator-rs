//! 诊断：dump candle 全部 STFT 频谱（bin-major，每声道）供与 torch.stft 对比。
//! 运行：cargo run --release -p audio-separator-core --example polar_dump_spectra

use audio_separator_core::backend::local::arch::roformer_stft::PreciseStft;

fn main() {
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
    let frames = samples / 512 + 1; // 44
    let bins = 1025;
    let mut prec = PreciseStft::new(512).unwrap();
    let mut spectra = Vec::new();
    for channel in audio.chunks_exact(samples) {
        spectra.push(prec.forward(channel).unwrap());
    }
    // 每声道 dump：bin-major [bins*frames]
    for (ch, spec) in spectra.iter().enumerate() {
        let mut bin = Vec::new();
        for v in spec {
            bin.extend_from_slice(&v.re.to_le_bytes());
            bin.extend_from_slice(&v.im.to_le_bytes());
        }
        std::fs::write(format!("test-assets/candle_polar_spec{ch}.npy"), bin).unwrap();
        println!("saved spec{ch}: {bins} bins x {frames} frames = {}", spec.len());
    }
}
