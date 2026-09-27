//! 诊断：真实歌曲 dump transformer 输出（final_norm 后 x）与 mask0，对照 torch_real_x.npy。
//! 运行：cargo run --release -p audio-separator-core --example polar_dump_x

use candle_core::{Device, Tensor};

use audio_separator_core::backend::local::arch::bs_polarformer::{
    BANDS, DIM, BsPolarformer, RoformerOptions, linear_forward, rms_norm,
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
    let (time_batch, frequency_batch) = model.batch_sizes();

    let mut features: Vec<Tensor> = Vec::with_capacity(BANDS.len());
    let mut first_bin = 0;
    for ((gamma, linear), &bins) in model.bands().iter().zip(&BANDS) {
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
        features.push(linear_forward(linear, &x).unwrap());
        first_bin += bins;
    }
    let mut x = Tensor::cat(&features, 1).unwrap();
    for [time, frequency] in model.layers() {
        x = time.forward_batches(&x, time_batch).unwrap();
        let freq_in = x.permute((0, 2, 1, 3)).unwrap();
        x = frequency.forward_batches(&freq_in, frequency_batch).unwrap();
        x = x.permute((0, 2, 1, 3)).unwrap();
    }
    x = rms_norm(&x, &model.final_norm(), DIM).unwrap();
    // dump x [1, frames, 62, DIM]
    let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    write_npy("test-assets/candle_real_x.npy", &xv, &[1, frames, 62, DIM]);

    // mask0：band 0 slice → estimator
    let band_slice = x.narrow(1, 0, 1).unwrap(); // [1, 1, frames, DIM]
    let mask = model.mask_forward(0, &band_slice).unwrap(); // [frames, 16]
    let mv = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    write_npy("test-assets/candle_real_mask0.npy", &mv, &[frames, 16]);
    println!("done");
}
