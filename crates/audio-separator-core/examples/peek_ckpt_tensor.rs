//! 调试：从原始 ckpt 直接提取指定 tensor 的值，与 safetensors 产物对比。
//! 运行：cargo run -p audio-separator-core --example peek_ckpt_tensor

use audio_separator_core::weights::torch_zip::TorchArchive;

fn main() {
    let ckpt = std::path::Path::new(
        "C:/Users/alice/AppData/Local/asep-ort/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    );
    let archive = TorchArchive::load(ckpt).unwrap();
    println!(
        "tensors: {}, byteorder: {}",
        archive.tensors.len(),
        archive.byteorder
    );

    for name in [
        "band_split.to_features.0.0.gamma",
        "band_split.to_features.9.0.gamma",
        "band_split.to_features.10.0.gamma",
        "band_split.to_features.11.0.gamma",
    ] {
        let rec = match archive.tensors.iter().find(|(n, _)| n == name) {
            Some((_, r)) => r,
            None => {
                println!("{name}: 不存在");
                continue;
            }
        };
        let storage = rec.storage.as_ref().unwrap();
        let raw = archive.read_storage(&storage.id).unwrap();
        let width = 4u64;
        let elem: u64 = rec.size.iter().product();
        let start = rec.offset as u64 * width;
        println!(
            "{name}: size={:?} stride={:?} offset={} storage.id={} numel={} dtype={:?} raw_len={}",
            rec.size,
            rec.stride,
            rec.offset,
            storage.id,
            storage.numel,
            storage.dtype,
            raw.len()
        );
        if (start + elem * width) as usize <= raw.len() {
            let mut vals = Vec::new();
            for i in 0..elem {
                let b = &raw[(start + i * width) as usize..(start + (i + 1) * width) as usize];
                vals.push(f32::from_le_bytes(b.try_into().unwrap()));
            }
            let finite: usize = vals.iter().filter(|v| v.is_finite()).count();
            println!(
                "  值({} 个, finite {finite}): {:?}",
                vals.len(),
                &vals[..vals.len().min(8)]
            );
        } else {
            println!("  越界！");
        }
    }
}
