//! ckpt → safetensors 转换主流程。

use std::path::Path;

use crate::error::{Error, Result};
use crate::weights::pickle::{Dtype, TensorRec};
use crate::weights::safetensors::{write, TensorData};
use crate::weights::torch_zip::TorchArchive;

/// 把 torch `.ckpt` / `.pt` 权重存档转换为 safetensors 文件。
///
/// 转换是**离线一次性**的：core 加载模型时若缓存中无对应 `.safetensors`，可调用本函数
/// 生成（懒转换 + 缓存）。安全前提：受限 pickle 解析（白名单），不执行任意代码。
pub fn convert_ckpt_to_safetensors(ckpt: &Path, out: &Path) -> Result<()> {
    let archive = TorchArchive::load(ckpt)?;
    let big_endian = archive.byteorder.trim() == "big";
    let mut tensors = Vec::with_capacity(archive.tensors.len());
    for (name, rec) in &archive.tensors {
        let storage = rec
            .storage
            .as_ref()
            .ok_or_else(|| Error::Model(format!("{name} missing storage")))?;
        let raw = archive.read_storage(&storage.id)?;
        let bytes = materialize(&raw, rec, storage.dtype, big_endian)?;
        // fp16 权重在 materialize 内已转 fp32（统一 safetensors 为 F32）。
        let dtype = if storage.dtype == Dtype::F16 {
            Dtype::F32
        } else {
            storage.dtype
        };
        tensors.push(TensorData {
            name: name.clone(),
            dtype,
            shape: rec.size.clone(),
            bytes,
        });
    }
    write(out, tensors)?;
    Ok(())
}

/// 按 stride 物化连续行主序字节（含 offset、字节序处理）。
fn materialize(storage: &[u8], rec: &TensorRec, dtype: Dtype, big_endian: bool) -> Result<Vec<u8>> {
    let width = dtype.bytes() as u64;
    let elem: u64 = rec.size.iter().product();
    let numel = rec.storage.as_ref().map(|s| s.numel).unwrap_or(0);
    if rec.offset as u64 + elem > numel {
        return Err(Error::Model(format!(
            "weight out of bounds: offset={} elem={} numel={}",
            rec.offset, elem, numel
        )));
    }
    let total = elem * width;
    let start = rec.offset as u64 * width;
    if start as usize + total as usize > storage.len() {
        return Err(Error::Model(format!(
            "shard data too short: need {} bytes, have {}",
            start + total,
            storage.len()
        )));
    }

    let contiguous = is_row_major_contiguous(&rec.size, &rec.stride);
    let mut out = if contiguous {
        storage[start as usize..(start + total) as usize].to_vec()
    } else {
        let mut out = vec![0u8; total as usize];
        for idx in 0..elem {
            let mut linear = rec.offset as u64;
            let mut rem = idx;
            for d in (0..rec.size.len()).rev() {
                let dim = rec.size[d];
                let coord = rem % dim;
                rem /= dim;
                linear += coord * rec.stride[d];
            }
            let src = linear * width;
            let dst = idx * width;
            out[dst as usize..(dst + width) as usize]
                .copy_from_slice(&storage[src as usize..(src + width) as usize]);
        }
        out
    };

    if big_endian {
        for chunk in out.chunks_exact_mut(width as usize) {
            chunk.reverse();
        }
    }
    // fp16 权重（如 bs_polarformer_float16）：物化后转 fp32，统一 safetensors 为 F32。
    if dtype == Dtype::F16 {
        return Ok(half_bytes_to_f32(&out));
    }
    Ok(out)
}

/// IEEE 754 half（2 字节小端）序列 → f32（4 字节小端）序列。
fn half_bytes_to_f32(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for chunk in src.chunks_exact(2) {
        let h = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&half_to_f32(h).to_le_bytes());
    }
    out
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31) // ±0
        } else {
            // 次正规：1.f × 2^-14，规格化后指数-15
            let mut e = -14i32;
            let mut m = frac;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            f32::from_bits((sign << 31) | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7f80_0000 | (frac << 13)) // inf / nan
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
    }
}

fn is_row_major_contiguous(size: &[u64], stride: &[u64]) -> bool {
    if size.len() != stride.len() {
        return false;
    }
    let mut expected = 1u64;
    for d in (0..size.len()).rev() {
        if stride[d] != expected {
            return false;
        }
        expected *= size[d];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(size: &[u64], stride: &[u64]) -> TensorRec {
        TensorRec {
            storage: Some(crate::weights::pickle::Storage {
                id: "0".into(),
                device: "cpu".into(),
                numel: size.iter().product(),
                dtype: Dtype::F32,
            }),
            offset: 0,
            size: size.to_vec(),
            stride: stride.to_vec(),
        }
    }

    #[test]
    fn contiguous_copy() {
        // [2,3] row-major，8 个 f32 原始字节
        let mut storage = Vec::new();
        for i in 0..6u32 {
            storage.extend_from_slice(&i.to_le_bytes());
        }
        let out = materialize(&storage, &rec(&[2, 3], &[3, 1]), Dtype::F32, false).unwrap();
        assert_eq!(out.len(), 24);
        assert_eq!(u32::from_le_bytes(out[12..16].try_into().unwrap()), 3);
    }

    #[test]
    fn strided_copy() {
        // 转置布局 [2,3] stride [1,2]：元素按 0,2,4,1,3,5 存
        let mut storage = Vec::new();
        for i in 0..6u32 {
            storage.extend_from_slice(&i.to_le_bytes());
        }
        let out = materialize(&storage, &rec(&[2, 3], &[1, 2]), Dtype::F32, false).unwrap();
        let vals: Vec<u32> = out
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // 行主序 [0,0]=0, [0,1]=2, [0,2]=4, [1,0]=1, [1,1]=3, [1,2]=5
        assert_eq!(vals, vec![0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn offset_respected() {
        let mut storage = Vec::new();
        for i in 0..8u32 {
            storage.extend_from_slice(&i.to_le_bytes());
        }
        let mut r = rec(&[2], &[1]);
        r.offset = 2;
        // storage 共 8 元素，从第 2 个元素起取 2 个
        r.storage.as_mut().unwrap().numel = 8;
        let out = materialize(&storage, &r, Dtype::F32, false).unwrap();
        let vals: Vec<u32> = out
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![2, 3]);
    }

    #[test]
    fn half_to_f32_values() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x8000), -0.0);
        assert_eq!(half_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(half_to_f32(0xC000), -2.0); // -2
        assert_eq!(half_to_f32(0x0001).to_bits(), 0x33800000); // 最小次正规 2^-24
        assert_eq!(half_to_f32(0x7C00), f32::INFINITY); // +inf
        assert!(half_to_f32(0x7E00).is_nan()); // nan
        // 1.5 = 0x3E00
        assert_eq!(half_to_f32(0x3E00), 1.5);
    }

    #[test]
    fn half_bytes_conversion() {
        // [1.0, -2.0, 0.5]
        let mut src = Vec::new();
        for h in [0x3C00u16, 0xC000u16, 0x3800u16] {
            src.extend_from_slice(&h.to_le_bytes());
        }
        let out = half_bytes_to_f32(&src);
        assert_eq!(out.len(), 12);
        let vals: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![1.0, -2.0, 0.5]);
    }
}
