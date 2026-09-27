//! safetensors 写入器（subset：f32/f16/bf16/f64/i32/i64/u8/bool，little endian，8 字节对齐）。
//!
//! 参考 safetensors 规范：header JSON + 数据区；header 中每个张量带 `data_offsets: [start, end)`
//! 与 `shape`、`dtype`；数据区按 8 字节对齐。本实现**物化连续布局**（按 stride 重排），
//! 保证 candle / python safetensors 均可直接读取。

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use crate::error::{Error, Result};
use crate::weights::pickle::Dtype;

/// 一个待写入的权重。
pub struct TensorData {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// 连续化后的元素（按 dtype 位宽对应的小端字节序列，长度 = 元素数 × 位宽）。
    pub bytes: Vec<u8>,
}

fn dtype_str(d: Dtype) -> &'static str {
    match d {
        Dtype::F32 => "F32",
        Dtype::F16 => "F16",
        Dtype::Bf16 => "BF16",
        Dtype::F64 => "F64",
        Dtype::I32 => "I32",
        Dtype::I64 => "I64",
        Dtype::U8 => "U8",
        Dtype::Bool => "BOOL",
    }
}

/// 写入 safetensors 文件。`tensors` 会按 name 排序（safetensors 要求 header 键排序）。
pub fn write(path: &Path, tensors: Vec<TensorData>) -> Result<()> {
    let mut by_name: BTreeMap<String, TensorData> = BTreeMap::new();
    for t in tensors {
        by_name.insert(t.name.clone(), t);
    }

    let mut offset: u64 = 0;
    let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for (name, t) in &by_name {
        let start = offset;
        let end = offset + t.bytes.len() as u64;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype_str(t.dtype),
                "shape": t.shape,
                "data_offsets": [start, end],
            }),
        );
        offset = align8(end);
    }
    let mut header_json = serde_json::to_vec(&header)?;
    header_json.push(b'\n');
    // safetensors 规范：文件头部 8 字节字段 = **8 字节对齐后**的 header 长度，
    // 数据区紧跟对齐边界开始。若写未对齐长度，reader 按字段跳过 header 会
    // 落在 padding 中间，导致全部 data_offsets 错位（曾踩坑）。
    let header_len_aligned = align8(header_json.len() as u64) as usize;

    let mut file = std::fs::File::create(path)?;
    file.write_all(&(header_len_aligned as u64).to_le_bytes())?;
    file.write_all(&header_json)?;
    let mut written = (8 + header_json.len()) as u64;
    let data_start = align8(written);
    // header 区尾部补齐到 8 字节对齐（与头部字段一致）。用空格而非 \x00：
    // JSON 解析器把尾部空白视为合法，0x00 则报 "Extra data"。
    while written < data_start {
        file.write_all(b" ")?;
        written += 1;
    }
    for t in by_name.values() {
        file.write_all(&t.bytes)?;
        // 每 tensor 尾对齐到 8 字节
        let mut pad = t.bytes.len() as u64;
        while (pad & 7) != 0 {
            file.write_all(&[0u8])?;
            pad += 1;
        }
    }
    Ok(())
}

fn align8(x: u64) -> u64 {
    (x + 7) & !7
}

/// 校验 safetensors 文件（读回 header 核对长度与偏移，供测试/诊断）。
pub fn validate(path: &Path) -> Result<()> {
    let data = std::fs::read(path)?;
    if data.len() < 8 {
        return Err(Error::Model("safetensors file too short".to_string()));
    }
    let header_len = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header_bytes = &data[8..8 + header_len];
    let header: serde_json::Value = serde_json::from_slice(header_bytes)?;
    let obj = header
        .as_object()
        .ok_or_else(|| Error::Model("safetensors header is not an object".to_string()))?;
    for (name, v) in obj {
        let offsets = v
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or_else(|| Error::Model(format!("{name} missing data_offsets")))?;
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        if end > data.len() || start >= end {
            return Err(Error::Model(format!("{name} offsets out of bounds")));
        }
    }
    Ok(())
}
