//! 权重转换子系统：torch `.ckpt` / `.pt` → safetensors。
//!
//! - `pickle`：受限 pickle 虚拟机（白名单 opcode/GLOBAL，安全解析）
//! - `torch_zip`：torch zip 存档容器
//! - `safetensors`：safetensors 写入与校验
//! - `convert`：端到端转换

pub mod convert;
pub mod pickle;
pub mod safetensors;
pub mod torch_zip;

pub use convert::convert_ckpt_to_safetensors;
