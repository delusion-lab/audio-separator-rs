//! 各架构的推理实现。
//!
//! 每种架构 = 专属参数 schema + 推理内核（STFT/iSTFT 前后处理 + 引擎调用），
//! 参数绝不跨架构混用（PLAN.md §5.3.2，D2 决策）。

pub mod mdx;

pub use mdx::{separate_mdx, MdxParams};
