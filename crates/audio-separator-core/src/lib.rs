//! audio-separator-core：音频分离核心库。
//!
//! 提供统一的后端抽象（[`Separator`]）、作业模型、配置加载与模型清单/架构注册表类型。
//! 本地多架构推理（onnx / candle）与 MVSEP 云后端在不同里程碑实现，但都实现同一个 trait，
//! 供 CLI 与服务端共用（见 PLAN.md §5）。

pub mod backend;
pub mod config;
pub mod error;
pub mod job;
pub mod model;

pub use backend::{Input, SeparationRequest, SeparationResult, Separator};
pub use config::{BackendKind, Config, MvsepRegion};
pub use error::{Error, Result};
pub use job::{JobState, ProgressEvent};
pub use model::{ModelEntry, ModelList, ModelRef, OutputFormat};
