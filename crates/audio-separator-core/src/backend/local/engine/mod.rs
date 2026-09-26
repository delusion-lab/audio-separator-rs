//! 本地推理引擎：onnx（M1）与 candle（M2，Roformer 家族）。

pub mod onnx;

pub use onnx::OnnxSession;
