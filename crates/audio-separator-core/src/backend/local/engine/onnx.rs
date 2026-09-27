//! ONNX Runtime 引擎封装（`ort` 2.0.0-rc.13）。
//!
//! 全局环境经 `ort::init().commit()` 配置一次；`Session` 为自持类型（无生命周期参数），
//! 进程级环境由 ort 内部管理。输入用 ndarray 0.17（与 ort 配套版本）构造。

use std::path::Path;
use std::sync::OnceLock;

use ndarray::ArrayD;
use ort::session::Session;
use ort::value::Value;

use crate::error::{Error, Result};

static INIT: OnceLock<()> = OnceLock::new();

/// 配置全局 ONNX Runtime 环境（进程内一次）。
fn ensure_init() -> Result<()> {
    if INIT.get().is_none() {
        // commit() 返回是否首次配置成功；重复配置返回 false，但环境同样可用。
        let _ = ort::init().with_name("audio-separator-rs").commit();
        let _ = INIT.set(());
    }
    Ok(())
}

/// 一次加载、可多次运行的 ONNX 会话。
pub struct OnnxSession {
    session: Session,
    input_name: String,
}

impl OnnxSession {
    /// 从本地 ONNX 文件加载会话。
    pub fn load(path: &Path) -> Result<Self> {
        ensure_init()?;
        let mut builder = Session::builder()
            .map_err(|e| Error::Backend(format!("failed to create ONNX Runtime session builder: {e}")))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|e| Error::Backend(format!("failed to load ONNX model {}: {e}", path.display())))?;
        let input_name = session
            .inputs()
            .first()
            .map(|m| m.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        Ok(Self {
            session,
            input_name,
        })
    }

    /// 读取模型自定义 metadata 属性（如 UVR 转换模型的 n_fft/hop_length/dim_f/dim_t）。
    pub fn metadata_value(&self, key: &str) -> Option<String> {
        self.session.metadata().ok()?.custom(key)
    }

    /// 运行推理：输入按模型第一个输入名绑定，返回所有输出（保持输出顺序）。
    pub fn run(&mut self, input: ArrayD<f32>) -> Result<Vec<ArrayD<f32>>> {
        let value =
            Value::from_array(input).map_err(|e| Error::Backend(format!("failed to build input tensor: {e}")))?;
        let inputs = ort::inputs![self.input_name.as_str() => value];
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| Error::Backend(format!("inference failed: {e}")))?;
        outputs
            .iter()
            .map(|(_, v)| {
                v.try_extract_array::<f32>()
                    .map(|x| x.to_owned())
                    .map_err(|e| Error::Backend(format!("failed to read inference output: {e}")))
            })
            .collect()
    }
}
