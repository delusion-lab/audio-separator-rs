//! 模型参数配置文件解析（yaml / json）。
//!
//! 开源音频分离模型常把网络结构参数与权重同仓库发布为配置文件：
//! - MSST / UVR 系训练 yaml：`audio:` / `model:` 两段，含 `!!python/tuple` 标签、行内注释；
//! - HuggingFace 仓库：`config.yaml` / `config.json`。
//!
//! 解析策略：剥除 `!!python/tuple` 标签（保留冒号与缩进序列）→ YAML 解析 →
//! 顶层段（audio/model 等对象段）拍平合并为单层对象（键冲突时后出现的段覆盖），
//! 供架构注册表按各自 schema 读取；`freqs_per_bands` 等序列保持数组。

use crate::error::Error;

/// 解析模型参数配置文件文本（按首非空白字符判断 json / yaml）。
pub fn parse_config(text: &str) -> Result<serde_json::Value, Error> {
    let text = strip_python_tags(text);
    let trimmed = text.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        serde_json::from_str(&text).map_err(|e| Error::Model(format!("config JSON parse failed: {e}")))
    } else {
        let yaml: serde_yaml::Value = serde_yaml::from_str(&text)
            .map_err(|e| Error::Model(format!("config YAML parse failed: {e}")))?;
        let json = serde_json::to_value(yaml)
            .map_err(|e| Error::Model(format!("config YAML->JSON conversion failed: {e}")))?;
        Ok(flatten_top(&json))
    }
}

/// 剥除 YAML 的 `!!python/tuple` 标签：`key: !!python/tuple\n  - 2` → `key:\n  - 2`，
/// 序列保留、标签去除。内联形式 `!!python/tuple [a, b]` 也一并去除标签 token。
fn strip_python_tags(s: &str) -> String {
    s.replace("!!python/tuple", "")
}

/// 顶层段拍平：`{audio: {...}, model: {...}}` → 各段叶子键合并为单层对象；
/// 无对象段（如 `{overlap: 0.25, batch_size: 1}`）原样返回。
fn flatten_top(v: &serde_json::Value) -> serde_json::Value {
    let Some(obj) = v.as_object() else {
        return v.clone();
    };
    let segs: Vec<(&String, &serde_json::Map<String, serde_json::Value>)> = obj
        .iter()
        .filter_map(|(k, x)| x.as_object().map(|m| (k, m)))
        .collect();
    if segs.is_empty() {
        return v.clone();
    }
    let mut out = serde_json::Map::new();
    for (_, seg) in segs {
        for (key, val) in seg {
            out.insert(key.clone(), val.clone()); // 后写覆盖（model 段通常在后）
        }
    }
    for (k, val) in obj.iter().filter(|(_, x)| !x.is_object()) {
        out.insert(k.clone(), val.clone());
    }
    serde_json::Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MSST_YAML: &str = r#"
audio:
  chunk_size: 352800
  dim_f: 1024
  dim_t: 801 # don't work (use in model)
  hop_length: 441 # don't work (use in model)
  n_fft: 2048
  num_channels: 2
  sample_rate: 44100
  min_mean_abs: 0.0001
model:
  dim: 384
  depth: 12
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  linear_transformer_depth: 0
  freqs_per_bands: !!python/tuple
    - 2
    - 2
    - 4
    - 4
  dim_head: 64
  heads: 8
  attn_dropout: 0.1
  ff_dropout: 0.1
  flash_attn: true
  dim_freqs_in: 1025
  sample_rate: 44100
  stft_n_fft: 2048
  stft_hop_length: 441
  stft_win_length: 2048
  stft_normalized: false
  mask_estimator_depth: 2
"#;

    #[test]
    fn parse_msst_yaml() {
        let v = parse_config(MSST_YAML).unwrap();
        let obj = v.as_object().expect("should be an object");
        // 拍平：叶子键直接可见
        assert_eq!(obj["chunk_size"], 352800);
        assert_eq!(obj["dim"], 384);
        assert_eq!(obj["depth"], 12);
        assert_eq!(obj["heads"], 8);
        assert_eq!(obj["dim_head"], 64);
        assert_eq!(obj["flash_attn"], true);
        // 序列（!!python/tuple 剥除后为数组）
        let bands = obj["freqs_per_bands"].as_array().unwrap();
        assert_eq!(bands, &[2, 2, 4, 4]);
        // 键冲突：model.sample_rate 覆盖 audio.sample_rate（值相同）
        assert_eq!(obj["sample_rate"], 44100);
        // 注释被剥
        assert!(!obj.contains_key("don't work (use in model)"));
    }

    #[test]
    fn parse_flat_json() {
        let v = parse_config(r#"{"overlap": 0.25, "batch_size": 1}"#).unwrap();
        assert_eq!(v["overlap"], 0.25);
        assert_eq!(v["batch_size"], 1);
    }
}
