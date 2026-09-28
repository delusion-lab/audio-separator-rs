//! config_url 集成验证：加载本地 models.json → resolve 三个 Roformer 模型名 →
//! 打印从 URL 下载解析出的架构参数关键键。
//! 运行：cargo run -p audio-separator-core --example fetch_config_url

use audio_separator_core::backend::local::model_manage::ModelManager;
use audio_separator_core::config::{ModelListSource, ModelsConfig};
use audio_separator_core::model::ModelRef;

fn main() {
    let cfg = ModelsConfig {
        list: Some(ModelListSource::Path("models.json".into())),
        cache_dir: None,
            rankings: None,
    };
    let mgr = ModelManager::load(&cfg, None).unwrap();
    let names = [
        "model_bs_roformer_ep_368_sdr_12.9628",
        "model_mel_band_roformer_ep_3005_sdr_11.4360",
        "model_bs_polarformer_float16",
    ];
    for name in names {
        let resolved = mgr.resolve(&ModelRef::Name(name.to_string()), None, None, None).unwrap();
        let entry = resolved.entry.as_ref().unwrap();
        let p = &entry.params;
        let pick = |k: &str| {
            p.get(k).and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| {
                v.as_i64().map(|i| i.to_string())
            })).unwrap_or_else(|| "∅".to_string())
        };
        println!(
            "[{}] config_url={:?} → n_fft={} hop={} dim={} depth={} heads={} num_bands={} sample_rate={}",
            name,
            entry.config_url,
            pick("n_fft"),
            pick("hop_length"),
            pick("dim"),
            pick("depth"),
            pick("heads"),
            pick("num_bands"),
            pick("sample_rate"),
        );
    }
}
