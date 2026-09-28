//! audio-separator-server：音频分离 HTTP 服务（M4）。
//!
//! 启动参数（均可被同名校验覆盖）：
//! - `--config <path>`：核心配置 TOML（models / mvsep / network / server 段）
//! - `--addr <host:port>`：监听地址（默认 127.0.0.1:8080）
//! - `--workers <n>`：并行任务数（默认 1；MVSEP 非 Premium 平台侧仅允许 1）
//! - `--auth-token <t>`：启用 Bearer Token 鉴权（默认关，D4）
//! - `--api-key <k>`：MVSEP API Key（也可环境变量 `ASEP_MVSEP_API_KEY`）
//! - `--upload-dir <dir>`：上传/输出根目录（默认 `./uploads`）
//!
//! 环境变量 `ASEP_MVSEP_API_KEY` 恒生效；工作目录存在 `models.json` 且未配置
//! models.list 时自动用作模型清单（与 CLI 行为一致）。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::net::TcpListener;

use audio_separator_core::config::{Config, ModelListSource};

mod api;
mod store;
mod worker;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config::default();
    let mut upload_dir = PathBuf::from("uploads");

    // 命令行覆盖（简单解析；参数即 `--key value` 形式）
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    cfg = match Config::load(std::path::Path::new(p)) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("config load failed ({p}): {e}");
                            std::process::exit(1);
                        }
                    };
                }
            }
            "--addr" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.server.addr = v.clone();
                }
            }
            "--workers" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    cfg.server.workers = v;
                }
            }
            "--auth-token" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.server.auth_token = Some(v.clone());
                }
            }
            "--api-key" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.mvsep.api_key = Some(v.clone());
                }
            }
            "--upload-dir" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    upload_dir = PathBuf::from(v);
                }
            }
            "--webhook-url" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.server.webhook_url = Some(v.clone());
                }
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("{}", usage());
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // 环境变量覆盖 MVSEP API Key
    if let Ok(k) = std::env::var("ASEP_MVSEP_API_KEY") {
        if !k.is_empty() {
            cfg.mvsep.api_key = Some(k);
        }
    }

    // 模型清单兜底：未配置 list 且工作目录存在 models.json
    if cfg.models.list.is_none() {
        let local = std::path::Path::new("models.json");
        if local.exists() {
            cfg.models.list = Some(ModelListSource::Path(local.to_path_buf()));
        }
    }

    // 后端集合（启动时构建；local 必建，mvsep 依赖 API Key）
    let (local, mvsep) = match worker::build_separators(&cfg) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("server start failed: {e}");
            std::process::exit(1);
        }
    };

    // MVSEP 客户端（算法目录查询；无 Key 也可建）
    let mvsep_client = audio_separator_core::backend::mvsep::MvsepClient::new(
        &cfg.mvsep,
        cfg.network.proxy.as_deref(),
    )
    .ok();

    // 本地清单与排名（启动时已由 build_separators 加载；此处直接构建用于 /models 与 /rankings）
    let local_manager = audio_separator_core::backend::local::model_manage::ModelManager::load(
        &cfg.models,
        cfg.network.proxy.as_deref(),
    )
    .ok();
    let local_manifest = local_manager
        .as_ref()
        .map(|m| m.list().clone())
        .unwrap_or_else(|| audio_separator_core::model::ModelList { version: 0, models: Vec::new() });
    let local_rankings = local_manager
        .as_ref()
        .map(|m| m.rankings().clone())
        .unwrap_or_default();

    let store: Arc<dyn store::TaskStore> = Arc::new(store::InMemoryTaskStore::new());
    let _workers = worker::WorkerPool::spawn(
        Arc::clone(&store),
        local,
        mvsep,
        cfg.server.workers,
        upload_dir.clone(),
        cfg.server.webhook_url.clone(),
    );

    let state = Arc::new(api::AppState {
        cfg: cfg.clone(),
        store,
        local_manifest,
        local_rankings,
        mvsep_client,
        upload_dir,
        started: Instant::now(),
    });

    let app = api::build_router(state);

    let listener = match TcpListener::bind(&cfg.server.addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to listen on {}: {e}", cfg.server.addr);
            std::process::exit(1);
        }
    };
    println!(
        "audio-separator-server listening on {} (workers={}, auth={}, mvsep={})",
        cfg.server.addr,
        cfg.server.workers,
        if cfg.server.auth_token.is_some() { "on" } else { "off" },
        if cfg.mvsep.api_key.is_some() { "enabled" } else { "disabled" },
    );
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("server exited abnormally: {e}");
        std::process::exit(1);
    }
}

fn usage() -> &'static str {
    "audio-separator-server [options]\n\
     \x20 --config <path>    core config TOML\n\
     \x20 --addr <host:port> listen address (default 127.0.0.1:8080)\n\
     \x20 --workers <n>      parallel tasks (default 1)\n\
     \x20 --auth-token <t>   enable Bearer Token (default off)\n\
     \x20 --api-key <k>      MVSEP API key (or env ASEP_MVSEP_API_KEY)\n\
     \x20 --upload-dir <dir>  upload root (default ./uploads)\n\
     \x20 --webhook-url <u>   task completion callback URL (optional)"
}
