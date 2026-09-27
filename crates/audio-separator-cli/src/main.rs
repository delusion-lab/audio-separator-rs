//! asep CLI: audio separation command-line tool.
//!
//! - `separate`：单次分离（本地后端 M1 起可用；MVSEP 后端 M3 交付）
//! - `models` / `model-info`：模型清单查询（M1 后半交付）
//! - `job-status`：任务状态查询（M3 交付）
//! - `serve`：HTTP 服务（M4 交付）

mod server_client;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use audio_separator_core::backend::local::model_manage::ModelManager;
use audio_separator_core::backend::local::LocalSeparator;
use audio_separator_core::backend::mvsep::{MvsepClient, MvsepSeparator};
use audio_separator_core::backend::{Input, SeparationRequest, Separator};
use audio_separator_core::config::{Config, ModelListSource, MvsepRegion};
use audio_separator_core::error::{Error, Result};
use audio_separator_core::job::ProgressEvent;
use server_client::{ServerClient, SubmitArgs};
use audio_separator_core::model::{ModelRef, OutputFormat};
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

/// Audio separation tool: local multi-architecture inference / MVSEP cloud API / HTTP server.
#[derive(Parser)]
#[command(name = "asep", version, about = "Audio separation tool: local inference / MVSEP cloud API / HTTP server", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Separate a single audio file (local or MVSEP backend).
    Separate(SeparateArgs),

    /// List available models (local = manifest / mvsep = platform API / server = asep-server API).
    Models {
        /// Backend.
        #[arg(long, value_enum, default_value_t = BackendArg::Local)]
        backend: BackendArg,
        /// Override model list source: local JSON path.
        #[arg(long)]
        models_file: Option<PathBuf>,
        /// Override model list source: remote JSON URL.
        #[arg(long)]
        models_url: Option<String>,
        /// asep-server base URL (used when backend=server).
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        server_url: String,
        /// Bearer token for the asep-server API (optional).
        #[arg(long)]
        auth_token: Option<String>,
    },

    /// Show model details (architecture, engine, stems, params, source).
    #[command(name = "model-info")]
    ModelInfo {
        /// Model name / download URL / local path.
        model: String,
        /// 覆盖模型清单源：本地 JSON 路径。
        #[arg(long)]
        models_file: Option<PathBuf>,
        /// 覆盖模型清单源：远程 JSON URL。
        #[arg(long)]
        models_url: Option<String>,
    },

    /// Query task status (MVSEP hash or asep-server task id).
    #[command(name = "job-status")]
    JobStatus {
        /// Task hash or job id.
        id: String,
        /// MVSEP API key (when querying MVSEP tasks).
        #[arg(long, env = "ASEP_MVSEP_API_KEY")]
        api_key: Option<String>,
        /// asep-server base URL; when set, query the server instead of MVSEP.
        #[arg(long)]
        server_url: Option<String>,
        /// Bearer token for the asep-server API (optional).
        #[arg(long)]
        auth_token: Option<String>,
    },

    /// Start the HTTP server.
    Serve(ServeArgs),
}

/// Arguments for the `separate` subcommand.
#[derive(clap::Args)]
struct SeparateArgs {
    /// Input audio file or remote URL.
    input: String,

    /// Output directory.
    #[arg(short, long)]
    output: PathBuf,

    /// Backend: local / mvsep.
    #[arg(long, value_enum, default_value_t = BackendArg::Local)]
    backend: BackendArg,

    /// Model: name / download URL / local path (lazy download on first use of a manifest name).
    #[arg(long)]
    model: Option<String>,

    /// Explicit architecture (only for URL / local path forms; manifest decides for names).
    /// Values: mdx / bs_roformer / mel_band_roformer / bs_polarformer.
    #[arg(long)]
    arch: Option<String>,

    /// Model config file (yaml/json) URL or local path. When set, it is the authoritative
    /// source of architecture params; combined with --model URL / local path it works
    /// without a models.json entry. When referencing a name, it overrides its config_url.
    #[arg(long)]
    config_url: Option<String>,

    /// Only output selected stems (comma-separated); the full stem set is defined by the model.
    #[arg(long, value_delimiter = ',')]
    stems: Option<Vec<String>>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = FormatArg::Wav)]
    format: FormatArg,

    /// MVSEP API key (required when backend=mvsep).
    #[arg(long, env = "ASEP_MVSEP_API_KEY")]
    api_key: Option<String>,

    /// MVSEP region.
    #[arg(long, value_enum, default_value_t = RegionArg::Auto)]
    region: RegionArg,

    /// MVSEP polling timeout (seconds).
    #[arg(long)]
    poll_timeout: Option<u64>,

    /// MVSEP concurrent tasks (1 for non-Premium).
    #[arg(long)]
    concurrency: Option<usize>,

    /// MVSEP extra option add_opt1 (passed through when backend=mvsep; overrides manifest mapping).
    #[arg(long)]
    add_opt1: Option<String>,

    /// MVSEP extra option add_opt2.
    #[arg(long)]
    add_opt2: Option<String>,

    /// MVSEP extra option add_opt3.
    #[arg(long)]
    add_opt3: Option<String>,

    /// MVSEP completion callback URL (platform POSTs the result, no polling needed).
    #[arg(long)]
    webhook_url: Option<String>,

    /// 覆盖模型清单源：本地 JSON 路径。
    #[arg(long)]
    models_file: Option<PathBuf>,

    /// 覆盖模型清单源：远程 JSON URL。
    #[arg(long)]
    models_url: Option<String>,

    /// Config file path.
    #[arg(long)]
    config: Option<PathBuf>,

    /// asep-server base URL (used when backend=server).
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    server_url: String,

    /// Bearer token for the asep-server API (optional).
    #[arg(long)]
    auth_token: Option<String>,

    /// Verbosity (repeatable).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

/// Arguments for the `serve` subcommand.
#[derive(clap::Args)]
struct ServeArgs {
    /// Listen address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,

    /// 配置文件路径。
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum BackendArg {
    Local,
    Mvsep,
    Server,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum FormatArg {
    /// 16-bit PCM WAV (default).
    Wav,
    /// 32-bit float WAV.
    Wav32,
    /// 16-bit FLAC.
    Flac,
    /// 24-bit FLAC.
    Flac24,
    /// MP3 320kbps.
    Mp3,
    /// M4A (MVSEP backend only).
    M4a,
}

impl From<FormatArg> for OutputFormat {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Wav => OutputFormat::Wav16,
            FormatArg::Wav32 => OutputFormat::Wav32,
            FormatArg::Flac => OutputFormat::Flac16,
            FormatArg::Flac24 => OutputFormat::Flac24,
            FormatArg::Mp3 => OutputFormat::Mp3,
            FormatArg::M4a => OutputFormat::M4a,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RegionArg {
    Auto,
    De,
    De2,
    Sg,
}

impl From<RegionArg> for MvsepRegion {
    fn from(r: RegionArg) -> Self {
        match r {
            RegionArg::Auto => MvsepRegion::Auto,
            RegionArg::De => MvsepRegion::De,
            RegionArg::De2 => MvsepRegion::De2,
            RegionArg::Sg => MvsepRegion::Sg,
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Separate(args) => run_separate(args).await,
        Command::Models {
            backend,
            models_file,
            models_url,
            server_url,
            auth_token,
        } => run_models(backend, models_file, models_url, server_url, auth_token).await,
        Command::ModelInfo {
            model,
            models_file,
            models_url,
        } => run_model_info(&model, models_file, models_url).await,
        Command::JobStatus {
            id,
            api_key,
            server_url,
            auth_token,
        } => run_job_status(&id, api_key, server_url, auth_token).await,
        Command::Serve(_) => not_implemented("serve", "M4"),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `separate` 执行：解析模型三态、构建后端（本地 / MVSEP）、订阅进度、等待完成。
async fn run_separate(args: SeparateArgs) -> Result<()> {
    // 配置：默认值 ← CLI 覆盖
    let mut cfg = match &args.config {
        Some(p) => Config::load(p)?,
        None => Config::default(),
    };
    if let Some(f) = &args.models_file {
        cfg.models.list = Some(ModelListSource::Path(f.clone()));
    }
    if let Some(u) = &args.models_url {
        cfg.models.list = Some(ModelListSource::Url(u.clone()));
    }
    if cfg.models.list.is_none() {
        // 兜底：未指定清单源且工作目录存在 models.json 时自动加载（本地开发 / 独立清单仓库 checkout）
        let local = Path::new("models.json");
        if local.exists() {
            cfg.models.list = Some(ModelListSource::Path(local.to_path_buf()));
        }
    }
    if let Some(k) = &args.api_key {
        cfg.mvsep.api_key = Some(k.clone());
    }
    cfg.mvsep.region = args.region.into();
    if let Some(t) = args.poll_timeout {
        cfg.mvsep.poll_timeout_secs = t;
    }
    if let Some(c) = args.concurrency {
        cfg.mvsep.concurrency = c;
    }

    let model = parse_model_ref(&args.model, args.arch.clone())?;
    let input = if args.input.starts_with("http://") || args.input.starts_with("https://") {
        Input::Url(args.input.clone())
    } else {
        Input::Path(PathBuf::from(&args.input))
    };
    let req = SeparationRequest {
        input,
        model,
        output_format: args.format.into(),
        output_dir: args.output.clone(),
        select_stems: args.stems.clone(),
        config_url: args.config_url.clone(),
    };

    // 进度订阅（进度/下载/推理输出到 stderr，结果到 stdout）
    let (tx, mut rx) = mpsc::channel::<ProgressEvent>(64);
    let consumer = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            print_progress(&ev);
        }
    });

    let separator: Box<dyn Separator> = match args.backend {
        BackendArg::Local => {
            // ModelManager::load 可能访问网络/文件系统且内部使用 blocking Client，
            // 须在阻塞线程执行，避免阻塞操作跨入 tokio 异步上下文。
            let cfg = cfg.clone();
            Box::new(
                tokio::task::spawn_blocking(move || LocalSeparator::new(&cfg))
                    .await
                    .map_err(|e| Error::Other(format!("local backend init failed: {e}")))??,
            )
        }
        BackendArg::Server => {
            let client = ServerClient::new(&args.server_url, args.auth_token.clone())?;
            return run_separate_server(&client, &args).await;
        }
        BackendArg::Mvsep => {
            // 加载清单（manifest 的 mvsep 映射用于按名引用）；网络型清单在阻塞线程拉取。
            let cfg_clone = cfg.clone();
            let manager = tokio::task::spawn_blocking(move || {
                ModelManager::load(&cfg_clone.models, cfg_clone.network.proxy.as_deref())
            })
            .await
            .map_err(|e| Error::Other(format!("model list load failed: {e}")))??;
            let mut add_opts = BTreeMap::new();
            if let Some(v) = &args.add_opt1 {
                add_opts.insert("add_opt1".to_string(), v.clone());
            }
            if let Some(v) = &args.add_opt2 {
                add_opts.insert("add_opt2".to_string(), v.clone());
            }
            if let Some(v) = &args.add_opt3 {
                add_opts.insert("add_opt3".to_string(), v.clone());
            }
            Box::new(MvsepSeparator::new(
                &cfg,
                add_opts,
                args.webhook_url.clone(),
                Some(manager.list().clone()),
            )?)
        }
    };

    let result = separator.separate(req, Some(tx), None).await?;
    let _ = consumer.await;

    println!();
    println!(
        "Separation done ({:.1}s, backend {}):",
        result.elapsed.as_secs_f64(),
        match result.backend {
            audio_separator_core::config::BackendKind::Local => "local",
            audio_separator_core::config::BackendKind::Mvsep => "mvsep",
        }
    );
    for (stem, path) in &result.stems {
        println!("  {stem}: {}", path.display());
    }
    Ok(())
}

/// `separate --backend server`：把任务交给 asep-server 执行，轮询状态并下载分轨。
async fn run_separate_server(client: &ServerClient, args: &SeparateArgs) -> Result<()> {
    // 输入形态：URL → audio_url 字段透传（server 仅 MVSEP 后端支持）；本地文件 → multipart 上传。
    let (audio_path, audio_url) =
        if args.input.starts_with("http://") || args.input.starts_with("https://") {
            (None, Some(args.input.clone()))
        } else {
            (Some(PathBuf::from(&args.input)), None)
        };
    let model = args.model.as_deref().ok_or_else(|| {
        Error::Config(
            "Please specify a model with --model: a manifest name / model download URL / local model path"
                .to_string(),
        )
    })?;
    let submit = SubmitArgs {
        audio_path,
        audio_url,
        model: model.to_string(),
        backend: "local".to_string(),
        format: format_arg_to_server(args.format),
        stems: args.stems.clone().unwrap_or_default(),
        config_url: args.config_url.clone(),
    };
    let task_id = client.submit(&submit).await?;
    println!("Task {task_id} submitted (backend server, model {model})");

    // 轮询：阶段变化打印 [stage]，否则原位刷新进度。
    let mut last_msg = String::new();
    let status = loop {
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        let st = client.status(&task_id).await?;
        // 先克隆字段，避免 loop 内 st 的 move（break st）与后续借用冲突。
        let st_status = st.status.clone();
        let st_message = st.message.clone();
        let st_error = st.error.clone();
        if st_message != last_msg {
            eprintln!();
            eprintln!("[stage] {st_message}");
            last_msg = st_message.clone();
        } else {
            eprint!("\rstatus: {st_status} ({:.0}%)", st.progress * 100.0);
        }
        match st_status.as_str() {
            "done" => break st,
            "failed" => {
                let err = st_error.unwrap_or_else(|| st_message.clone());
                return Err(Error::Backend(format!("task failed: {err}")));
            }
            "cancelled" => {
                return Err(Error::Backend(format!("task cancelled: {st_message}")))
            }
            _ => {}
        }
    };
    eprintln!();
    println!("Separation done (backend server):");

    // 下载：默认全部分轨；--stems 指定子集。
    let stems: Vec<String> = if let Some(s) = &args.stems {
        s.clone()
    } else {
        status.files.iter().map(|f| f.name.clone()).collect()
    };
    if stems.is_empty() {
        println!("(no output files)");
        return Ok(());
    }
    for stem in &stems {
        let path = client.download(&task_id, stem, &args.output).await?;
        println!("  {stem}: {}", path.display());
    }
    Ok(())
}

/// FormatArg → server 侧 format 字符串。
fn format_arg_to_server(f: FormatArg) -> String {
    match f {
        FormatArg::Wav => "wav".to_string(),
        FormatArg::Wav32 => "wav32".to_string(),
        FormatArg::Flac => "flac".to_string(),
        FormatArg::Flac24 => "flac24".to_string(),
        FormatArg::Mp3 => "mp3".to_string(),
        FormatArg::M4a => "m4a".to_string(),
    }
}

/// 模型三态解析：URL → Url；存在的路径 → LocalPath；否则视为 manifest 名字。
/// `arch` 仅对 URL / 本地路径形态生效（按名字时由 manifest 条目决定）。
fn parse_model_ref(s: &Option<String>, arch: Option<String>) -> Result<ModelRef> {
    let s = s.as_ref().ok_or_else(|| {
        Error::Config(
            "Please specify a model with --model: a manifest name / model download URL / local model path"
                .to_string(),
        )
    })?;
    if s.starts_with("http://") || s.starts_with("https://") {
        return Ok(ModelRef::Url {
            url: s.clone(),
            arch,
        });
    }
    if Path::new(s).exists() {
        return Ok(ModelRef::LocalPath {
            path: PathBuf::from(s),
            arch,
        });
    }
    Ok(ModelRef::Name(s.clone()))
}

/// 进度事件打印（stderr，\r 覆盖便于下载/推理进度原位刷新）。
fn print_progress(ev: &ProgressEvent) {
    match ev {
        ProgressEvent::Stage(s) => {
            eprintln!();
            eprintln!("[stage] {s}");
        }
        ProgressEvent::Download { downloaded, total } => {
            let pct = total
                .map(|t| if t > 0 { downloaded * 100 / t } else { 0 })
                .unwrap_or(0);
            let total_str = total
                .map(|t| t.to_string())
                .unwrap_or_else(|| "?".to_string());
            if let Some(t) = total {
                if downloaded >= t {
                    eprintln!("\rdownload complete ({t} bytes)");
                    return;
                }
            }
            eprint!("\rdownloading model… {pct}% ({downloaded} / {total_str} bytes)");
        }
        ProgressEvent::Process { percent, message } => {
            let pct = (percent * 100.0) as u32;
            match message {
                Some(m) => eprint!("\rinferring… {m} ({pct}%)"),
                None => eprint!("\rinferring… {pct}%"),
            }
        }
        ProgressEvent::Writing { stem, percent } => {
            let pct = (percent * 100.0) as u32;
            eprint!("\rwriting stem {stem}… {pct}%");
        }
        ProgressEvent::Finished => {}
    }
}

/// 尚未实现的子命令提示。
fn not_implemented(what: &str, milestone: &str) -> Result<()> {
    Err(Error::Backend(format!(
        "`{what}` is not implemented yet (planned for {milestone}); see PLAN.md §9 milestones"
    )))
}

/// `job-status <hash>`：查询任务状态（MVSEP hash 或 asep-server 任务 id）。
async fn run_job_status(
    id: &str,
    api_key: Option<String>,
    server_url: Option<String>,
    auth_token: Option<String>,
) -> Result<()> {
    if let Some(url) = server_url {
        let client = ServerClient::new(&url, auth_token)?;
        let st = client.status(id).await?;
        println!("Task {id}");
        println!("status: {}", st.status);
        println!("progress: {:.1}%", st.progress * 100.0);
        println!("description: {}", st.message);
        for f in &st.files {
            println!("  - {} ({})", f.name, f.size);
        }
        if let Some(e) = &st.error {
            println!("error: {e}");
        }
        return Ok(());
    }
    let mut cfg = Config::default();
    if let Some(k) = api_key {
        cfg.mvsep.api_key = Some(k);
    }
    let client = MvsepClient::new(&cfg.mvsep, cfg.network.proxy.as_deref())?;
    let st = client.get(id).await?;
    println!("Task {id}");
    println!(
        "status: {}",
        match st.status {
            audio_separator_core::backend::mvsep::MvsepStatus::Done => "done",
            audio_separator_core::backend::mvsep::MvsepStatus::Waiting => "waiting",
            audio_separator_core::backend::mvsep::MvsepStatus::Processing => "processing",
            audio_separator_core::backend::mvsep::MvsepStatus::Distributing => "distributing",
            audio_separator_core::backend::mvsep::MvsepStatus::Merging => "merging",
            audio_separator_core::backend::mvsep::MvsepStatus::Failed => "failed",
            audio_separator_core::backend::mvsep::MvsepStatus::NotFound => "not_found",
        }
    );
    if let Some(a) = &st.algorithm {
        println!("algorithm: {a}");
    }
    if let Some(m) = &st.message {
        println!("description: {m}");
    }
    match (st.queue_count, st.current_order) {
        (Some(q), Some(o)) => println!("queue position: {o} / {q} pending"),
        _ => {}
    }
    for f in &st.files {
        println!(
            "  - {} ({}): {}",
            f.stem,
            f.size.clone().unwrap_or_else(|| "?".to_string()),
            f.url
        );
    }
    if st.files.is_empty() && matches!(st.status, audio_separator_core::backend::mvsep::MvsepStatus::Done) {
        println!("(no output files)");
    }
    Ok(())
}

/// 从 CLI 覆盖项构建配置（config 文件 → 默认 → CLI 覆盖）。
fn build_config(models_file: Option<PathBuf>, models_url: Option<String>) -> Result<Config> {
    let mut cfg = Config::default();
    if let Some(f) = &models_file {
        cfg.models.list = Some(ModelListSource::Path(f.clone()));
    }
    if let Some(u) = &models_url {
        cfg.models.list = Some(ModelListSource::Url(u.clone()));
    }
    Ok(cfg)
}

/// `models`：列出可用模型（本地=manifest / mvsep=平台算法目录）。
async fn run_models(
    backend: BackendArg,
    models_file: Option<PathBuf>,
    models_url: Option<String>,
    server_url: String,
    auth_token: Option<String>,
) -> Result<()> {
    if backend == BackendArg::Mvsep {
        return run_models_mvsep().await;
    }
    if backend == BackendArg::Server {
        return run_models_server(&server_url, auth_token).await;
    }
    let cfg = build_config(models_file, models_url)?;
    let manager = tokio::task::spawn_blocking(move || {
        ModelManager::load(&cfg.models, cfg.network.proxy.as_deref())
    })
    .await
    .map_err(|e| Error::Other(format!("model list load failed: {e}")))??;
    let list = manager.list();
    print_model_list(list);
    Ok(())
}

/// 打印模型清单（本地 manifest / server API 通用）。
fn print_model_list(list: &audio_separator_core::model::ModelList) {
    if list.models.is_empty() {
        println!("model list is empty (models.list not configured or list has no models)");
        return;
    }
    println!("model list v{} ({} models)", list.version, list.models.len());
    for m in &list.models {
        let stems = m.stems.join(", ");
        let mvsep = m
            .mvsep
            .as_ref()
            .map(|mv| format!(" mvsep:{}", mv.sep_type))
            .unwrap_or_default();
        let src = m
            .local_path
            .as_ref()
            .map(|p| format!("local:{}", p.display()))
            .or_else(|| {
                m.source_url
                    .as_ref()
                    .map(|u| format!("download:{}", u))
            })
            .unwrap_or_else(|| "no source".to_string());
        println!(
            "- {} [{} / {}] stems: {} | {}{}",
            m.name, m.architecture, m.engine, stems, src, mvsep
        );
    }
}

/// `models --backend server`：从 asep-server API 拉取本地清单。
async fn run_models_server(server_url: &str, auth_token: Option<String>) -> Result<()> {
    let client = ServerClient::new(server_url, auth_token)?;
    let v = client.models("local").await?;
    let list: audio_separator_core::model::ModelList = serde_json::from_value(v)
        .map_err(|e| Error::Other(format!("invalid server model list: {e}")))?;
    print_model_list(&list);
    Ok(())
}

/// `models --backend mvsep`：拉取平台算法目录（无需 API Key）。
async fn run_models_mvsep() -> Result<()> {
    let cfg = Config::default();
    let client = MvsepClient::new(&cfg.mvsep, cfg.network.proxy.as_deref())?;
    let algos = client.algorithms().await?;
    println!("MVSEP algorithms ({} entries, single_upload):", algos.len());
    let mut grouped: BTreeMap<String, Vec<(u64, String)>> = BTreeMap::new();
    for a in &algos {
        let group = a
            .group
            .as_ref()
            .and_then(|g| g.name.clone())
            .unwrap_or_else(|| "other".to_string());
        grouped
            .entry(group)
            .or_default()
            .push((a.render_id, a.name.clone()));
    }
    for (group, items) in grouped {
        println!("\n[{group}]");
        for (id, name) in items {
            println!("  - {id}: {name}");
        }
    }
    Ok(())
}

/// `model-info`：查看单个模型详情。
async fn run_model_info(
    model: &str,
    models_file: Option<PathBuf>,
    models_url: Option<String>,
) -> Result<()> {
    let cfg = build_config(models_file, models_url)?;
    let manager = tokio::task::spawn_blocking(move || {
        ModelManager::load(&cfg.models, cfg.network.proxy.as_deref())
    })
    .await
    .map_err(|e| Error::Other(format!("model list load failed: {e}")))??;
    let entry = manager
        .list()
        .get(model)
        .ok_or_else(|| Error::Model(format!("model \"{model}\" not found in the manifest")))?;
    println!(
        "{}",
        serde_json::to_string_pretty(entry).map_err(Error::Json)?
    );
    Ok(())
}
