//! asep CLI：音频分离命令行工具。
//!
//! - `separate`：单次分离（本地后端 M1 起可用；MVSEP 后端 M3 交付）
//! - `models` / `model-info`：模型清单查询（M1 后半交付）
//! - `job-status`：任务状态查询（M3 交付）
//! - `serve`：HTTP 服务（M4 交付）

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use audio_separator_core::backend::local::model_manage::ModelManager;
use audio_separator_core::backend::local::LocalSeparator;
use audio_separator_core::backend::mvsep::{MvsepClient, MvsepSeparator};
use audio_separator_core::backend::{Input, SeparationRequest, Separator};
use audio_separator_core::config::{Config, ModelListSource, MvsepRegion};
use audio_separator_core::error::{Error, Result};
use audio_separator_core::job::ProgressEvent;
use audio_separator_core::model::{ModelRef, OutputFormat};
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

/// 音频分离工具：本地多架构推理 / MVSEP 云 API / HTTP 服务。
#[derive(Parser)]
#[command(name = "asep", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 单次分离一个音频文件（本地或 MVSEP 后端）。
    Separate(SeparateArgs),

    /// 列出可用模型（本地=manifest / mvsep=平台 API）。
    Models {
        /// 后端。
        #[arg(long, value_enum, default_value_t = BackendArg::Local)]
        backend: BackendArg,
        /// 覆盖模型清单源：本地 JSON 路径。
        #[arg(long)]
        models_file: Option<PathBuf>,
        /// 覆盖模型清单源：远程 JSON URL。
        #[arg(long)]
        models_url: Option<String>,
    },

    /// 查看单个模型详情（架构、引擎、分轨、参数、来源）。
    #[command(name = "model-info")]
    ModelInfo {
        /// 模型名 / 下载 URL / 本地路径。
        model: String,
        /// 覆盖模型清单源：本地 JSON 路径。
        #[arg(long)]
        models_file: Option<PathBuf>,
        /// 覆盖模型清单源：远程 JSON URL。
        #[arg(long)]
        models_url: Option<String>,
    },

    /// 查询任务状态（MVSEP hash 或本地作业 id）。
    #[command(name = "job-status")]
    JobStatus {
        /// 任务 hash 或作业 id。
        id: String,
        /// MVSEP API Key（查询 MVSEP 任务时）。
        #[arg(long, env = "ASEP_MVSEP_API_KEY")]
        api_key: Option<String>,
    },

    /// 启动 HTTP 服务（M4 实现）。
    Serve(ServeArgs),
}

/// `separate` 子命令参数。
#[derive(clap::Args)]
struct SeparateArgs {
    /// 输入音频文件或远程 URL。
    input: String,

    /// 输出目录。
    #[arg(short, long)]
    output: PathBuf,

    /// 后端：local / mvsep。
    #[arg(long, value_enum, default_value_t = BackendArg::Local)]
    backend: BackendArg,

    /// 模型：名字 / 下载 URL / 本地路径（名字命中 manifest 才懒下载）。
    #[arg(long)]
    model: Option<String>,

    /// 显式指定架构（仅对 URL / 本地路径形态生效；按名字时以 manifest 为准）。
    /// 取值如 mdx / bs_roformer / mel_band_roformer / bs_polarformer。
    #[arg(long)]
    arch: Option<String>,

    /// 模型参数配置（yaml/json）的 URL 或本地路径。传了则作为架构参数的权威来源，
    /// 与 --model 的 URL / 本地路径形态搭配可直接使用，无需 models.json 条目；
    /// 按名引用时覆盖清单条目的 config_url。
    #[arg(long)]
    config_url: Option<String>,

    /// 只输出指定分轨（逗号分隔）；分轨全集由所选模型定义。
    #[arg(long, value_delimiter = ',')]
    stems: Option<Vec<String>>,

    /// 输出格式。
    #[arg(long, value_enum, default_value_t = FormatArg::Wav)]
    format: FormatArg,

    /// MVSEP API Key（backend=mvsep 时必需）。
    #[arg(long, env = "ASEP_MVSEP_API_KEY")]
    api_key: Option<String>,

    /// MVSEP 区域。
    #[arg(long, value_enum, default_value_t = RegionArg::Auto)]
    region: RegionArg,

    /// MVSEP 轮询超时（秒）。
    #[arg(long)]
    poll_timeout: Option<u64>,

    /// MVSEP 并发任务数（非 Premium 仅 1）。
    #[arg(long)]
    concurrency: Option<usize>,

    /// MVSEP 附加选项 add_opt1（backend=mvsep 时透传，覆盖清单条目映射）。
    #[arg(long)]
    add_opt1: Option<String>,

    /// MVSEP 附加选项 add_opt2。
    #[arg(long)]
    add_opt2: Option<String>,

    /// MVSEP 附加选项 add_opt3。
    #[arg(long)]
    add_opt3: Option<String>,

    /// MVSEP 完成回调 URL（处理后平台 POST 结果，免轮询）。
    #[arg(long)]
    webhook_url: Option<String>,

    /// 覆盖模型清单源：本地 JSON 路径。
    #[arg(long)]
    models_file: Option<PathBuf>,

    /// 覆盖模型清单源：远程 JSON URL。
    #[arg(long)]
    models_url: Option<String>,

    /// 配置文件路径。
    #[arg(long)]
    config: Option<PathBuf>,

    /// 日志详细程度（可重复）。
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

/// `serve` 子命令参数。
#[derive(clap::Args)]
struct ServeArgs {
    /// 监听地址。
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum FormatArg {
    /// 16-bit PCM WAV（默认）。
    Wav,
    /// 32-bit float WAV。
    Wav32,
    /// 16-bit FLAC。
    Flac,
    /// 24-bit FLAC。
    Flac24,
    /// MP3 320kbps。
    Mp3,
    /// M4A（仅 MVSEP 后端支持）。
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
        } => run_models(backend, models_file, models_url).await,
        Command::ModelInfo {
            model,
            models_file,
            models_url,
        } => run_model_info(&model, models_file, models_url).await,
        Command::JobStatus { id, api_key } => run_job_status(&id, api_key).await,
        Command::Serve(_) => not_implemented("serve", "M4"),
    };
    if let Err(e) = result {
        eprintln!("错误: {e}");
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
                    .map_err(|e| Error::Other(format!("本地后端初始化异常: {e}")))??,
            )
        }
        BackendArg::Mvsep => {
            // 加载清单（manifest 的 mvsep 映射用于按名引用）；网络型清单在阻塞线程拉取。
            let cfg_clone = cfg.clone();
            let manager = tokio::task::spawn_blocking(move || {
                ModelManager::load(&cfg_clone.models, cfg_clone.network.proxy.as_deref())
            })
            .await
            .map_err(|e| Error::Other(format!("模型清单加载异常: {e}")))??;
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
        "分离完成（{:.1}s，后端 {}）：",
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

/// 模型三态解析：URL → Url；存在的路径 → LocalPath；否则视为 manifest 名字。
/// `arch` 仅对 URL / 本地路径形态生效（按名字时由 manifest 条目决定）。
fn parse_model_ref(s: &Option<String>, arch: Option<String>) -> Result<ModelRef> {
    let s = s.as_ref().ok_or_else(|| {
        Error::Config(
            "请用 --model 指定模型：manifest 中的名字 / 模型下载 URL / 本地模型路径"
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
            eprintln!("[阶段] {s}");
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
                    eprintln!("\r下载完成（{t} 字节）");
                    return;
                }
            }
            eprint!("\r下载模型… {pct}%（{downloaded} / {total_str} 字节）");
        }
        ProgressEvent::Process { percent, message } => {
            let pct = (percent * 100.0) as u32;
            match message {
                Some(m) => eprint!("\r推理… {m} ({pct}%)"),
                None => eprint!("\r推理… {pct}%"),
            }
        }
        ProgressEvent::Writing { stem, percent } => {
            let pct = (percent * 100.0) as u32;
            eprint!("\r写入分轨 {stem}… {pct}%");
        }
        ProgressEvent::Finished => {}
    }
}

/// 尚未实现的子命令提示。
fn not_implemented(what: &str, milestone: &str) -> Result<()> {
    Err(Error::Backend(format!(
        "`{what}` 尚未实现（计划于 {milestone} 交付），参见 PLAN.md §9 里程碑"
    )))
}

/// `job-status <hash>`：查询 MVSEP 任务状态与输出文件。
async fn run_job_status(id: &str, api_key: Option<String>) -> Result<()> {
    let mut cfg = Config::default();
    if let Some(k) = api_key {
        cfg.mvsep.api_key = Some(k);
    }
    let client = MvsepClient::new(&cfg.mvsep, cfg.network.proxy.as_deref())?;
    let st = client.get(id).await?;
    println!("任务 {id}");
    println!(
        "状态: {}",
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
        println!("算法: {a}");
    }
    if let Some(m) = &st.message {
        println!("说明: {m}");
    }
    match (st.queue_count, st.current_order) {
        (Some(q), Some(o)) => println!("排队: {o} 号 / 共 {q} 个待处理"),
        _ => {}
    }
    for f in &st.files {
        println!(
            "  - {}（{}）: {}",
            f.stem,
            f.size.clone().unwrap_or_else(|| "?".to_string()),
            f.url
        );
    }
    if st.files.is_empty() && matches!(st.status, audio_separator_core::backend::mvsep::MvsepStatus::Done) {
        println!("（无输出文件）");
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
) -> Result<()> {
    if backend == BackendArg::Mvsep {
        return run_models_mvsep().await;
    }
    let cfg = build_config(models_file, models_url)?;
    let manager = tokio::task::spawn_blocking(move || {
        ModelManager::load(&cfg.models, cfg.network.proxy.as_deref())
    })
    .await
    .map_err(|e| Error::Other(format!("模型清单加载异常: {e}")))??;
    let list = manager.list();
    if list.models.is_empty() {
        println!("清单为空（未配置 models.list 或清单无模型）");
        return Ok(());
    }
    println!("模型清单 v{}（{} 个模型）", list.version, list.models.len());
    for m in &list.models {
        let stems = m.stems.join("、");
        let mvsep = m
            .mvsep
            .as_ref()
            .map(|mv| format!(" mvsep:{}", mv.sep_type))
            .unwrap_or_default();
        let src = m
            .local_path
            .as_ref()
            .map(|p| format!("本地:{}", p.display()))
            .or_else(|| {
                m.source_url
                    .as_ref()
                    .map(|u| format!("下载:{}", u))
            })
            .unwrap_or_else(|| "无来源".to_string());
        println!(
            "- {} [{} / {}] 分轨: {} | {}{}",
            m.name, m.architecture, m.engine, stems, src, mvsep
        );
    }
    Ok(())
}

/// `models --backend mvsep`：拉取平台算法目录（无需 API Key）。
async fn run_models_mvsep() -> Result<()> {
    let cfg = Config::default();
    let client = MvsepClient::new(&cfg.mvsep, cfg.network.proxy.as_deref())?;
    let algos = client.algorithms().await?;
    println!("MVSEP 算法目录（{} 项，single_upload）：", algos.len());
    let mut grouped: BTreeMap<String, Vec<(u64, String)>> = BTreeMap::new();
    for a in &algos {
        let group = a
            .group
            .as_ref()
            .and_then(|g| g.name.clone())
            .unwrap_or_else(|| "其他".to_string());
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
    .map_err(|e| Error::Other(format!("模型清单加载异常: {e}")))??;
    let entry = manager
        .list()
        .get(model)
        .ok_or_else(|| Error::Model(format!("模型「{model}」不在清单中")))?;
    println!(
        "{}",
        serde_json::to_string_pretty(entry).map_err(Error::Json)?
    );
    Ok(())
}
