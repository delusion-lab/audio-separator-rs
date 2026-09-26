//! asep CLI：音频分离命令行工具（M0 骨架：完整子命令定义，实现随里程碑交付）。
//!
//! 命令与参数对应 PLAN.md §6 设计：
//! - `separate`：单次分离（M1 起本地后端，M3 起 MVSEP 后端）
//! - `models` / `model-info`：模型清单查询（M1 起）
//! - `job-status`：任务状态查询（M3 起）
//! - `serve`：HTTP 服务（M4 起）

use std::path::PathBuf;

use clap::{Parser, Subcommand};

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
    Wav,
    Flac,
    Mp3,
    M4a,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RegionArg {
    Auto,
    De,
    De2,
    Sg,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Separate(args) => {
            // M0：仅展示解析结果，执行逻辑 M1/M3 交付。
            println!("待执行：separate");
            println!("  输入: {}", args.input);
            println!("  输出目录: {}", args.output.display());
            println!("  后端: {:?}", args.backend);
            if let Some(model) = &args.model {
                println!("  模型: {model}");
            }
            println!("  格式: {:?}", args.format);
            not_implemented("separate", "M1（本地） / M3（MVSEP）");
        }
        Command::Models { .. } => not_implemented("models", "M1"),
        Command::ModelInfo { .. } => not_implemented("model-info", "M1"),
        Command::JobStatus { .. } => not_implemented("job-status", "M3"),
        Command::Serve(_) => not_implemented("serve", "M4"),
    }
}

/// 尚未实现的子命令提示（M0 骨架阶段）。
fn not_implemented(what: &str, milestone: &str) -> ! {
    eprintln!("`{what}` 尚未实现（计划于 {milestone} 交付），参见 PLAN.md §9 里程碑。");
    std::process::exit(1);
}
