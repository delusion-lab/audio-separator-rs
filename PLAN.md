# audio-separator-rs 项目规划（v0.2 草案）

> 状态：架构与关键决策已获用户确认（2026-09-26），Roformer 引擎路线已确认（candle 移植为主、ONNX 导出为备选）；**M0 workspace 骨架已完成（2026-09-26）**；**M1 本地内核 + mdx 端到端已完成（2026-09-26）**；**Roformer spike 结论已出（2026-09-26，见 docs/spike-roformer.md）：放弃 ONNX 导出，candle 为主 + ckpt→safetensors 预转换/运行时安全解析**；**M2-A 权重转换器已完成（2026-09-26）：Rust 受限 pickle 解析器 + torch 存档容器 + safetensors 写入，610MB bs_roformer ep_368 ckpt（实测 dim=512，与 uvr_roformer 参考一致）转换并独立验证通过（无 NaN、形状正确）**。调研依据：MVSEP 官方 API 文档、crates.io / PyPI 音频分离生态、MSST 训练框架与 UVR5 模型体系。

## 1. 项目目标

用 Rust 实现音频分离（人声 / 乐器 / 多分轨），核心能力封装为可复用 crate，提供三种使用形态：



| 形态        | 说明                                                 |
| --------- | -------------------------------------------------- |
| CLI       | 单次运行：`asep separate 输入.mp3 -o 输出目录`，本地或 MVSEP 后端均可 |
| 服务端       | 常驻进程，HTTP API 持续接收任务（上传 / URL），异步执行、查询、下载          |
| MVSEP 云后端 | 使用平台 API Key，把分离任务提交给 MVSEP 云端执行（支持 130+ 模型）       |

本地推理同时纳入（多架构，见 §5.3），作为不依赖外部平台的默认后端。

## 2. 背景调研（已核实的事实）

### 2.1 MVSEP API（官方文档 [https://mvsep.com/zh/full\_api](https://mvsep.com/zh/full_api)）



* **创建任务**：`POST https://mvsep.com/api/separation/create`


  * 表单参数：`api_token`（API Key）、`audiofile`（二进制上传）或 `url`（远程链接，可选 `remote_type`：direct/mega/drive/dropbox）、`sep_type`（模型 ID，默认 20）、`add_opt1/2/3`（模型专属选项）、`output_format`（0=mp3 320k，1=wav16，2=flac16，3=m4a，4=wav32，5=flac24）、`is_demo`、`webhook_url`、`preset_id`（预设会覆盖 sep\_type 等）

  * 响应：`{success, data:{link, hash}}` 或 `{success:false, data:{message}}`

* **轮询结果**：`GET https://mvsep.com/api/separation/get?hash=<hash>`（可选 `mirror`、`api_token`），状态流转 `waiting → processing → done`，`data.files` 含分轨文件

* **其他端点**：取消分离、站点信息、获取分离类型列表（模型目录）、队列状态等

* **区域**：de /de2 /sg 区域端点，主域名 `https://mvsep.com/api` 按地理位置自动分流；下载链接由创建区域提供（绝对地址）

* **配额**：非 Premium 仅 1 个并发任务；Premium 无限并发、按积分计费、更高文件上限（1000MB / 100 分钟）

* **错误码**：400 参数缺失 / 无效；401 api\_token 未知 / 无效

* **官方示例**：[https://github.com/ZFTurbo/MVSep-API-Examples](https://github.com/ZFTurbo/MVSep-API-Examples) （Python）

### 2.2 本地多架构分离生态



* **ONNX 成熟家族**：MDX-Net（UVR-MDX-NET 系列）、VR Arch、Demucs（htdemucs）、MDXC 等，UVR5 与 `audio-separator`（Python）均以 ONNX 推理，模型文件即 `.onnx` 资产，可直接下载分发。

* **Roformer 家族（当前为 PyTorch 权重，ONNX 不成熟）**：


  * BS-Roformer：字节跳动 Band-Split RoPE Transformer（arXiv:2309.02612，SDX23 冠军）；社区权重如 `model_bs_roformer_ep_368_sdr_12.9628.ckpt`。

  * MelBand Roformer：mel 频带映射 + 轴向注意力；`audio-separator` 模型清单含 `mel_band_roformer_kim_ft_unwa.ckpt` 等。

  * BS PolarFormer：极坐标嵌入变体（hunterhogan/hunterFormsBS 的 `BandSplitRotator`），权重 `model_bs_polarformer_float16.ckpt`（HuggingFace / MSST release），MVSEP 已上线该算法。

  * 这些架构主流推理路径是 PyTorch（UVR5 内部、`bs-roformer-infer`、`melband-roformer-infer` 均为 Python）。

* **训练 / 推理参考实现**：MSST（ZFTurbo/Music-Source-Separation-Training，Apache-2.0）统一框架，含 BSRoformer / MelBandRoformer / SCNet 等实现与预训练模型清单；lucidrains 的 BS-RoFormer 纯 PyTorch 实现。

* **模型清单先例**：`melband-roformer-infer` 内置 89 个模型的注册表、首次使用自动下载 + sha256 校验 —— 与本项目 "模型清单 JSON + 懒下载" 设计同构。

* **纯 Rust 音频栈**：解码 symphonia（多格式）；STFT/iSTFT 可基于 realfft；编码 hound（WAV）/flacenc（FLAC）；MP3 需 mp3lame C 绑定或暂缓。

## 3. 已确认决策（2026-09-26 用户确认）



| 编号 | 决策       | 结论                                                                                                                                         |
| -- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| D1 | 本地后端实现路线 | **A 方案：自研推理内核**，不依赖第三方分离库；**多架构支持**（类似 UVR5），架构注册表机制，覆盖 MDX-Net / VR / Demucs / MDXC 及 **BS-Roformer / MelBand Roformer / BS PolarFormer** |
| D2 | 分轨与参数归属  | **分轨数由模型定义**（模型条目声明 `stems`）；**参数按架构隔离**—— 每种架构有专属参数 schema 与推理实现，**不允许跨架构复用参数**（不采用 UVR5 把 MDX 参数套给 Roformer 的做法）                         |
| D3 | 服务端持久化   | v1 内存任务表，预留持久化 trait（后续 SQLite）                                                                                                            |
| D4 | 服务端鉴权    | Bearer Token，默认关闭，文档提示启用                                                                                                                   |

## 4. Cargo Workspace 结构



```
audio-separator-rs/
├── Cargo.toml                     # [workspace] members = crates/*
├── crates/
│   ├── audio-separator-core/      # 核心库：后端抽象、作业、配置、模型管理、IO、错误
│   ├── audio-separator-cli/       # 二进制：CLI（clap）
│   └── audio-separator-server/    # 二进制：HTTP 服务（axum + tokio）
├── docs/
├── config.example.toml
└── PLAN.md
```

依赖方向：`cli` / `server` → `core`。同一 crate 名即最终对外发布的库名。

## 5. 核心 crate 设计（audio-separator-core）

### 5.1 模块划分



```
audio-separator-core/
├── src/
│   ├── lib.rs
│   ├── config.rs       # Config（TOML/env）、BackendKind、输出格式枚举
│   ├── error.rs        # 错误类型：Backend、Network、Format、Model、Quota、Io
│   ├── backend/
│   │   ├── mod.rs      # Separator trait + BackendFactory
│   │   ├── local/      # 本地推理：model_manage + engine（onnx / candle）
│   │   └── mvsep.rs    # MVSEP 客户端（create/poll/cancel/webhook）
│   ├── job.rs          # 作业状态机、进度事件、取消令牌
│   ├── io.rs           # 音频解码（symphonia）、输出编码、文件名规则
│   └── model.rs        # 模型清单 manifest + 架构注册表（见 §5.3）
```

### 5.2 核心抽象



```
pub enum ModelRef {
    Name(String),                       // 按名查 manifest（懒下载）
    Url { url: String, arch: Option<String> },      // 直接指定下载链接（可显式指定架构）
    LocalPath { path: PathBuf, arch: Option<String> }, // 用户已有模型文件
}

pub struct SeparationRequest {
    pub input: Input,                   // Path / Url / Bytes
    pub model: ModelRef,
    pub output_format: OutputFormat,    // Wav16 / Flac16 / Mp3 / ...
    pub output_dir: PathBuf,
    pub select_stems: Option<Vec<Stem>>, // 可选：只输出部分分轨（分轨全集由模型定义）
}

pub struct SeparationResult {
    pub stems: BTreeMap<Stem, PathBuf>, // Stem: Vocals/Drums/Bass/Other/Instrumental...
    pub backend: BackendKind,
    pub elapsed: Duration,
}

#[async_trait]
pub trait Separator: Send + Sync {
    async fn separate(
        &self,
        req: SeparationRequest,
        progress: Option<mpsc::Sender<ProgressEvent>>,
        cancel: Option<CancellationToken>,
    ) -> Result<SeparationResult>;
}
```



* 本地推理为阻塞密集操作：`spawn_blocking` 包裹，进度经 channel 上报。

* MVSEP 后端：网络异步，并发信号量（非 Premium 恒 1，可配置），轮询带回退与超时，支持透传 `webhook_url`。

* 同一 `ModelRef` 语义对 CLI 与服务端完全一致。

### 5.3 模型管理体系（本地后端核心）

**设计原则**：模型是独立资产（文件 + 清单条目），架构是推理代码（引擎 + 参数 schema）。两者分离，参数绝不跨架构混用。

#### 5.3.1 模型清单 manifest JSON



* 模型列表来自一个 JSON 文件：**本地路径或 URL**（如 GitHub raw），`config.models.list` 指定，默认指向独立模型仓库（M3 初始化）。

* 条目 schema（示例，具体参数名以对应实现为准）：



```
{
  "version": 1,
  "models": [
    {
      "name": "UVR-MDX-NET-Inst_HQ_3",
      "architecture": "mdx",
      "engine": "onnx",
      "source_url": "https://github.com/TRvlvr/model_repo/releases/download/.../UVR-MDX-NET-Inst_HQ_3.onnx",
      "sha256": "…",
      "stems": ["vocals", "instrumental"],
      "license": "…",
      "params": {
        "segment_size": 256, "overlap": 0.25, "batch_size": 1,
        "hop": 1024, "dim_f": 3072, "dim_t": 256, "n_fft": 6144,
        "num_bands": 48, "num_blocks": 12, "sample_rate": 44100
      }
    },
    {
      "name": "model_bs_roformer_ep_368_sdr_12.9628",
      "architecture": "bs_roformer",
      "engine": "candle",
      "source_url": "https://github.com/TRvlvr/model_repo/releases/download/.../model_bs_roformer_ep_368_sdr_12.9628.ckpt",
      "sha256": "…",
      "stems": ["vocals", "instrumental"],
      "license": "…",
      "params": {
        "dim": 384, "depth": 12, "heads": 8, "dim_head": 64,
        "attn_dropout": 0.1, "ff_dropout": 0.1, "flash_attn": true,
        "mask_estimator_depth": 3, "multi_stems": false, "sample_rate": 44100
      }
    },
    {
      "name": "mel_band_roformer_kim_ft_unwa",
      "architecture": "mel_band_roformer",
      "engine": "candle",
      "source_url": "…",
      "sha256": "…",
      "stems": ["vocals", "instrumental"],
      "license": "…",
      "params": { "n_mels": 96, "dim": 384, "depth": 12, "heads": 8, "…": "…" }
    },
    {
      "name": "model_bs_polarformer_float16",
      "architecture": "bs_polarformer",
      "engine": "candle",
      "source_url": "…",
      "sha256": "…",
      "stems": ["vocals", "instrumental"],
      "license": "…",
      "params": { "segment": 60, "overlap": 0.25, "…": "…" }
    }
  ]
}
```

#### 5.3.2 架构注册表（architecture registry）



* 每种架构注册为：**推理内核实现**（STFT/iSTFT 前后处理 + 引擎调用）+ **参数 schema**（serde 反序列化 + 校验）+ 权重格式（`.onnx` / `.ckpt` / `.safetensors`）。

* 参数 schema 按架构独立定义与校验 ——`mdx` 的参数集与 `bs_roformer` 的参数集互不可见、不可互填（避免 UVR5 式混用）。

* v0.2 计划覆盖架构（引擎双轨）：



| 架构                      | 引擎     | 分轨示例                      | 说明                     |
| ----------------------- | ------ | ------------------------- | ---------------------- |
| mdx（MDX-Net）            | onnx   | vocals/instrumental 或 4 轨 | UVR-MDX-NET 系列，ONNX 成熟 |
| vr（VR Arch）             | onnx   | vocals/instrumental       | UVR VR 系列，ONNX 成熟      |
| demucs（htdemucs）        | onnx   | vocals/drums/bass/other   | Demucs v4 ONNX 转换可用    |
| mdxc（MDXC / MelBand 变体） | onnx   | vocals/other              | 社区 ONNX 转换             |
| bs\_roformer            | candle | vocals/instrumental（可多轨）  | PyTorch .ckpt，M2 移植    |
| mel\_band\_roformer     | candle | vocals/instrumental       | PyTorch .ckpt，M2 移植    |
| bs\_polarformer         | candle | vocals/instrumental       | 极坐标嵌入变体，M2 移植          |

> **引擎双轨的由来与 spike 结论**
>
> ：调研确认 Roformer 家族当前无成熟 ONNX 生态（主流为 PyTorch 权重）。M1 spike（2026-09-26，见 `docs/spike-roformer.md`）结论：**放弃 ONNX 导出**——纯 Rust 直接加载原始 `.ckpt` 已有先例（`uvr_roformer`，Burn 引擎；`pt-loader` / `anamnesis` 提供 ckpt→safetensors 转换）。因此走 **candle 引擎 + ckpt→safetensors 预转换（或运行时安全解析）**，从 MSST /lucidrains/uvr_roformer 移植各架构前向（复数 STFT、band-split、RoPE、轴向注意力、mel 映射均在 Rust 实现）。candle 为纯 Rust 推理框架，符合 "无 Python 依赖" 目标。

#### 5.3.3 模型解析与获取流程



```
ModelRef 解析优先级：
1. Name    → 查 manifest → 条目含 local_path？直接使用
              └ 否则按 source_url 懒下载（首次使用该名字才下载）→ 缓存 + sha256 校验
2. Url     → 直接下载到缓存（可带显式 arch，否则从 URL / 文件头嗅探）
3. LocalPath → 直接使用（仅需 arch 或自动嗅探）
```



* 缓存目录：系统缓存目录下 `audio-separator-rs/models/`，条目级 sha256 校验；支持版本 / 更新提示。

* 清单源：`config.models.list = "path/to/models.json" | "https://…/models.json"`，默认指向独立 GitHub 模型仓库（M3 初始化该仓库，集中维护各架构模型条目与下载链接）。

* 安全：远程清单条目必须携带 `sha256`，下载后校验，防供应链篡改；`license` 字段随条目展示。

## 6. CLI 设计（audio-separator-cli）



```
asep separate <输入文件|URL> -o <输出目录>
    [--backend local|mvsep]           # 默认 local
    [--model <名字|下载URL|本地路径>]   # 三态均可；名字命中 manifest 才懒下载
    [--stems vocals,drums]            # 输出过滤；分轨全集由所选模型定义
    [--format wav|flac|mp3]           # 默认 wav
    [--api-key <token>] [--region auto|de|de2|sg]
    [--poll-timeout <秒>] [--concurrency <n>]
    [--models-file <json路径>] [--models-url <json URL>]   # 覆盖模型清单源
    [--config <toml>] [-v...]

asep models [--backend local|mvsep]   # 列出可用模型（本地=manifest，mvsep=API 拉取）
asep model info <名字>                # 查看模型：架构、引擎、分轨、参数、来源
asep job status <hash|job-id>         # 查询 MVSEP 任务或本地作业
asep serve [--addr <ip:port>] [--config <toml>]
```



* 配置优先级：命令行 > 环境变量（`ASEP_MVSEP_API_KEY`、`ASEP_CONFIG`）> `config.toml`。

* CLI 与 Server 共用 `core::config` 与 `core::backend`，保证行为一致。

## 7. 服务端设计（audio-separator-server）



* 栈：`axum` + `tokio`，`tower-http`（CORS / 限流可选）。



| 方法   | 路径                                | 说明                                     |
| ---- | --------------------------------- | -------------------------------------- |
| POST | /v1/separations                   | multipart 上传或 `{url}` JSON；返回 `{id}`   |
| GET  | /v1/separations/{id}              | 状态（queued/running/done/failed）+ 分轨文件列表 |
| GET  | /v1/separations/{id}/files/{stem} | 下载分轨文件                                 |
| POST | /v1/separations/{id}/cancel       | 取消（本地终止 / MVSEP 取消）                    |
| GET  | /v1/models                        | 可用后端与模型                                |
| GET  | /v1/health                        | 健康检查                                   |



* 任务模型：内存任务表 `Arc<RwLock<HashMap<Id, TaskState>>>`；每个任务一个 tokio 任务；预留持久化 trait（D3）。

* 本地执行：worker 池（并发数可配置），大文件上传流式落盘（限制文件大小）。

* MVSEP 执行：并发信号量（默认 1）+ 轮询循环，任务 id ↔ MVSEP hash 映射，透传 webhook。

* 鉴权：可选 Bearer Token（D4），默认关闭。

## 8. 数据流



```
CLI 路径:   输入文件 → io.decode → 模型解析(manifest/懒下载) → backend.separate → stems → io.encode → 输出目录
服务端路径: 上传 → 任务表 → worker(local 推理 | mvsep 提交) → 结果文件 → 下载 API
MVSEP 路径: create(hash) → poll get(waiting→processing→done) → 下载 files → 按需转码/重命名 → 交付
```

## 9. 里程碑



| 阶段 | 内容                                                                                                                                                             | 验收                                |
| -- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| M0 | workspace 骨架、core 类型（config/error/job/model）、CLI 骨架                                                                                                            | ✅ 已完成：`cargo build` 通过，`asep --help` 正常 |
| M1 | **本地内核 + 模型管理**：onnx 引擎、manifest JSON（本地 + URL）、懒下载与 sha256 校验、URL / 本地路径模型、架构注册表骨架；**mdx 架构端到端**；spike：Roformer 家族 ONNX 导出可行性结论                               | ✅ 已完成（2026-09-26）：`UVR_MDXNET_9482` 端到端分出人声 / 伴奏（合成与真实歌曲均验证，两轨相关系数 0.044）；`asep separate / models / model-info` 可用；Roformer spike 结论已出（docs/spike-roformer.md） |
| M2 | **权重转换与 Roformer 移植**：M2-A✅（2026-09-26）ckpt→safetensors 转换器（受限 pickle 解析 + 懒转换缓存）；M2-B bs\_roformer（ep\_368 实测 dim=512，参考 uvr\_roformer 结构一致）→ mel\_band\_roformer → bs\_polarformer，candle 引擎，参数 schema 对齐 lucidrains/MSST；**onnx 家族扩展**：demucs/mdxc（复用 mdx 管线）；vr 架构原始权重为 .pth（UVR 生态），格式与推理管线核实后接入 | 三款 Roformer 模型 CLI 可跑，参数按架构校验     |
| M3 | MVSEP 后端：客户端（create/poll/cancel/webhook）、模型目录映射、CLI 单次分离；**初始化模型清单 GitHub 仓库**                                                                                 | 真实 API Key 跑通人声 / 伴奏分离            |
| M4 | 服务端：axum 上传 / 任务 / 下载 / 取消、并发控制、Bearer Token                                                                                                                   | curl 全流程：上传→轮询→下载                 |
| M5 | 完善：FLAC/MP3 输出、webhook、单元 / 集成测试、README、CI、Docker                                                                                                              | 文档与测试齐备                           |

## 10. 风险与注意事项



* **Roformer 家族无成熟 ONNX 生态**（已核实）：M1 spike 先行；若 ONNX 导出不可行，candle 引擎的 Rust 前向移植（复数 STFT、band-split、RoPE、轴向注意力、mel 映射）是 M2 最大工作量来源，需按架构逐个验证输出与参考实现一致。

* `ort` 首次构建需下载 ONNX Runtime 二进制（网络依赖）；candle 为纯 Rust（CPU 起步，CUDA/Metal 可选）。

* 复数域 STFT/FFT 在 ONNX 导出与 Rust 移植中均有坑（参照 ADC25 报告），前后处理需与引擎分离、单独测试。

* MVSEP 非 Premium 单并发 + 积分计费：需排队管理与配额 / 余额监控提示。

* 模型资产信任：远程清单条目强制 `sha256` 校验；license 随条目声明（UVR / 社区模型许可各异）。

* 纯 Rust 编码 MP3/M4A 受限：本地 v1 提供 WAV/FLAC，MP3 视 M5 评估 mp3lame 绑定。

* 服务端大文件：流式落盘 + 大小上限，避免内存峰值。

## 11. 待确认清单（剩余）



1. ~~D1-D4~~ 已确认（§3）。

2. ~~M1 spike 结论前，Roformer 家族引擎路线以 "candle 移植" 为主计划、ONNX 导出为备选 —— 是否认可该主备次序？~~ **已确认**：candle 移植为主、ONNX 导出为备选（2026-09-26）。

3. 模型清单默认仓库（GitHub 组织 / 用户名占位，M3 初始化）。

4. 服务端目标场景与预期并发（影响 worker 池与限流设计，M4 前确认即可）。