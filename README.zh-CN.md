# audio-separator-rs

[English](README.md) | **中文** | [日本語](README.ja.md) | [한국어](README.ko.md)

Rust 实现的人声 / 伴奏分离工具。核心逻辑独立成 crate，提供三种使用形态：

- **CLI**（`asep`）：单次运行，本地推理、MVSEP 云 API 或连接自己的 asep-server；
- **HTTP 服务端**（`audio-separator-server`）：常驻进程，提供上传 / 任务 / 下载 / 取消 REST 接口；
- **crate**（`audio-separator-core`）：作为库嵌入其它 Rust 程序。

## 架构

```
crates/
├── audio-separator-core   核心：模型管理、推理引擎、音频 IO、MVSEP 客户端
├── audio-separator-cli    CLI 入口（二进制 asep）
└── audio-separator-server axum HTTP 服务端（任务队列 + 双后端）
```

### 本地推理引擎（后端 local）

| 架构 | 引擎 | 状态 |
| --- | --- | --- |
| mdx（UVR-MDX-NET） | ONNX Runtime | ✅ |
| bs_roformer | candle（纯 Rust，f64 STFT + band-split + 双轴向注意力） | ✅ |
| mel_band_roformer | candle（Slaney mel 滤波器组 + MSST 结构） | ✅ |
| bs_polarformer | candle（PoPE 极坐标嵌入） | ✅ |

Roformer 权重以 `ckpt`（pickle）或 `safetensors` 提供：`ckpt` 首次使用自动转换并缓存为 `safetensors`（懒转换）。CPU 推理，性能与参考实现同量级（详见 `PLAN.md` §10）。

### MVSEP 云后端（后端 mvsep）

调用 [MVSEP API](https://mvsep.com/zh/full_api) 完成分离。非 Premium 账号单并发、每天有分离额度，代码内置信号量限流。

## 快速开始

### 构建

需要 Rust 工具链；本地 ORT（ONNX Runtime 1.28）配置见下文「本地构建说明」。

```sh
cargo build --release --workspace
```

产物：`target/release/asep.exe`、`target/release/audio-separator-server.exe`。

### CLI 分离（本地）

```sh
# 首次使用某模型时自动从模型清单下载权重（清单见 models.json）
asep separate input.wav -o out/ --model model_bs_polarformer_float16

# 指定输出格式（wav/wav32/flac/flac24/mp3/m4a）
asep separate input.wav -o out/ --model UVR_MDXNET_9482 --format flac

# 直传模型权重 + 参数配置（yaml/json），完全绕过 models.json：
#   --model 支持 下载 URL / 本地模型路径（+ --arch 指定架构）
#   --config-url 支持 参数文件 URL / 本地路径，作为架构参数的权威来源
asep separate input.wav -o out/ --model ./model.ckpt --arch bs_polarformer --config-url ./config.yaml
asep separate input.wav -o out/ --model https://.../model.ckpt --arch mel_band_roformer --config-url https://.../config.yaml
```

### CLI 分离（MVSEP 云）

```sh
asep separate input.wav -o out/ --backend mvsep \
  --model model_bs_polarformer_float16 --api-key YOUR_MVSEP_KEY --format flac

# 或通过环境变量提供 Key
ASEP_MVSEP_API_KEY=YOUR_KEY asep separate input.wav -o out/ --backend mvsep --model ...
```

### CLI 分离（自己的 server）

先启动你的 asep-server，再用 CLI 驱动它——CLI 负责上传文件、轮询任务并下载分轨：

```sh
# 终端 1：启动服务端（默认 127.0.0.1:8080）
audio-separator-server

# 终端 2：经服务端分离
asep separate input.wav -o out/ --backend server --model model_bs_polarformer_float16

# 远端服务端 / Bearer 鉴权
asep separate input.wav -o out/ --backend server \
  --server-url http://10.0.0.5:8080 --auth-token secret --model UVR_MDXNET_9482

# 查询服务端模型清单 / 查看服务端任务
asep models --backend server --server-url http://127.0.0.1:8080
asep job-status <task_id> --server-url http://127.0.0.1:8080
```

说明：本地文件会上传到服务端；URL 输入以 `audio_url` 字段透传（服务端仅 MVSEP 后端支持 URL 输入）。服务端需先自行启动，CLI 不会代为启动。客户端直连 `--server-url`，不经过下载代理。

### 服务端

```sh
# 启动（默认 127.0.0.1:8080）
audio-separator-server --api-key YOUR_MVSEP_KEY --workers 1

# 启用鉴权（Bearer Token，默认关闭）
audio-separator-server --auth-token secret
```

REST 接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/separate` | multipart 提交任务（`audio` 文件或 `audio_url` + `model` + `backend` + `format` + 可选 `config_url`） |
| GET | `/api/v1/tasks/{id}` | 查询任务状态（queued/running/done/failed/cancelled + 进度） |
| GET | `/api/v1/tasks/{id}/download?stem=` | 下载分轨结果 |
| DELETE | `/api/v1/tasks/{id}` | 取消运行中任务或清理终态任务 |
| GET | `/api/v1/models?backend=local\|mvsep` | 模型列表 / MVSEP 算法目录 |
| GET | `/api/v1/health` | 健康检查 |

上传示例：

```sh
curl -F "audio=@input.wav" -F "model=model_bs_polarformer_float16" \
  -F "backend=local" -F "format=flac" http://127.0.0.1:8080/api/v1/separate
```

## 模型清单（models.json）

模型元信息（下载地址、sha256、架构参数、MVSEP 映射等）由 JSON 文件维护：

- 默认读取仓库内 `models.json`；
- `--models-url <url>` / 配置 `models.list = { url = "..." }` 可拉取远程清单（本地或 URL 均可）；
- 清单独立维护在 GitHub 仓库 [delusion-lab/asep-models](https://github.com/delusion-lab/asep-models)（仅托管清单 JSON，权重留在官方源）。

条目示例：

```json
{
  "name": "model_bs_polarformer_float16",
  "architecture": "bs_polarformer",
  "source_url": "https://huggingface.co/.../model_bs_polarformer_float16.ckpt",
  "sha256": "可选，缺省跳过校验",
  "config_url": "可选，yaml/json 参数文件（支持 URL）",
  "mvsep": { "sep_type": 123 }
}
```

`config_url` 允许把开源模型作者随权重发布的参数文件（yaml/json）直接以 URL 引入，作为架构参数的权威来源。

**不经过 models.json 直传**：`--model <URL|本地路径> --config-url <URL|本地路径>`（CLI）或 multipart `config_url` 字段（服务端）
可完全绕过清单——权重与参数文件均可直接指定，无需在清单中登记条目；按名引用时 `config_url` 覆盖清单条目的同名配置。
服务端 `POST /api/v1/separate` 的 `config_url` 字段同样生效。

## 网络与代理

所有联网操作（模型下载、清单拉取、MVSEP 调用）统一走 HTTP 代理。解析顺序：

`config.network.proxy`（显式配置）→ `ALL_PROXY` → `HTTPS_PROXY` → `HTTP_PROXY`

## 本地构建说明（Windows / ONNX Runtime）

`ort` crate 需要本机 ONNX Runtime 动态库（crates.io 的 `ort` 仅发布 rc 版，默认静态库与 MSVC STL 冲突）。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch-ort.ps1
```

脚本下载官方 `onnxruntime-win-x64-1.28.0` 到 `%LOCALAPPDATA%\asep-ort`，并按 `scripts/` 注释把 `onnxruntime.dll` 分发到 `target/<profile>`（运行时 exe 目录优先加载，请勿使用旧版本 dll）。`scripts/fetch-ort.ps1` 会生成 `.cargo/config.toml`（本机配置，不入库）。

## 测试

```sh
cargo test --workspace
```

覆盖：模型清单解析、架构参数、编码器回读（WAV16/32、FLAC16/24、MP3）、服务端健康检查 / 鉴权 / 模型列表。

## Docker

```sh
docker build -t audio-separator-server .
docker run -p 8080:8080 -e ASEP_MVSEP_API_KEY=... audio-separator-server
```

镜像内同时支持 local（含 ONNX Runtime Linux 库）与 mvsep 后端。注意 Roformer 权重首次使用需联网下载。
