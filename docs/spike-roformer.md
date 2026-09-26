# Roformer 家族引擎路线 spike 结论（2026-09-26）

> 前置任务：M1 规划的"Roformer ONNX 导出可行性 spike"。结论：**放弃 ONNX 导出路径，维持
> 用户已确认的 candle 移植为主路线**，权重加载采用"ckpt → safetensors 预转换"或"运行时安全解析"，
> 两者均有成熟 Rust 先例可借鉴。下文为证据与执行依据。

## 1. 结论摘要

| 问题 | 结论 |
| --- | --- |
| ONNX 导出是否必要？ | **不必要**。Roformer 家族无成熟 ONNX 生态（已核实），导出需 torch 环境 + 复数/FFT 入图，收益低 |
| 纯 Rust 推理是否可行？ | **已有人做到**：`uvr_roformer`（crates.io 0.1.2）用 **Burn** 框架 CPU FP32 直接加载原始 `model_bs_roformer_ep_368_sdr_12.9628.ckpt` 并推理分离 |
| 引擎选型 | 维持 **candle 为主**（用户确认）；Burn 实现（uvr_roformer）作为前向结构与权重解析的参考 |
| ckpt 权重怎么进 Rust？ | 两条路：① 预转换 `ckpt → safetensors`（`pt-loader` / `anamnesis` 已有 Rust 实现）；② 运行时安全解析（参考 uvr_roformer weights 模块，注意避开 Python pickle 代码执行风险，参考 CVE-2025-49839） |
| M2 架构顺序 | `vr`（复用 onnx 引擎，快速增量）→ `bs_roformer`（1296 版）→ `mel_band_roformer` → `bs_polarformer` |

## 2. 证据

### 2.1 纯 Rust 先例

- **`uvr_roformer`**（docs.rs/uvr-roformer/0.1.2）："CPU inference and PCM separation for
  BS-RoFormer 1296… loading the original checkpoint in Rust"，`RoformerModel::load` 提供
  Burn CPU 路径（可选 openvino feature）。固定 1296 版，原始网络保留参考 ISTFT 长度，需
  外部调度器 pad/crop —— 与本项目架构注册表 + io 管线的分工一致。
- 同作者 crate 家族：`uvr_runtime`（文件级分离调度）、`uvr_models`（模型目录与指纹，4 个 checkpoint）。

### 2.2 ckpt 权重解析 / 转换（Rust）

- **`pt-loader`**（crates.io 0.1.4）："Safe parser-based PyTorch checkpoint converter… Parses
  torch zip .pt checkpoints with strict safety limits. Converts checkpoints to
  model.safetensors + model.yaml."
- **`anamnesis`**（crates.io 0.7.4）：框架无关的 tensor 文件解析，支持 PyTorch .pth / safetensors。
- 结论：ckpt → safetensors 的转换工具链在 Rust 侧已存在，可直接依赖或借鉴，无需 Python 参与。

### 2.3 参考实现与权重直链

| 架构 | 参考实现 | 权重（已核实直链） |
| --- | --- | --- |
| bs_roformer | lucidrains/BS-RoFormer（PyTorch）；uvr_roformer（Rust 结构） | `model_bs_roformer_ep_368_sdr_12.9628.ckpt`（TRvlvr/model_repo releases） |
| mel_band_roformer | openmirlab/melband-roformer-infer（内置 89 模型注册表 + 懒下载 + sha256） | 同上注册表 |
| bs_polarformer | ZFTurbo/MSST（配置+权重发布） | `https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.20/model_bs_polarformer_float16.ckpt`（+ 同名 .yaml 配置） |

### 2.4 风险提示

- Python `torch.load` 存在 pickle 反序列化代码执行面（CVE-2025-49839 即 UVR 系工具）；本项目
  纯 Rust 解析天然规避，但自研解析器须做长度/类型上限校验，勿盲目信任文件内容。
- `model_bs_polarformer_float16.ckpt` 为 fp16 权重，candle 加载需转 fp32 或确认算子支持。

## 3. M2 执行依据

1. `vr` 架构（onnx 引擎，dim_f=3072 系参数；模型如 `UVR-DeEcho-Dereverb-AGGRESSIVE.onnx` 等，需核实用例）。
2. `bs_roformer`：先做权重加载（ckpt→safetensors 或运行时解析），再按 lucidrains/uvr_roformer
   结构移植前向（band-split、RoPE、轴向注意力、复数 STFT），以 1296 版为第一个对齐目标。
3. `mel_band_roformer` / `bs_polarformer`：在 bs_roformer 基线上替换 band 与位置编码部分，
   参数 schema 对齐 MSST yaml。
4. 每架构验证：输出与参考实现（Python）在测试音频上对比，参数按架构独立校验（D2）。
