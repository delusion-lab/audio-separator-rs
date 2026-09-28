# audio-separator-rs

**English** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

A Rust vocal / accompaniment separation tool. The core logic is a reusable crate with three usage modes:

- **CLI** (`asep`): one-shot runs — local inference, MVSEP cloud API, or your own asep-server;
- **HTTP server** (`audio-separator-server`): a long-running process exposing upload / task / download / cancel REST endpoints;
- **crate** (`audio-separator-core`): embed the library in other Rust programs.

## Architecture

```
crates/
├── audio-separator-core    core: model management, inference engines, audio IO, MVSEP client
├── audio-separator-cli     CLI entry (binary asep)
└── audio-separator-server  axum HTTP server (task queue + dual backends)
```

### Local inference engine (backend `local`)

| Architecture | Engine | Status |
| --- | --- | --- |
| mdx (UVR-MDX-NET) | ONNX Runtime | ✅ |
| bs_roformer | candle (pure Rust, f64 STFT + band-split + dual-axis attention) | ✅ |
| mel_band_roformer | candle (Slaney mel filterbank + MSST structure) | ✅ |
| bs_polarformer | candle (PoPE polar-coordinate positional embedding) | ✅ |

Roformer weights come as `ckpt` (pickle) or `safetensors`: on first use a `ckpt` is converted and cached as `safetensors` automatically (lazy conversion). CPU inference, performance on par with the reference implementation (see `PLAN.md` §10).

### MVSEP cloud backend (backend `mvsep`)

Separation via the [MVSEP API](https://mvsep.com/zh/full_api). Non-Premium accounts get a single concurrent task and a daily quota; the code enforces rate limiting with a semaphore.

## Quick start

### Build

Requires the Rust toolchain; local ORT (ONNX Runtime 1.28) setup is described in "Local build notes (Windows / ONNX Runtime)" below.

```sh
cargo build --release --workspace
```

Artifacts: `target/release/asep.exe`, `target/release/audio-separator-server.exe`.

### CLI separation (local)

```sh
# On first use, the model weights are downloaded automatically from the model list (see models.json)
asep separate input.wav -o out/ --model model_bs_polarformer_float16

# Pick the output format (wav/wav32/flac/flac24/mp3/m4a). M4A on the local backend
# is encoded via an external ffmpeg process (AAC 320 kbps)
asep separate input.wav -o out/ --model UVR_MDXNET_9482 --format flac

# M4A with an explicit ffmpeg path (lookup: --ffmpeg > config ffmpeg.path > PATH)
asep separate input.wav -o out/ --model UVR_MDXNET_9482 --format m4a --ffmpeg "C:\path\to\ffmpeg.exe

# Pass model weights + parameter config directly (yaml/json), bypassing models.json entirely:
#   --model       accepts a download URL / local model path (+ --arch to pin the architecture)
#   --config-url  accepts a parameter file URL / local path; the authoritative source of architecture params
asep separate input.wav -o out/ --model ./model.ckpt --arch bs_polarformer --config-url ./config.yaml
asep separate input.wav -o out/ --model https://.../model.ckpt --arch mel_band_roformer --config-url https://.../config.yaml
```

### M4A output (local backend)

M4A is encoded with an external `ffmpeg` process (AAC-LC 320 kbps). ffmpeg is looked up in this order:
`--ffmpeg <path>` (CLI) → config `ffmpeg.path` (TOML `[ffmpeg] path = "..."`) → `PATH`.
If ffmpeg is not found, a clear error tells you to set a path or install ffmpeg. M4A *input* files are decoded natively by symphonia and need no ffmpeg.

### CLI separation (MVSEP cloud)

```sh
asep separate input.wav -o out/ --backend mvsep \
  --model model_bs_polarformer_float16 --api-key YOUR_MVSEP_KEY --format flac

# Or provide the key via environment variable
ASEP_MVSEP_API_KEY=YOUR_KEY asep separate input.wav -o out/ --backend mvsep --model ...
```

### CLI separation (your own server)

Run your asep-server first, then drive it with the CLI — the CLI uploads the file, polls the task, and downloads the stems:

```sh
# Terminal 1: start the server (default 127.0.0.1:8080)
audio-separator-server

# Terminal 2: separate through the server
asep separate input.wav -o out/ --backend server --model model_bs_polarformer_float16

# Remote server / Bearer auth
asep separate input.wav -o out/ --backend server \
  --server-url http://10.0.0.5:8080 --auth-token secret --model UVR_MDXNET_9482

# Query the model list served by the server / inspect a server task
asep models --backend server --server-url http://127.0.0.1:8080
asep job-status <task_id> --server-url http://127.0.0.1:8080
```

Notes: a local input file is uploaded to the server; a URL input is passed through as `audio_url` (URL input is only supported by the MVSEP backend on the server side). The server must already be running — the CLI does not start it. The client connects to `--server-url` directly and does not use the download proxy.

### Model list & ranking

List all models in the manifest, optionally ranked by MUSDB18-HQ SDR or community recommendation:

```sh
# List all models (sorted by name)
asep models

# Rank by vocal separation quality (MUSDB18-HQ median SDR, descending)
asep models --sort sdr

# Top 10 models by SDR
asep models --sort sdr --top 10

# Rank by community recommendation (deton24 guide, ascending; shows category + rank)
asep models --sort community

# Inspect a single model (includes SDR scores and ranking info if available)
asep model-info model_bs_roformer_ep_368_sdr_12.9628

# Ranking data lives in a separate file (rankings.json); inspect it directly:
asep rankings                            # community guide section, grouped by category + rank
asep rankings --section mvsep            # MVSEP multisong leaderboard snapshot, by instrum rank
asep rankings --section mvsep --view vocals --top 10
asep rankings --section community --sort sdr
```

Scores and rankings come from three sources: the python-audio-separator benchmark (MUSDB18-HQ median SDR, kept in `models.json`), the deton24 UVR-MDX-Demucs-GSEP community guide (category rank + fullness/bleedless/SDR metrics), and the MVSEP multisong leaderboard snapshot (per-stem SDR). The latter two live in the separate `rankings.json` (see below). Models without SDR data are listed last when sorting by SDR; models without community ranking are listed last when sorting by community. Architectures other than mdx / bs_roformer / mel_band_roformer / bs_polarformer are catalogued for the MVSEP cloud backend but cannot run locally.

### Server

```sh
# Start (default 127.0.0.1:8080)
audio-separator-server --api-key YOUR_MVSEP_KEY --workers 1

# Enable auth (Bearer Token, off by default)
audio-separator-server --auth-token secret
```

REST endpoints:

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/v1/separate` | multipart task submission (`audio` file or `audio_url` + `model` + `backend` + `format` + optional `config_url`) |
| GET | `/api/v1/tasks/{id}` | task status (queued/running/done/failed/cancelled + progress) |
| GET | `/api/v1/tasks/{id}/download?stem=` | download a stem |
| DELETE | `/api/v1/tasks/{id}` | cancel a running task or clean up a finished one |
| GET | `/api/v1/models?backend=local\|mvsep` | model list / MVSEP algorithm catalog |
| GET | `/api/v1/rankings` | ranking data (community guide / MVSEP leaderboard snapshot) |
| GET | `/api/v1/health` | health check |

Upload example:

```sh
curl -F "audio=@input.wav" -F "model=model_bs_polarformer_float16" \
  -F "backend=local" -F "format=flac" http://127.0.0.1:8080/api/v1/separate
```

## Model list (models.json)

Model metadata (download URL, sha256, architecture params, MVSEP mapping, etc.) is maintained in a JSON file:

- by default the repo-local `models.json` is read;
- `--models-url <url>` / config `models.list = { url = "..." }` pulls a remote list (local path or URL both work);
- the list lives independently in the GitHub repo [delusion-lab/asep-models](https://github.com/delusion-lab/asep-models) (only the manifest JSON is hosted; weights stay at their official sources).

Entry example:

```json
{
  "name": "model_bs_polarformer_float16",
  "architecture": "bs_polarformer",
  "source_url": "https://huggingface.co/.../model_bs_polarformer_float16.ckpt",
  "sha256": "optional, skipped when absent",
  "config_url": "optional, yaml/json param file (URL supported)",
  "mvsep": { "sep_type": 123 }
}
```

`config_url` lets you point directly at the parameter file (yaml/json) that open-source model authors ship with their weights, as the authoritative source of architecture params.

**Bypass models.json entirely**: `--model <URL|local path> --config-url <URL|local path>` (CLI), or the multipart `config_url` field (server) — both weights and the param file can be specified directly without registering a manifest entry. When referencing a name, `config_url` overrides the manifest entry's config. The server `POST /api/v1/separate` `config_url` field works the same way.

## Rankings (rankings.json)

Ranking / recommendation data is maintained separately from the model list, since its sources (community guide, MVSEP leaderboard) are updated independently of the catalog:

- `community` section: deton24 UVR-MDX-Demucs-GSEP community guide entries (`name` matches the model list, plus rank / category / metrics / source / url);
- `mvsep` section: MVSEP multisong leaderboard snapshot (platform algorithm names, per-stem SDR, rank in each sort view, quality-checker entry URL).

Loading follows the same mechanism as the model list: by default the repo-local `rankings.json` is read (the same directory fallback as `models.json`); `--rankings-url <url>` / config `models.rankings = { url = "..." }` pull a remote file; `--rankings-file <path>` / `{ path = "..." }` point at a local one. The ranking file is optional — everything still works without it (community display and sorting simply fall back to empty).

## Network & proxy

All network operations (model download, list fetch, MVSEP calls) go through an HTTP proxy uniformly. Resolution order:

`config.network.proxy` (explicit) → `ALL_PROXY` → `HTTPS_PROXY` → `HTTP_PROXY`

## Local build notes (Windows / ONNX Runtime)

The `ort` crate needs the native ONNX Runtime DLL (crates.io `ort` only publishes RC builds; the default static lib conflicts with the MSVC STL).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch-ort.ps1
```

The script downloads the official `onnxruntime-win-x64-1.28.0` into `%LOCALAPPDATA%\asep-ort` and distributes `onnxruntime.dll` into `target/<profile>` per the comments in `scripts/` (the exe directory is loaded first at runtime; do not use older DLLs). `scripts/fetch-ort.ps1` also generates `.cargo/config.toml` (machine-local, not committed).

## Tests

```sh
cargo test --workspace
```

Covers: model list parsing, architecture params, encoder round-trips (WAV16/32, FLAC16/24, MP3), server health check / auth / model list.

## Docker

```sh
docker build -t audio-separator-server .
docker run -p 8080:8080 -e ASEP_MVSEP_API_KEY=... audio-separator-server
```

The image supports both the `local` (including ONNX Runtime Linux libraries) and `mvsep` backends. Note that Roformer weights need network access on first use.
