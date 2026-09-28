# audio-separator-rs

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | **한국어**

Rust로 구현된 보컬 / 반주 분리 도구. 핵심 로직은 재사용 가능한 crate로 분리되어 있으며 세 가지 사용 형태를 제공합니다:

- **CLI**(`asep`): 1회 실행 — 로컬 추론, MVSEP 클라우드 API, 또는 자신의 asep-server 연결;
- **HTTP 서버**(`audio-separator-server`): 상주 프로세스로 업로드 / 작업 / 다운로드 / 취소 REST API 제공;
- **crate**(`audio-separator-core`): 라이브러리로 다른 Rust 프로그램에 내장.

## 아키텍처

```
crates/
├── audio-separator-core    코어: 모델 관리, 추론 엔진, 오디오 IO, MVSEP 클라이언트
├── audio-separator-cli     CLI 엔트리(바이너리 asep)
└── audio-separator-server  axum HTTP 서버(작업 큐 + 이중 백엔드)
```

### 로컬 추론 엔진(백엔드 `local`)

| 아키텍처 | 엔진 | 상태 |
| --- | --- | --- |
| mdx(UVR-MDX-NET) | ONNX Runtime | ✅ |
| bs_roformer | candle(순수 Rust, f64 STFT + band-split + 이중축 어텐션) | ✅ |
| mel_band_roformer | candle(Slaney mel 필터뱅크 + MSST 구조) | ✅ |
| bs_polarformer | candle(PoPE 극좌표 위치 임베딩) | ✅ |

Roformer 가중치는 `ckpt`(pickle) 또는 `safetensors`로 제공됩니다. `ckpt`는 첫 사용 시 자동으로 `safetensors`로 변환·캐시됩니다(지연 변환). CPU 추론이며 성능은 참조 구현과 동일 수준입니다(자세한 내용은 `PLAN.md` §10).

### MVSEP 클라우드 백엔드(백엔드 `mvsep`)

[MVSEP API](https://mvsep.com/zh/full_api)를 통해 분리를 실행합니다. 비프리미엄 계정은 동시 작업 1개·일일 할당량 제한이 있으며, 코드 내부에서 세마포어로 레이트 제한을 적용합니다.

## 빠른 시작

### 빌드

Rust 툴체인이 필요합니다. 로컬 ORT(ONNX Runtime 1.28) 설정은 아래 "로컬 빌드 안내(Windows / ONNX Runtime)"를 참조하세요.

```sh
cargo build --release --workspace
```

산출물: `target/release/asep.exe`, `target/release/audio-separator-server.exe`.

### CLI 분리(로컬)

```sh
# 모델을 처음 사용할 때 모델 목록(models.json 참조)에서 가중치를 자동 다운로드
asep separate input.wav -o out/ --model model_bs_polarformer_float16

# 출력 형식 지정(wav/wav32/flac/flac24/mp3/m4a). 로컬 백엔드의 M4A는
# 외부 ffmpeg 프로세스로 인코딩(AAC 320kbps)
asep separate input.wav -o out/ --model UVR_MDXNET_9482 --format flac

# ffmpeg 경로 명시(탐색 순서: --ffmpeg > config ffmpeg.path > PATH)
asep separate input.wav -o out/ --model UVR_MDXNET_9482 --format m4a --ffmpeg "C:\path\to\ffmpeg.exe

# 모델 가중치 + 파라미터 설정(yaml/json)을 직접 지정하여 models.json을 완전히 우회:
#   --model       다운로드 URL / 로컬 모델 경로(+ --arch로 아키텍처 지정)
#   --config-url  파라미터 파일 URL / 로컬 경로. 아키텍처 파라미터의 권위 소스
asep separate input.wav -o out/ --model ./model.ckpt --arch bs_polarformer --config-url ./config.yaml
asep separate input.wav -o out/ --model https://.../model.ckpt --arch mel_band_roformer --config-url https://.../config.yaml
```

### M4A 출력(로컬 백엔드)

M4A는 외부 `ffmpeg` 프로세스로 인코딩됩니다(AAC-LC 320kbps). ffmpeg 탐색 순서: `--ffmpeg <path>`(CLI) → 설정 `ffmpeg.path`(TOML `[ffmpeg] path = "..."`) → `PATH`. ffmpeg를 찾지 못하면 경로 설정 또는 설치를 안내하는 명확한 오류를 반환합니다. M4A *입력* 파일은 symphonia로 네이티브 디코딩되며 ffmpeg가 필요 없습니다.

### CLI 분리(MVSEP 클라우드)

```sh
asep separate input.wav -o out/ --backend mvsep \
  --model model_bs_polarformer_float16 --api-key YOUR_MVSEP_KEY --format flac

# 환경 변수로 Key 제공
ASEP_MVSEP_API_KEY=YOUR_KEY asep separate input.wav -o out/ --backend mvsep --model ...
```

### CLI 분리(자신의 서버)

먼저 자신의 asep-server를 실행한 뒤 CLI로 구동합니다 — CLI가 파일 업로드, 작업 폴링, 스템 다운로드를 담당합니다:

```sh
# 터미널 1: 서버 시작(기본 127.0.0.1:8080)
audio-separator-server

# 터미널 2: 서버를 통한 분리
asep separate input.wav -o out/ --backend server --model model_bs_polarformer_float16

# 원격 서버 / Bearer 인증
asep separate input.wav -o out/ --backend server \
  --server-url http://10.0.0.5:8080 --auth-token secret --model UVR_MDXNET_9482

# 서버가 제공하는 모델 목록 조회 / 서버 작업 확인
asep models --backend server --server-url http://127.0.0.1:8080
asep job-status <task_id> --server-url http://127.0.0.1:8080
```

참고: 로컬 입력 파일은 서버로 업로드됩니다. URL 입력은 `audio_url` 필드로 전달됩니다(서버 측에서 URL 입력은 MVSEP 백엔드만 지원). 서버는 미리 실행되어 있어야 하며 CLI가 대신 시작하지 않습니다. 클라이언트는 `--server-url`에 직접 연결하며 다운로드 프록시를 사용하지 않습니다.

### 모델 목록과 랭킹

매니페스트의 모든 모델을 나열하며, 필요에 따라 MUSDB18-HQ SDR로 순위를 매깁니다:

```sh
# 모든 모델 나열(이름순)
asep models

# 보컬 분리 품질로 랭킹(MUSDB18-HQ 중앙값 SDR, 내림차순)
asep models --sort sdr

# SDR 상위 10개 모델
asep models --sort sdr --top 10

# 단일 모델 상세 조회(점수가 있으면 함께 표시)
asep model-info model_bs_roformer_ep_368_sdr_12.9628
```

점수는 python-audio-separator 벤치마크(MUSDB18-HQ 중앙값 SDR)에서 가져옵니다. 커뮤니티 랭킹(Google Doc)은 추후 제공 예정입니다. SDR 정렬 시 점수가 없는 모델은 마지막에 배치됩니다. mdx / bs_roformer / mel_band_roformer / bs_polarformer 이외의 아키텍처는 MVSEP 클라우드 백엔드용으로 등록되어 있으며 로컬에서 실행할 수 없습니다.

### 서버

```sh
# 시작(기본 127.0.0.1:8080)
audio-separator-server --api-key YOUR_MVSEP_KEY --workers 1

# 인증 활성화(Bearer Token, 기본 꺼짐)
audio-separator-server --auth-token secret
```

REST 엔드포인트:

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| POST | `/api/v1/separate` | multipart 작업 제출(`audio` 파일 또는 `audio_url` + `model` + `backend` + `format` + 선택 `config_url`) |
| GET | `/api/v1/tasks/{id}` | 작업 상태(queued/running/done/failed/cancelled + 진행률) |
| GET | `/api/v1/tasks/{id}/download?stem=` | 스템 결과 다운로드 |
| DELETE | `/api/v1/tasks/{id}` | 실행 중 작업 취소 / 종료 작업 정리 |
| GET | `/api/v1/models?backend=local\|mvsep` | 모델 목록 / MVSEP 알고리즘 카탈로그 |
| GET | `/api/v1/health` | 헬스 체크 |

업로드 예시:

```sh
curl -F "audio=@input.wav" -F "model=model_bs_polarformer_float16" \
  -F "backend=local" -F "format=flac" http://127.0.0.1:8080/api/v1/separate
```

## 모델 목록(models.json)

모델 메타정보(다운로드 URL, sha256, 아키텍처 파라미터, MVSEP 매핑 등)는 JSON 파일로 관리됩니다:

- 기본적으로 리포지토리 내 `models.json`을 읽음;
- `--models-url <url>` / 설정 `models.list = { url = "..." }`로 원격 목록 가져오기(로컬 경로 / URL 모두 가능);
- 목록은 GitHub 리포지토리 [delusion-lab/asep-models](https://github.com/delusion-lab/asep-models)에서 독립 관리(매니페스트 JSON만 호스팅, 가중치는 공식 소스에 유지).

엔트리 예시:

```json
{
  "name": "model_bs_polarformer_float16",
  "architecture": "bs_polarformer",
  "source_url": "https://huggingface.co/.../model_bs_polarformer_float16.ckpt",
  "sha256": "선택 사항, 없으면 검증 건너뜀",
  "config_url": "선택 사항, yaml/json 파라미터 파일(URL 지원)",
  "mvsep": { "sep_type": 123 }
}
```

`config_url`을 사용하면 오픈소스 모델 작성자가 가중치와 함께 배포한 파라미터 파일(yaml/json)을 URL로 직접 참조하여 아키텍처 파라미터의 권위 소스로 삼을 수 있습니다.

**models.json 완전 우회**: `--model <URL|로컬 경로> --config-url <URL|로컬 경로>`(CLI) 또는 multipart `config_url` 필드(서버)
로 가중치와 파라미터 파일을 매니페스트 등록 없이 직접 지정할 수 있습니다. 이름으로 참조할 때 `config_url`은 매니페스트 엔트리의 동일 설정을 덮어씁니다.
서버 `POST /api/v1/separate`의 `config_url` 필드도 동일하게 적용됩니다.

## 네트워크와 프록시

모든 네트워크 작업(모델 다운로드, 목록 가져오기, MVSEP 호출)은 HTTP 프록시를 통해 통일됩니다. 해석 순서:

`config.network.proxy`(명시적 설정) → `ALL_PROXY` → `HTTPS_PROXY` → `HTTP_PROXY`

## 로컬 빌드 안내(Windows / ONNX Runtime)

`ort` crate는 네이티브 ONNX Runtime DLL이 필요합니다(crates.io의 `ort`는 RC 버전만 배포하며 기본 정적 라이브러리는 MSVC STL과 충돌합니다).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch-ort.ps1
```

스크립트는 공식 `onnxruntime-win-x64-1.28.0`을 `%LOCALAPPDATA%\asep-ort`에 다운로드하고, `scripts/`의 주석에 따라 `onnxruntime.dll`을 `target/<profile>`에 배포합니다(실행 시 exe 디렉터리가 우선 로드되며, 이전 버전 DLL을 사용하지 마세요). `scripts/fetch-ort.ps1`은 `.cargo/config.toml`도 생성합니다(머신 로컬, 커밋 대상 아님).

## 테스트

```sh
cargo test --workspace
```

대상: 모델 목록 파싱, 아키텍처 파라미터, 인코더 라운드트립(WAV16/32, FLAC16/24, MP3), 서버 헬스 체크 / 인증 / 모델 목록.

## Docker

```sh
docker build -t audio-separator-server .
docker run -p 8080:8080 -e ASEP_MVSEP_API_KEY=... audio-separator-server
```

이미지는 `local`(ONNX Runtime Linux 라이브러리 포함)과 `mvsep` 백엔드를 모두 지원합니다. Roformer 가중치는 첫 사용 시 네트워크 접근이 필요합니다.
