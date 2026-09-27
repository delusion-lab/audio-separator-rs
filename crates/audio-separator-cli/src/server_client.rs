//! CLI 侧 asep-server 客户端：提交分离任务、轮询状态、下载分轨、查询模型。
//!
//! 复用的 server REST API（crates/audio-separator-server/src/api.rs）：
//! `POST /api/v1/separate`（multipart）、`GET /api/v1/tasks/{id}`、
//! `GET /api/v1/tasks/{id}/download?stem=`、`DELETE /api/v1/tasks/{id}`、
//! `GET /api/v1/models?backend=`、`GET /api/v1/health`。
//! 可选 `Authorization: Bearer <token>` 鉴权。

use std::path::{Path, PathBuf};

use audio_separator_core::error::{Error, Result};
use serde::Deserialize;
use serde_json::Value;

/// 分离任务提交参数（与 server multipart 字段一一对应）。
pub struct SubmitArgs {
    /// 本地音频文件（优先）；与 `audio_url` 二选一。
    pub audio_path: Option<PathBuf>,
    /// 远程音频 URL（server 仅 MVSEP 后端支持透传）。
    pub audio_url: Option<String>,
    /// 模型：清单名字 / 下载 URL / 本地路径。
    pub model: String,
    /// server 执行后端：`local` | `mvsep`。
    pub backend: String,
    /// 输出格式：wav/wav32/flac/flac24/mp3/m4a。
    pub format: String,
    /// 只输出这些分轨；空 = 模型全部分轨。
    pub stems: Vec<String>,
    /// 模型参数配置（yaml/json）URL 或本地路径。
    pub config_url: Option<String>,
}

/// 任务输出文件条目。
#[derive(Debug, Clone, Deserialize)]
pub struct ServerTaskFile {
    pub name: String,
    pub size: u64,
}

/// 任务状态快照（server `task_json` 的子集）。
#[derive(Debug, Clone, Deserialize)]
pub struct ServerTaskStatus {
    pub id: String,
    /// queued / running / done / failed / cancelled。
    pub status: String,
    pub progress: f32,
    pub message: String,
    #[serde(default)]
    pub files: Vec<ServerTaskFile>,
    pub error: Option<String>,
}

/// asep-server HTTP 客户端。
pub struct ServerClient {
    base: String,
    token: Option<String>,
    http: reqwest::Client,
}

impl ServerClient {
    /// `base`：如 `http://127.0.0.1:8080`；`token`：server 启用鉴权时的 Bearer Token。
    /// 注意：直连 server（本机/内网），不经过下载代理。
    pub fn new(base: &str, token: Option<String>) -> Result<Self> {
        let base = base.trim_end_matches('/').to_string();
        if base.is_empty() || (!base.starts_with("http://") && !base.starts_with("https://")) {
            return Err(Error::Config(format!(
                "invalid --server-url: {base} (expected http(s)://host:port)"
            )));
        }
        let http = reqwest::Client::new();
        Ok(Self { base, token, http })
    }

    fn authed(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(t) => req.bearer_auth(t),
            None => req,
        }
    }

    /// POST /api/v1/separate —— 提交任务，返回 task_id。
    pub async fn submit(&self, args: &SubmitArgs) -> Result<String> {
        let url = format!("{}/api/v1/separate", self.base);
        let mut form = reqwest::multipart::Form::new()
            .text("model", args.model.clone())
            .text("backend", args.backend.clone())
            .text("format", args.format.clone());
        if let Some(p) = &args.audio_path {
            let file = tokio::fs::File::open(p)
                .await
                .map_err(|e| Error::Other(format!("failed to open {}: {e}", p.display())))?;
            let fname = p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "audio.bin".to_string());
            form = form.part("audio", reqwest::multipart::Part::stream(file).file_name(fname));
        }
        if let Some(u) = &args.audio_url {
            form = form.text("audio_url", u.clone());
        }
        if !args.stems.is_empty() {
            form = form.text("stems", args.stems.join(","));
        }
        if let Some(c) = &args.config_url {
            form = form.text("config_url", c.clone());
        }
        let resp = self
            .authed(self.http.post(&url).multipart(form))
            .send()
            .await
            .map_err(|e| Error::Other(format!("server request failed: {e}")))?;
        if resp.status() != reqwest::StatusCode::ACCEPTED {
            return Err(server_error(resp).await);
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::Other(format!("invalid server response: {e}")))?;
        v["task_id"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| Error::Other("server response missing task_id".to_string()))
    }

    /// GET /api/v1/tasks/{id} —— 查询任务状态。
    pub async fn status(&self, id: &str) -> Result<ServerTaskStatus> {
        let url = format!("{}/api/v1/tasks/{id}", self.base);
        let resp = self
            .authed(self.http.get(&url))
            .send()
            .await
            .map_err(|e| Error::Other(format!("server request failed: {e}")))?;
        if resp.status() != reqwest::StatusCode::OK {
            return Err(server_error(resp).await);
        }
        resp.json()
            .await
            .map_err(|e| Error::Other(format!("invalid server response: {e}")))
    }

    /// GET /api/v1/tasks/{id}/download?stem= —— 下载单个分轨到 `dest_dir`，
    /// 文件名取 server Content-Disposition（如 `vocals.wav`），返回写入路径。
    pub async fn download(&self, id: &str, stem: &str, dest_dir: &Path) -> Result<PathBuf> {
        let url = format!("{}/api/v1/tasks/{id}/download?stem={stem}", self.base);
        let resp = self
            .authed(self.http.get(&url))
            .send()
            .await
            .map_err(|e| Error::Other(format!("server request failed: {e}")))?;
        if resp.status() != reqwest::StatusCode::OK {
            return Err(server_error(resp).await);
        }
        let fname = resp
            .headers()
            .get(reqwest::header::CONTENT_DISPOSITION)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_attachment_filename)
            .unwrap_or_else(|| format!("{stem}.wav"));
        let dest = dest_dir.join(&fname);
        tokio::fs::create_dir_all(dest_dir)
            .await
            .map_err(|e| Error::Other(format!("failed to create {}: {e}", dest_dir.display())))?;
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| Error::Other(format!("failed to read download: {e}")))?;
        tokio::fs::write(&dest, &bytes)
            .await
            .map_err(|e| Error::Other(format!("failed to write {}: {e}", dest.display())))?;
        Ok(dest)
    }

    /// DELETE /api/v1/tasks/{id} —— 取消进行中任务 / 删除已完成产物。
    pub async fn cancel(&self, id: &str) -> Result<()> {
        let url = format!("{}/api/v1/tasks/{id}", self.base);
        let resp = self
            .authed(self.http.delete(&url))
            .send()
            .await
            .map_err(|e| Error::Other(format!("server request failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(server_error(resp).await);
        }
        Ok(())
    }

    /// GET /api/v1/models?backend= —— 查询 server 侧模型/算法列表。
    pub async fn models(&self, backend: &str) -> Result<Value> {
        let url = format!("{}/api/v1/models?backend={backend}", self.base);
        let resp = self
            .authed(self.http.get(&url))
            .send()
            .await
            .map_err(|e| Error::Other(format!("server request failed: {e}")))?;
        if resp.status() != reqwest::StatusCode::OK {
            return Err(server_error(resp).await);
        }
        resp.json()
            .await
            .map_err(|e| Error::Other(format!("invalid server response: {e}")))
    }

    /// GET /api/v1/health —— 健康检查。
    pub async fn health(&self) -> Result<Value> {
        let url = format!("{}/api/v1/health", self.base);
        let resp = self
            .authed(self.http.get(&url))
            .send()
            .await
            .map_err(|e| Error::Other(format!("server request failed: {e}")))?;
        if resp.status() != reqwest::StatusCode::OK {
            return Err(server_error(resp).await);
        }
        resp.json()
            .await
            .map_err(|e| Error::Other(format!("invalid server response: {e}")))
    }
}

/// 把非成功响应转为 `Error`：优先取 JSON `{"error": "..."}`。
async fn server_error(resp: reqwest::Response) -> Error {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    let msg = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v["error"].as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| body);
    Error::Backend(format!("server request failed (HTTP {status}): {msg}"))
}

/// 解析 `Content-Disposition: attachment; filename="vocals.wav"`。
fn parse_attachment_filename(v: &str) -> Option<String> {
    for part in v.split(';') {
        let part = part.trim();
        if let Some(f) = part.strip_prefix("filename=") {
            return Some(f.trim().trim_matches('"').to_string());
        }
    }
    None
}
