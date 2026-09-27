//! REST API 处理器：分离提交、任务查询、下载、取消、模型列表、健康检查。

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{Multipart, Path as AxumPath, Query, State};
use axum::routing::{delete, get, post};
use axum::Router;
use axum::http::{header, HeaderValue, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::io::ReaderStream;

use audio_separator_core::config::{BackendKind, Config};
use audio_separator_core::model::OutputFormat;

use crate::store::{TaskRecord, TaskStatus, TaskStore, now};

/// 应用共享状态。
pub struct AppState {
    /// 核心配置（mvsep / network / models / server）。
    pub cfg: Config,
    /// 任务存储。
    pub store: Arc<dyn TaskStore>,
    /// 本地清单（启动时加载；`/models?backend=local` 直接返回）。
    pub local_manifest: audio_separator_core::model::ModelList,
    /// MVSEP 客户端（算法目录查询用；无需 API Key）。
    pub mvsep_client: Option<audio_separator_core::backend::mvsep::MvsepClient>,
    /// 上传根目录。
    pub upload_dir: std::path::PathBuf,
    /// 进程启动时间。
    pub started: Instant,
}

/// 统一错误响应。
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
    fn not_found(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, msg)
    }
    fn bad_request(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, msg)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, format!("IO 错误: {e}"))
    }
}

impl From<String> for ApiError {
    fn from(m: String) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, m)
    }
}

type ApiResult = std::result::Result<Response, ApiError>;

/// POST /api/v1/separate —— multipart 提交分离任务（音频文件或 URL）。
///
/// 字段：`audio`(文件) 或 `audio_url`(文本)，`model`(必填)，`backend`(local|mvsep)，
/// `format`(wav/wav16/wav32/flac/flac16/flac24/mp3/m4a)，`stems`(逗号分隔，可选)。
pub async fn separate(
    State(st): State<Arc<AppState>>,
    mut mp: Multipart,
) -> ApiResult {
    let mut audio_file: Option<(String, Vec<u8>)> = None;
    let mut audio_url: Option<String> = None;
    let mut model: Option<String> = None;
    let mut backend = st.cfg.backend;
    let mut format = st.cfg.output.format;
    let mut stems: Vec<String> = Vec::new();
    let mut config_url: Option<String> = None;

    let max_bytes = st.cfg.server.max_upload_bytes;
    while let Some(mut field) = mp.next_field().await.map_err(|e| ApiError::bad_request(e.to_string()))? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "audio" => {
                let fname = field
                    .file_name()
                    .map(|s| sanitize_file_name(s))
                    .unwrap_or_else(|| "audio.bin".to_string());
                let mut bytes: Vec<u8> = Vec::new();
                while let Some(chunk) = field.chunk().await.map_err(|e| ApiError::bad_request(e.to_string()))? {
                    if bytes.len() + chunk.len() > max_bytes as usize {
                        return Err(ApiError::new(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            format!("上传超过大小上限（{} 字节）", max_bytes),
                        ));
                    }
                    bytes.extend_from_slice(&chunk);
                }
                audio_file = Some((fname, bytes));
            }
            "audio_url" => {
                let v = field.text().await.map_err(|e| ApiError::bad_request(e.to_string()))?;
                audio_url = Some(v);
            }
            "model" => {
                model = Some(field.text().await.map_err(|e| ApiError::bad_request(e.to_string()))?);
            }
            "backend" => {
                let v = field.text().await.map_err(|e| ApiError::bad_request(e.to_string()))?;
                backend = match v.as_str() {
                    "local" => BackendKind::Local,
                    "mvsep" => BackendKind::Mvsep,
                    other => {
                        return Err(ApiError::bad_request(format!("backend 取值无效: {other}（local|mvsep）")))
                    }
                };
            }
            "format" => {
                let v = field.text().await.map_err(|e| ApiError::bad_request(e.to_string()))?;
                format = parse_format(&v).map_err(ApiError::bad_request)?;
            }
            "stems" => {
                let v = field.text().await.map_err(|e| ApiError::bad_request(e.to_string()))?;
                stems = v
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "config_url" => {
                let v = field.text().await.map_err(|e| ApiError::bad_request(e.to_string()))?;
                config_url = if v.trim().is_empty() { None } else { Some(v.trim().to_string()) };
            }
            _ => {
                // 忽略未知字段
            }
        }
    }

    let model = model.ok_or_else(|| ApiError::bad_request("缺少必填字段 model"))?;

    // 输入形态：上传文件优先；否则 URL。
    let id = uuid::Uuid::new_v4().simple().to_string();
    let task_dir = st.upload_dir.join(&id);
    std::fs::create_dir_all(&task_dir).map_err(ApiError::from)?;

    let (input_name, input_kind) = if let Some((fname, bytes)) = audio_file {
        let in_path = task_dir.join(&fname);
        std::fs::write(&in_path, &bytes).map_err(ApiError::from)?;
        (fname, audio_separator_core::backend::Input::Path(in_path))
    } else if let Some(url) = audio_url {
        let name = url
            .split('/')
            .last()
            .filter(|s| !s.is_empty())
            .map(|s| sanitize_file_name(s))
            .unwrap_or_else(|| "audio.bin".to_string());
        (name, audio_separator_core::backend::Input::Url(url))
    } else {
        return Err(ApiError::bad_request("缺少输入：需要 audio 文件或 audio_url 字段"));
    };

    // MVSEP 输入 URL 直接透传平台；本地后端不支持 URL（解码需本地文件）。
    if matches!(input_kind, audio_separator_core::backend::Input::Url(_))
        && backend == BackendKind::Local
    {
        return Err(ApiError::bad_request("本地后端暂不支持 URL 输入（请上传文件）"));
    }

    let input_display = input_name.clone();
    let record = TaskRecord {
        id: id.clone(),
        backend,
        model: model.clone(),
        status: TaskStatus::Queued,
        progress: 0.0,
        message: "排队中".to_string(),
        created_at: now(),
        updated_at: now(),
        input_name,
        out_dir: task_dir.join("out"),
        format,
        select_stems: stems,
        config_url: config_url.clone(),
        files: Vec::new(),
        error: None,
        cancel: Some(tokio_util::sync::CancellationToken::new()),
    };
    st.store.insert(record).map_err(ApiError::from)?;

    Ok((
        StatusCode::ACCEPTED,
        Json(json!({
            "task_id": id,
            "status": "queued",
            "backend": backend_str(backend),
            "model": model,
            "input": input_display,
            "config_url": config_url,
        })),
    )
        .into_response())
}

/// GET /api/v1/tasks/{id} —— 任务状态。
pub async fn task_status(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult {
    let task = st.store.get(&id).ok_or_else(|| ApiError::not_found(format!("任务不存在: {id}")))?;
    Ok(Json(task_json(&task)).into_response())
}

/// GET /api/v1/tasks/{id}/download?stem=vocals —— 下载输出分轨。
pub async fn download(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<DownloadQuery>,
) -> ApiResult {
    let task = st.store.get(&id).ok_or_else(|| ApiError::not_found(format!("任务不存在: {id}")))?;
    if task.status != TaskStatus::Done {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            format!("任务未完成（当前状态: {:?}）", task.status),
        ));
    }
    let stem = q.stem.unwrap_or_else(|| "vocals".to_string());
    let file = task
        .files
        .iter()
        .find(|f| f.name == stem)
        .ok_or_else(|| ApiError::not_found(format!("任务无分轨「{stem}」（可用: {:?}）", task.files.iter().map(|f| &f.name).collect::<Vec<_>>())))?;

    let f = tokio::fs::File::open(&file.path)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("读取输出失败: {e}")))?;
    let stream = ReaderStream::new(f);
    let body = Body::from_stream(stream);
    let ext = file.path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_else(|| "bin".to_string());
    let filename = format!("{stem}.{ext}");

    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    resp.headers_mut().insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&file.size.to_string()).unwrap(),
    );
    resp.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{filename}\"")).unwrap(),
    );
    Ok(resp)
}

/// DELETE /api/v1/tasks/{id} —— 取消进行中任务或删除已完成产物。
pub async fn cancel(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult {
    let task = st.store.get(&id).ok_or_else(|| ApiError::not_found(format!("任务不存在: {id}")))?;
    match task.status {
        TaskStatus::Queued | TaskStatus::Running => {
            if let Some(c) = &task.cancel {
                c.cancel();
            }
            st.store
                .update(&id, Box::new(|t| {
                    if !t.status.is_terminal() {
                        t.status = TaskStatus::Cancelled;
                        t.message = "已取消".to_string();
                    }
                }))
                .map_err(ApiError::from)?;
            Ok(StatusCode::ACCEPTED.into_response())
        }
        TaskStatus::Done | TaskStatus::Failed | TaskStatus::Cancelled => {
            // 清理磁盘产物后移除记录
            let dir = st.upload_dir.join(&id);
            let _ = tokio::fs::remove_dir_all(&dir).await;
            st.store.remove(&id).map_err(ApiError::from)?;
            Ok(StatusCode::NO_CONTENT.into_response())
        }
    }
}

/// GET /api/v1/models?backend=local|mvsep —— 模型/算法列表。
pub async fn models(
    State(st): State<Arc<AppState>>,
    Query(q): Query<ModelsQuery>,
) -> ApiResult {
    match q.backend.as_deref().unwrap_or("local") {
        "local" => Ok(Json(json!(st.local_manifest)).into_response()),
        "mvsep" => {
            let client = st
                .mvsep_client
                .as_ref()
                .ok_or_else(|| ApiError::bad_request("未配置 MVSEP（缺 api_key 或网络不可用）"))?;
            let algos = client
                .algorithms()
                .await
                .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, format!("拉取算法目录失败: {e}")))?;
            let list: Vec<Value> = algos
                .iter()
                .map(|a| {
                    json!({
                        "id": a.render_id,
                        "name": a.name,
                        "group": a.group.as_ref().and_then(|g| g.name.clone()),
                    })
                })
                .collect();
            Ok(Json(json!({ "algorithms": list, "count": algos.len() })).into_response())
        }
        other => Err(ApiError::bad_request(format!("backend 取值无效: {other}"))),
    }
}

/// GET /api/v1/health —— 健康检查。
pub async fn health(State(st): State<Arc<AppState>>) -> Response {
    Json(json!({
        "status": "ok",
        "uptime_secs": st.started.elapsed().as_secs(),
        "queued": st.store.list().iter().filter(|t| t.status == TaskStatus::Queued).count(),
        "running": st.store.list().iter().filter(|t| t.status == TaskStatus::Running).count(),
    }))
    .into_response()
}

/// Bearer Token 中间件（D4：默认关闭；`server.auth_token` 非空时启用）。
pub async fn auth(
    State(st): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let Some(expected) = &st.cfg.server.auth_token else {
        return next.run(req).await;
    };
    let ok = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|t| t == expected)
        .unwrap_or(false);
    if ok {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "未授权：需要 Bearer Token" })),
        )
            .into_response()
    }
}

#[derive(Deserialize)]
pub struct DownloadQuery {
    stem: Option<String>,
}

#[derive(Deserialize)]
pub struct ModelsQuery {
    backend: Option<String>,
}

/// 任务记录 → API JSON（隐藏内部路径细节，暴露输出文件名）。
fn task_json(t: &TaskRecord) -> Value {
    json!({
        "id": t.id,
        "backend": backend_str(t.backend),
        "model": t.model,
        "status": t.status,
        "progress": t.progress,
        "message": t.message,
        "created_at": t.created_at,
        "updated_at": t.updated_at,
        "input": t.input_name,
        "format": t.format,
        "select_stems": t.select_stems,
        "files": t.files.iter().map(|f| json!({
            "name": f.name,
            "size": f.size,
        })).collect::<Vec<_>>(),
        "error": t.error,
    })
}

fn backend_str(b: BackendKind) -> &'static str {
    match b {
        BackendKind::Local => "local",
        BackendKind::Mvsep => "mvsep",
    }
}

fn parse_format(s: &str) -> std::result::Result<OutputFormat, String> {
    match s.to_ascii_lowercase().as_str() {
        "wav" | "wav16" => Ok(OutputFormat::Wav16),
        "wav32" => Ok(OutputFormat::Wav32),
        "flac" | "flac16" => Ok(OutputFormat::Flac16),
        "flac24" => Ok(OutputFormat::Flac24),
        "mp3" => Ok(OutputFormat::Mp3),
        "m4a" => Ok(OutputFormat::M4a),
        other => Err(format!("格式无效: {other}（wav/wav32/flac/flac24/mp3/m4a）")),
    }
}

/// 文件名净化：仅保留最后路径段，剔除路径穿越字符。
fn sanitize_file_name(name: &str) -> String {
    let base = name.split(['/', '\\']).last().unwrap_or(name);
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "audio.bin".to_string()
    } else {
        cleaned
    }
}


/// 组装完整路由（含鉴权中间件）。
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/separate", post(separate))
        .route("/api/v1/tasks/{id}", get(task_status))
        .route("/api/v1/tasks/{id}/download", get(download))
        .route("/api/v1/tasks/{id}", delete(cancel))
        .route("/api/v1/models", get(models))
        .route("/api/v1/health", get(health))
        .layer(axum::middleware::from_fn_with_state(Arc::clone(&state), auth))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InMemoryTaskStore;
    use axum::http::Request;
    use axum::http::StatusCode;
    use tower::ServiceExt;

    fn test_state(auth: Option<String>) -> Arc<AppState> {
        let mut cfg = audio_separator_core::config::Config::default();
        cfg.server.auth_token = auth;
        Arc::new(AppState {
            cfg,
            store: Arc::new(InMemoryTaskStore::new()),
            local_manifest: audio_separator_core::model::ModelList {
                version: 0,
                models: Vec::new(),
            },
            mvsep_client: None,
            upload_dir: std::env::temp_dir(),
            started: Instant::now(),
        })
    }

    #[tokio::test]
    async fn health_ok() {
        let app = build_router(test_state(None));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn models_local_ok_without_auth() {
        let app = build_router(test_state(None));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/models?backend=local")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_rejects_and_accepts() {
        let app = build_router(test_state(Some("secret".to_string())));
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn separate_accepts_config_url_field() {
        let state = test_state(None);
        let app = build_router(state.clone());
        let cfg_val = "C:/tmp/model.yaml";
        let boundary = "----aseptest";
        let body = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\nContent-Type: application/octet-stream\r\n\r\nRIFF-test\r\n--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm1\r\n--{b}\r\nContent-Disposition: form-data; name=\"config_url\"\r\n\r\n{cfg}\r\n--{b}--\r\n",
            b = boundary,
            cfg = cfg_val
        );
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/separate")
                    .header("content-type", format!("multipart/form-data; boundary={boundary}"))
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::ACCEPTED);
        let body = axum::body::to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["config_url"], cfg_val);
        let id = json["task_id"].as_str().unwrap();
        let rec = state.store.get(id).unwrap();
        assert_eq!(rec.config_url.as_deref(), Some(cfg_val));
        assert_eq!(rec.model, "m1");
    }

    #[tokio::test]
    async fn auth_wrong_token_rejected() {
        let app = build_router(test_state(Some("secret".to_string())));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .header("authorization", "Bearer wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }
}
