use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::DefaultBodyLimit,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::AppState;
use super::auth_helpers::authenticate_request;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractRequest {
    pub url: String,
    pub html: String,
    pub lang_hint: Option<String>,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractResponse {
    pub ok: bool,
    pub text: Option<String>,
    pub html: Option<String>,
    pub title: Option<String>,
    pub author: Option<String>,
    pub date: Option<String>,
    pub site_name: Option<String>,
    pub extractor: String,
    pub extractor_version: String,
    pub rejected: bool,
    pub flags: Vec<String>,
    pub score: Option<f32>,
    pub error: Option<String>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/extract", post(extract))
        .layer(DefaultBodyLimit::max(25 * 1024 * 1024))
        .with_state(state)
}

async fn extract(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ExtractRequest>,
) -> Result<Json<ExtractResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let (_auth_user, user) = authenticate_request(&state, &headers).await?;

    if request.url.trim().is_empty() || request.html.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "url and html are required".to_string()));
    }

    let payload = serde_json::to_vec(&request)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid payload: {}", e)))?;

    let mut child = Command::new("python3")
        .arg("/usr/local/bin/simple_ai_extract.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("failed to spawn extractor: {}", e)))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&payload)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("failed to send extractor payload: {}", e)))?;
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("extractor process failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let msg = if stderr.is_empty() {
            "extractor returned non-zero exit status".to_string()
        } else {
            stderr
        };
        return Err((StatusCode::INTERNAL_SERVER_ERROR, msg));
    }

    let response: ExtractResponse = serde_json::from_slice(&output.stdout)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("invalid extractor response: {}", e)))?;

    tracing::info!(
        user_id = %user.id,
        url = %request.url,
        rejected = response.rejected,
        latency_ms = start.elapsed().as_millis() as u64,
        "content extraction completed"
    );

    Ok(Json(response))
}
