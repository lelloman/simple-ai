use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{ConnectInfo, DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use simple_ai_common::{OcrOptions, OcrResponse};

use super::auth_helpers::{authenticate_request, extract_client_ip};
use crate::models::request::{Request, Response};
use crate::{AppState, RequestEvent};

const DEFAULT_FILE_NAME: &str = "upload.bin";

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/ocr", post(ocr))
        .layer(DefaultBodyLimit::max(25 * 1024 * 1024))
        .with_state(state)
}

async fn ocr(
    State(state): State<Arc<AppState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<OcrResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let (auth_user, user) = authenticate_request(&state, &headers).await?;

    if !state.config.gateway.enabled {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "OCR requires gateway mode and an OCR-capable runner".to_string(),
        ));
    }

    let mut file_name = DEFAULT_FILE_NAME.to_string();
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut options = OcrOptions::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            if let Some(upload_name) = field.file_name() {
                file_name = upload_name.to_string();
            }
            let bytes = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            file_bytes = Some(bytes.to_vec());
        } else if name == "options" {
            let text = field
                .text()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            options = serde_json::from_str(&text)
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid options: {}", e)))?;
        }
    }

    let file_bytes = file_bytes.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "missing multipart file".to_string(),
        )
    })?;
    let options_json = serde_json::to_string(&options)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid options: {}", e)))?;

    let mut req_log = Request::new(user.id.clone(), "/v1/ocr".to_string());
    req_log.model = Some("ocr".to_string());
    req_log.client_ip = extract_client_ip(&headers, connect_info.map(|c| c.0));
    req_log.request_body = options_json.clone();
    let request_id = state
        .audit_logger
        .log_request(&req_log)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let result = state
        .inference_router
        .ocr_multipart(file_name, file_bytes, options_json)
        .await;

    match result {
        Ok(routed) => {
            let mut resp_log = Response::new(request_id, 200);
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.runner_id = Some(routed.runner_id.clone());
            resp_log.model_class = Some("ocr".to_string());
            let _ = state.audit_logger.log_response(&resp_log);

            let _ = state.request_events.send(RequestEvent {
                id: req_log.id.clone(),
                timestamp: req_log.timestamp.to_rfc3339(),
                user_id: req_log.user_id.clone(),
                user_email: auth_user.email.clone(),
                request_path: req_log.request_path.clone(),
                model: req_log.model.clone(),
                client_ip: req_log.client_ip.clone(),
                status: Some(200),
                latency_ms: Some(resp_log.latency_ms as i64),
                tokens_prompt: None,
                tokens_completion: None,
                runner_id: resp_log.runner_id.clone(),
                wol_sent: false,
            });

            Ok(Json(routed.response))
        }
        Err(e) => {
            let mut resp_log = Response::new(request_id, 500);
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.response_body = e.to_string();
            resp_log.model_class = Some("ocr".to_string());
            let _ = state.audit_logger.log_response(&resp_log);

            let _ = state.request_events.send(RequestEvent {
                id: req_log.id.clone(),
                timestamp: req_log.timestamp.to_rfc3339(),
                user_id: req_log.user_id,
                user_email: auth_user.email,
                request_path: req_log.request_path,
                model: req_log.model,
                client_ip: req_log.client_ip,
                status: Some(500),
                latency_ms: Some(resp_log.latency_ms as i64),
                tokens_prompt: None,
                tokens_completion: None,
                runner_id: None,
                wol_sent: false,
            });

            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}
