use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{ConnectInfo, DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use simple_ai_common::{AudioEmbeddingOptions, AudioEmbeddingResponse};

use super::auth_helpers::{authenticate_request, extract_client_ip};
use crate::gateway::{can_request_model, classify_model, ModelClass, ModelRequest, SchedulerError};
use crate::models::request::{Request, Response};
use crate::{AppState, RequestEvent};

const DEFAULT_FILE_NAME: &str = "upload.bin";

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/audio/embeddings", post(create_audio_embedding))
        .layer(DefaultBodyLimit::max(200 * 1024 * 1024))
        .with_state(state)
}

fn can_request_audio_embedding_model(
    roles: &[String],
    model_request: &ModelRequest,
    model: &str,
    models_config: &crate::config::ModelsConfig,
) -> bool {
    can_request_model(roles, model_request)
        || classify_model(model, models_config) == Some(ModelClass::AudioEmbeddings)
}

async fn create_audio_embedding(
    State(state): State<Arc<AppState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<AudioEmbeddingResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let (auth_user, user) = authenticate_request(&state, &headers).await?;

    if !state.config.gateway.enabled {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Audio embeddings require gateway mode and an audio-embedding-capable runner"
                .to_string(),
        ));
    }

    let mut file_name = DEFAULT_FILE_NAME.to_string();
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut options: Option<AudioEmbeddingOptions> = None;

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
            options = Some(serde_json::from_str(&text).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("invalid audio embedding options: {}", e),
                )
            })?);
        }
    }

    let file_bytes = file_bytes.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "missing multipart file".to_string(),
        )
    })?;
    let options = options.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "missing multipart options".to_string(),
        )
    })?;

    let model = options.model.clone();
    let model_request = ModelRequest::parse(&model);
    if !matches!(model_request, ModelRequest::Specific(_)) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Audio embeddings require a specific model id".to_string(),
        ));
    }
    if !can_request_audio_embedding_model(
        &auth_user.roles,
        &model_request,
        &model,
        &state.config.models,
    ) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Permission denied: cannot request specific models. Configure the model under [models].audio_embeddings or use a token with model:specific.".to_string(),
        ));
    }
    let options_json = serde_json::to_string(&options)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid options: {}", e)))?;

    let mut req_log = Request::new(user.id.clone(), "/v1/audio/embeddings".to_string());
    req_log.model = Some(model.clone());
    req_log.client_ip = extract_client_ip(&headers, connect_info.map(|c| c.0));
    req_log.request_body = options_json.clone();
    let request_id = state
        .audit_logger
        .log_request(&req_log)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut runner_id: Option<String> = None;
    let mut wol_sent = false;
    let result = match state
        .request_scheduler
        .audio_embedding(
            &req_log.id,
            &model,
            &model_request,
            file_name,
            file_bytes,
            options_json,
        )
        .await
    {
        Ok(scheduled) => {
            runner_id = Some(scheduled.runner_id.clone());
            wol_sent = scheduled.wol_sent;
            state
                .wake_service
                .keepalive_runner(scheduled.runner_id.clone());
            Ok(scheduled.response)
        }
        Err(SchedulerError::Wake(e)) => Err(format!("Failed to wake inference runners: {}", e)),
        Err(SchedulerError::Router(e)) => Err(e.to_string()),
    };

    match result {
        Ok(response) => {
            let mut resp_log = Response::new(request_id, 200);
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.runner_id = runner_id.clone();
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = Some("audio_embeddings".to_string());
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
                runner_id,
                wol_sent,
            });
            Ok(Json(response))
        }
        Err(e) => {
            let mut resp_log = Response::new(request_id, 500);
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.response_body = e.clone();
            resp_log.model_class = Some("audio_embeddings".to_string());
            let _ = state.audit_logger.log_response(&resp_log);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelsConfig;

    #[test]
    fn test_api_key_style_roles_can_request_configured_audio_embedding_model() {
        let config = ModelsConfig {
            audio_embeddings: vec!["musicfm-msd".to_string(), "ast-audioset".to_string()],
            ..Default::default()
        };

        assert!(can_request_audio_embedding_model(
            &[],
            &ModelRequest::Specific("musicfm-msd".to_string()),
            "musicfm-msd",
            &config
        ));
        assert!(can_request_audio_embedding_model(
            &[],
            &ModelRequest::Specific("AST-AUDIOSET".to_string()),
            "AST-AUDIOSET",
            &config
        ));
    }

    #[test]
    fn test_api_key_style_roles_cannot_request_unconfigured_specific_model() {
        let config = ModelsConfig::default();

        assert!(!can_request_audio_embedding_model(
            &[],
            &ModelRequest::Specific("llama3:70b".to_string()),
            "llama3:70b",
            &config
        ));
    }
}
