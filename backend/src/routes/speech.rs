use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response as AxumResponse;
use axum::routing::post;
use axum::{Json, Router};
use futures_util::stream;
use simple_ai_common::{SpeechRequest, SpeechStreamFormat};

use super::auth_helpers::{authenticate_request, extract_client_ip};
use crate::gateway::{can_request_model, classify_model, ModelClass, ModelRequest, SchedulerError};
use crate::models::request::{Request, Response};
use crate::{AppState, RequestEvent};

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/audio/speech", post(create_speech))
        .with_state(state)
}

fn can_request_tts_model(
    roles: &[String],
    model_request: &ModelRequest,
    model: &str,
    models_config: &crate::config::ModelsConfig,
) -> bool {
    can_request_model(roles, model_request)
        || classify_model(model, models_config) == Some(ModelClass::Tts)
}

fn response_body(response: reqwest::Response) -> Body {
    let stream = stream::try_unfold(response, |mut response| async move {
        match response
            .chunk()
            .await
            .map_err(|e| std::io::Error::other(e.to_string()))?
        {
            Some(chunk) => Ok::<_, std::io::Error>(Some((chunk, response))),
            None => Ok(None),
        }
    });
    Body::from_stream(stream)
}

async fn create_speech(
    State(state): State<Arc<AppState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<SpeechRequest>,
) -> Result<AxumResponse, (StatusCode, String)> {
    let start = Instant::now();
    let (auth_user, user) = authenticate_request(&state, &headers).await?;

    if !state.config.gateway.enabled {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "TTS requires gateway mode and a TTS-capable runner".to_string(),
        ));
    }

    let model = request.model.clone();
    let model_request = ModelRequest::parse(&model);
    if !matches!(model_request, ModelRequest::Specific(_)) {
        return Err((
            StatusCode::BAD_REQUEST,
            "TTS requires a specific model id".to_string(),
        ));
    }
    if !can_request_tts_model(
        &auth_user.roles,
        &model_request,
        &model,
        &state.config.models,
    ) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Permission denied: cannot request specific models. Configure the model under [models].tts or use a token with model:specific.".to_string(),
        ));
    }

    let request_json = serde_json::to_string(&request).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid speech request: {}", e),
        )
    })?;

    let mut req_log = Request::new(user.id.clone(), "/v1/audio/speech".to_string());
    req_log.model = Some(model.clone());
    req_log.client_ip = extract_client_ip(&headers, connect_info.map(|c| c.0));
    req_log.request_body = request_json;
    let request_id = state
        .audit_logger
        .log_request(&req_log)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut runner_id: Option<String> = None;
    let mut wol_sent = false;
    let result = match state
        .request_scheduler
        .speech(&req_log.id, &model, &model_request, &request)
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
        Ok(provider_response) => {
            let content_type = provider_response
                .headers()
                .get(header::CONTENT_TYPE)
                .cloned()
                .unwrap_or_else(|| {
                    if request.stream_format_or_default() == SpeechStreamFormat::Sse {
                        HeaderValue::from_static("text/event-stream")
                    } else {
                        HeaderValue::from_static(
                            request.response_format_or_default().content_type(),
                        )
                    }
                });
            let mut resp_log = Response::new(request_id, 200);
            resp_log.response_body = "[audio]".to_string();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.runner_id = runner_id.clone();
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = Some("tts".to_string());
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

            let mut response = AxumResponse::new(response_body(provider_response));
            *response.status_mut() = StatusCode::OK;
            response
                .headers_mut()
                .insert(header::CONTENT_TYPE, content_type);
            if request.stream_format_or_default() == SpeechStreamFormat::Sse {
                response
                    .headers_mut()
                    .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
                response
                    .headers_mut()
                    .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
            }
            Ok(response)
        }
        Err(e) => {
            let mut resp_log = Response::new(request_id, 500);
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.response_body = e.clone();
            resp_log.model_class = Some("tts".to_string());
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
    fn test_api_key_style_roles_can_request_configured_tts_model() {
        let config = ModelsConfig {
            tts: vec!["tts-local".to_string()],
            ..Default::default()
        };

        assert!(can_request_tts_model(
            &[],
            &ModelRequest::Specific("tts-local".to_string()),
            "tts-local",
            &config
        ));
        assert!(can_request_tts_model(
            &[],
            &ModelRequest::Specific("TTS-LOCAL".to_string()),
            "TTS-LOCAL",
            &config
        ));
    }

    #[test]
    fn test_api_key_style_roles_cannot_request_unconfigured_specific_tts_model() {
        let config = ModelsConfig::default();

        assert!(!can_request_tts_model(
            &[],
            &ModelRequest::Specific("tts-local".to_string()),
            "tts-local",
            &config
        ));
    }
}
