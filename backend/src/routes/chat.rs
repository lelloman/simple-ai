use std::sync::Arc;
use std::time::Instant;
use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Json, Router,
};
use axum::http::HeaderMap;

use crate::AppState;
use crate::models::chat::{ChatCompletionRequest, ChatCompletionResponse};
use crate::models::request::{Request, Response};

/// POST /chat/completions - OpenAI-compatible chat endpoint
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let start = Instant::now();

    // Authenticate user
    let auth_user = state.jwks_client.authenticate(&headers).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, e.to_string()))?;

    // Find or create user in database
    let user = state.audit_logger.find_or_create_user(
        &auth_user.sub,
        auth_user.email.as_deref(),
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Check if user is enabled
    if !user.is_enabled {
        return Err((StatusCode::FORBIDDEN, "User is disabled".to_string()));
    }

    // Get model (from request or default)
    let model = request.model.clone()
        .unwrap_or_else(|| state.config.ollama_model.clone());

    // Log request BEFORE calling Ollama
    let mut req_log = Request::new(user.id.clone(), "/chat/completions".to_string());
    req_log.request_body = serde_json::to_string(&request).unwrap_or_default();
    req_log.model = Some(model.clone());

    let request_id = state.audit_logger.log_request(&req_log)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Forward to Ollama
    let result = state.ollama_client.chat(&request, &model).await;

    // Log response
    let (response, resp_log) = match result {
        Ok(resp) => {
            let mut resp_log = Response::new(request_id, 200);
            resp_log.response_body = serde_json::to_string(&resp).unwrap_or_default();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            if let Some(ref usage) = resp.usage {
                resp_log.tokens_prompt = Some(usage.prompt_tokens);
                resp_log.tokens_completion = Some(usage.completion_tokens);
            }
            (resp, resp_log)
        }
        Err(e) => {
            let mut resp_log = Response::new(request_id, 500);
            resp_log.response_body = e.to_string();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            let _ = state.audit_logger.log_response(&resp_log);
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };

    let _ = state.audit_logger.log_response(&resp_log);

    Ok(Json(response))
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/chat/completions", post(chat_completions))
        .with_state(state)
}
