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
use crate::models::audit::AuditLogEntry;

/// POST /chat/completions - OpenAI-compatible chat endpoint
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let start = Instant::now();

    // Authenticate user
    let user = state.jwks_client.authenticate(&headers).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, e.to_string()))?;

    // Prepare audit log
    let mut audit = AuditLogEntry::new(user.sub.clone(), "/chat/completions".to_string());
    audit.user_email = user.email.clone();
    audit.request_body = serde_json::to_string(&request).unwrap_or_default();

    // Get model (from request or default)
    let model = request.model.clone()
        .unwrap_or_else(|| state.config.ollama_model.clone());
    audit.model_used = Some(model.clone());

    // Forward to Ollama
    let result = state.ollama_client.chat(&request, &model).await;

    let response = match result {
        Ok(resp) => {
            audit.response_status = 200;
            audit.response_body = serde_json::to_string(&resp).unwrap_or_default();
            if let Some(ref usage) = resp.usage {
                audit.tokens_prompt = Some(usage.prompt_tokens);
                audit.tokens_completion = Some(usage.completion_tokens);
            }
            resp
        }
        Err(e) => {
            audit.response_status = 500;
            audit.response_body = e.to_string();
            audit.latency_ms = start.elapsed().as_millis() as u64;
            let _ = state.audit_logger.log(&audit);
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };

    audit.latency_ms = start.elapsed().as_millis() as u64;
    let _ = state.audit_logger.log(&audit);

    Ok(Json(response))
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/chat/completions", post(chat_completions))
        .with_state(state)
}
