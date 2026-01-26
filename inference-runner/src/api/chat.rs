//! Chat completions endpoint (OpenAI-compatible).

use std::sync::Arc;

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use simple_ai_common::{ChatCompletionRequest, ChatCompletionResponse};

use crate::error::{Error, Result};
use crate::state::AppState;

/// Build the chat router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/chat/completions", post(chat_completions))
}

/// POST /v1/chat/completions - OpenAI-compatible chat completion.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>> {
    // Get the model from request, or use default
    let model = request
        .model
        .as_deref()
        .ok_or_else(|| Error::InvalidRequest("model is required".to_string()))?;

    // Apply alias mapping if configured
    let resolved_model = state
        .config
        .aliases
        .mappings
        .get(model)
        .map(|s| s.as_str())
        .unwrap_or(model);

    tracing::debug!("Chat completion request for model: {} (resolved: {})", model, resolved_model);

    // Find an engine that can serve this model
    let engine = state
        .engine_registry
        .find_engine_for_model(resolved_model)
        .await
        .ok_or_else(|| Error::ModelNotFound(model.to_string()))?;

    // Execute the chat completion
    let response = engine.chat_completion(resolved_model, &request).await?;

    Ok(Json(response))
}
