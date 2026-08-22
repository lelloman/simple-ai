//! Chat completions endpoint (OpenAI-compatible).

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::{stream, StreamExt};
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
) -> Result<Response> {
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

    tracing::debug!(
        "Chat completion request for model: {} (resolved: {})",
        model,
        resolved_model
    );

    let lease = state.engine_registry.acquire_model(resolved_model).await?;

    if request.stream.unwrap_or(false) {
        let upstream = lease
            .engine
            .chat_completion_stream(&lease.engine_model, &request)
            .await?;
        let guarded = stream::unfold((upstream, lease), |(mut upstream, lease)| async move {
            upstream.next().await.map(|item| (item, (upstream, lease)))
        });
        let mut response = Response::new(Body::from_stream(guarded));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        response
            .headers_mut()
            .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
        Ok(response)
    } else {
        let response: ChatCompletionResponse = lease
            .engine
            .chat_completion(&lease.engine_model, &request)
            .await?;
        Ok(Json(response.strip_internal_metrics()).into_response())
    }
}
