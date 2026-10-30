//! Chat completions endpoint (OpenAI-compatible).

use std::sync::Arc;

use futures_util::{stream, StreamExt};
use simple_ai_common::{ChatCompletionRequest, ChatCompletionResponse};
use simple_server::web::body::Body;
use simple_server::web::extract::State;
use simple_server::web::http::{header, HeaderValue, StatusCode};
use simple_server::web::response::{IntoResponse, Response};
use simple_server::web::routing::post;
use simple_server::web::{Json, Router};

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

    if request.response_format.is_some() {
        let (engine, _) = state
            .engine_registry
            .resolve_engine_for_model(resolved_model)
            .await
            .ok_or_else(|| Error::ModelNotFound(resolved_model.into()))?;
        validate_response_format_backend(engine.engine_type(), &request)?;
    }
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
        simple_server::response_headers::replace(
            response.headers_mut(),
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        );
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

fn validate_response_format_backend(engine: &str, request: &ChatCompletionRequest) -> Result<()> {
    if request.response_format.is_some() && !matches!(engine, "vllm" | "tag") {
        return Err(Error::InvalidRequest(
            "response_format requires a vLLM-backed chat model".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_output_requires_vllm() {
        let request = serde_json::from_value(serde_json::json!({"messages":[], "response_format":{"type":"json_schema","json_schema":{"name":"tags","schema":{"type":"object"}}}})).unwrap();
        for engine in ["ollama", "llama_cpp", "halogen", "summaries"] {
            assert!(matches!(
                validate_response_format_backend(engine, &request),
                Err(Error::InvalidRequest(_))
            ));
        }
        for engine in ["vllm", "tag"] {
            validate_response_format_backend(engine, &request).unwrap();
        }
    }
}
