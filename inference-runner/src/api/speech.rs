//! Text-to-speech endpoint.

use std::collections::VecDeque;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use futures_util::stream;
use simple_ai_common::{SpeechRequest, SpeechStreamFormat};

use crate::error::{Error, Result};
use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/audio/speech", post(create_speech))
}

fn response_body(response: reqwest::Response) -> Body {
    let stream = stream::try_unfold(
        (response, VecDeque::<Bytes>::new()),
        |(mut response, mut pending)| async move {
            if let Some(bytes) = pending.pop_front() {
                return Ok::<_, std::io::Error>(Some((bytes, (response, pending))));
            }
            match response
                .chunk()
                .await
                .map_err(|e| std::io::Error::other(e.to_string()))?
            {
                Some(chunk) => Ok(Some((chunk, (response, pending)))),
                None => Ok(None),
            }
        },
    );
    Body::from_stream(stream)
}

async fn create_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SpeechRequest>,
) -> Result<Response> {
    let requested_model = request.model.clone();
    let resolved_model = state
        .config
        .aliases
        .mappings
        .get(requested_model.as_str())
        .cloned()
        .unwrap_or_else(|| requested_model.clone());
    let mut resolved_request = request.clone();
    resolved_request.model = resolved_model.clone();

    let engine = state
        .engine_registry
        .find_engine_for_model(&resolved_model)
        .await
        .ok_or_else(|| Error::ModelNotFound(requested_model.clone()))?;
    let provider_response = engine.speech(&resolved_model, &resolved_request).await?;

    let content_type = provider_response
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| {
            if resolved_request.stream_format_or_default() == SpeechStreamFormat::Sse {
                HeaderValue::from_static("text/event-stream")
            } else {
                HeaderValue::from_static(
                    resolved_request.response_format_or_default().content_type(),
                )
            }
        });

    let mut response = Response::new(response_body(provider_response));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, content_type);
    if resolved_request.stream_format_or_default() == SpeechStreamFormat::Sse {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        response
            .headers_mut()
            .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    }
    Ok(response)
}
