//! OpenAI-compatible HTTP API.

pub mod audio_embeddings;
pub mod chat;
pub mod embeddings;
pub mod health;
pub mod models;
pub mod ocr;
pub mod speech;

use std::sync::Arc;

use axum::Router;

use crate::state::AppState;

/// Build the API router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .merge(audio_embeddings::router())
        .merge(chat::router())
        .merge(embeddings::router())
        .merge(models::router())
        .merge(ocr::router())
        .merge(speech::router())
}
