//! OpenAI-compatible HTTP API.

pub mod audio_embeddings;
pub mod chat;
pub mod classifications;
pub mod decisions;
pub mod embeddings;
pub mod extractions;
pub mod health;
pub mod models;
pub mod ocr;
pub mod speech;
pub mod translations;

use std::sync::Arc;

use simple_server::web::Router;

use crate::state::AppState;

/// Build the API router.
pub fn router(state: Arc<AppState>) -> Router<Arc<AppState>> {
    Router::new()
        .merge(audio_embeddings::router())
        .merge(chat::router())
        .merge(classifications::router())
        .merge(translations::router())
        .merge(extractions::router())
        .merge(decisions::router())
        .merge(embeddings::router())
        .merge(models::router())
        .merge(ocr::router())
        .merge(speech::router())
        .layer(simple_server::web::middleware::from_fn_with_state(
            state,
            crate::drain::admission,
        ))
}
