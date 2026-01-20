//! OpenAI-compatible HTTP API.

pub mod chat;
pub mod health;
pub mod models;

use std::sync::Arc;

use axum::Router;

use crate::state::AppState;

/// Build the API router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .merge(chat::router())
        .merge(models::router())
}
