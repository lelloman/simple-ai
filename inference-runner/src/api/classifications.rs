//! Generic zero-shot NLI classification endpoint.

use std::sync::Arc;

use axum::{extract::State, routing::post, Json, Router};
use simple_ai_common::{ClassificationRequest, ClassificationResponse};

use crate::error::{Error, Result};
use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/classifications", post(classify))
}

async fn classify(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<ClassificationRequest>,
) -> Result<Json<ClassificationResponse>> {
    if request.input.is_empty() || request.labels.is_empty() {
        return Err(Error::InvalidRequest(
            "input and labels must not be empty".to_string(),
        ));
    }
    let requested_model = request.model.clone();
    let resolved_model = state
        .config
        .aliases
        .mappings
        .get(&requested_model)
        .cloned()
        .unwrap_or_else(|| requested_model.clone());
    request.model = resolved_model.clone();
    let lease = state.engine_registry.acquire_model(&resolved_model).await?;
    let mut response = lease.engine.classify(&lease.engine_model, &request).await?;
    response.model = requested_model;
    Ok(Json(response))
}
