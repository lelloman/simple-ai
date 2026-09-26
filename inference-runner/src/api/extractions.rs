//! Schema-driven information extraction endpoint.

use std::sync::Arc;

use simple_server::web::{extract::State, routing::post, Json, Router};
use simple_ai_common::{ExtractionRequest, ExtractionResponse};

use crate::error::{Error, Result};
use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/extractions", post(extract))
}

async fn extract(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<ExtractionRequest>,
) -> Result<Json<ExtractionResponse>> {
    request.validate().map_err(Error::InvalidRequest)?;
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
    let mut response = lease.engine.extract(&lease.engine_model, &request).await?;
    response.model = requested_model;
    Ok(Json(response))
}
