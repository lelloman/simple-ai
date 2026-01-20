//! Models endpoint (OpenAI-compatible).

use std::sync::Arc;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use crate::engine::ModelInfo;
use crate::error::Result;
use crate::state::AppState;

/// Build the models router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/models", get(list_models))
}

/// OpenAI-compatible model list response.
#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelData>,
}

#[derive(Debug, Serialize)]
struct ModelData {
    id: String,
    object: &'static str,
    created: i64,
    owned_by: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    permission: Option<Vec<()>>,
}

impl From<ModelInfo> for ModelData {
    fn from(info: ModelInfo) -> Self {
        Self {
            id: info.id,
            object: "model",
            created: 0, // Ollama doesn't provide creation time
            owned_by: "local",
            permission: None,
        }
    }
}

/// GET /v1/models - List available models.
async fn list_models(State(state): State<Arc<AppState>>) -> Result<Json<ModelsResponse>> {
    let mut all_models = Vec::new();

    // Collect models from all engines
    for engine in state.engine_registry.all().await {
        match engine.list_models().await {
            Ok(models) => all_models.extend(models),
            Err(e) => {
                tracing::warn!(
                    "Failed to list models from {} engine: {}",
                    engine.engine_type(),
                    e
                );
            }
        }
    }

    let response = ModelsResponse {
        object: "list",
        data: all_models.into_iter().map(ModelData::from).collect(),
    };

    Ok(Json(response))
}
