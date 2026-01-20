//! OpenAI-compatible /v1/models endpoint.

use std::sync::Arc;

use axum::{
    extract::State,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::AppState;

/// Model entry in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Response from /v1/models endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

/// GET /v1/models - List available models
async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let models = if state.config.gateway.enabled {
        // Get models from connected runners
        state
            .inference_router
            .list_models()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|m| ModelObject {
                id: m.id,
                object: "model".to_string(),
                created: 0,
                owned_by: "local".to_string(),
            })
            .collect()
    } else {
        // Return the default configured model
        vec![ModelObject {
            id: state.config.ollama.model.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "ollama".to_string(),
        }]
    };

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/models", get(list_models))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_response_serialization() {
        let response = ModelsResponse {
            object: "list".to_string(),
            data: vec![ModelObject {
                id: "test-model".to_string(),
                object: "model".to_string(),
                created: 1234567890,
                owned_by: "local".to_string(),
            }],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("test-model"));
        assert!(json.contains(r#""object":"list""#));
    }
}
