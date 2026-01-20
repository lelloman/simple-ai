//! Health check endpoint.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

/// Health response structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Overall health status.
    pub status: String,
    /// Individual engine health status.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub engines: Vec<EngineHealthStatus>,
}

/// Health status of a single engine.
///
/// Note: This uses `healthy` (not `is_healthy` like the WebSocket protocol's `EngineStatus`)
/// to keep the HTTP API response concise, following OpenAI-style conventions.
#[derive(Debug, Serialize, Deserialize)]
pub struct EngineHealthStatus {
    pub engine_type: String,
    pub healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub loaded_models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// GET /health - Health check endpoint.
///
/// Checks health of all registered engines and returns aggregate status.
/// Returns 200 OK if at least one engine is healthy, 503 if all are unhealthy.
pub async fn health(State(state): State<Arc<AppState>>) -> (StatusCode, Json<HealthResponse>) {
    let mut engine_statuses = Vec::new();
    let mut any_healthy = false;

    for engine in state.engine_registry.all().await {
        let status = match engine.health_check().await {
            Ok(health) => {
                if health.is_healthy {
                    any_healthy = true;
                }
                EngineHealthStatus {
                    engine_type: engine.engine_type().to_string(),
                    healthy: health.is_healthy,
                    version: health.version,
                    loaded_models: health.models_loaded,
                    error: None,
                }
            }
            Err(e) => EngineHealthStatus {
                engine_type: engine.engine_type().to_string(),
                healthy: false,
                version: None,
                loaded_models: vec![],
                error: Some(e.to_string()),
            },
        };
        engine_statuses.push(status);
    }

    // If no engines registered, report starting/ok
    let (status_code, status_str) = if engine_statuses.is_empty() {
        (StatusCode::OK, "starting")
    } else if any_healthy {
        (StatusCode::OK, "ok")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "unhealthy")
    };

    let response = HealthResponse {
        status: status_str.to_string(),
        engines: engine_statuses,
    };

    (status_code, Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            engines: vec![EngineHealthStatus {
                engine_type: "ollama".to_string(),
                healthy: true,
                version: Some("0.5.0".to_string()),
                loaded_models: vec!["llama3.2:3b".to_string()],
                error: None,
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains(r#""status":"ok""#));
        assert!(json.contains(r#""healthy":true"#));
        assert!(!json.contains(r#""error""#)); // None fields skipped
    }

    #[test]
    fn test_health_response_unhealthy() {
        let response = HealthResponse {
            status: "unhealthy".to_string(),
            engines: vec![EngineHealthStatus {
                engine_type: "ollama".to_string(),
                healthy: false,
                version: None,
                loaded_models: vec![],
                error: Some("Connection refused".to_string()),
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains(r#""error":"Connection refused""#));
    }
}
