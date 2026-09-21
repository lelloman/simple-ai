//! Health check endpoint.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use simple_server::axum::extract::State;
use simple_server::axum::http::StatusCode;
use simple_server::axum::Json;

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
    match run_health_check(state.engine_registry.clone(), state.ocr_provider.is_some()).await {
        Ok(response) => (StatusCode::OK, Json(response)),
        Err(failure) => (StatusCode::SERVICE_UNAVAILABLE, Json(failure.error)),
    }
}

async fn run_health_check(
    registry: Arc<crate::engine::EngineRegistry>,
    ocr_configured: bool,
) -> Result<HealthResponse, simple_server::health::CheckFailure<HealthResponse>> {
    simple_server::health::Check::new("engines", move || {
        let registry = registry.clone();
        async move { collect_health(&registry, ocr_configured).await }
    })
    .run()
    .await
}

async fn collect_health(
    registry: &crate::engine::EngineRegistry,
    ocr_configured: bool,
) -> Result<HealthResponse, HealthResponse> {
    let mut engine_statuses = Vec::new();
    let mut any_healthy = false;

    for engine in registry.all().await {
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
    let (is_ready, status_str) = if engine_statuses.is_empty() && ocr_configured {
        (true, "ok")
    } else if engine_statuses.is_empty() {
        (true, "starting")
    } else if any_healthy {
        (true, "ok")
    } else {
        (false, "unhealthy")
    };

    let response = HealthResponse {
        status: status_str.to_string(),
        engines: engine_statuses,
    };

    if is_ready {
        Ok(response)
    } else {
        Err(response)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;
    use crate::engine::{
        ChatCompletionStream, EngineHealth, EngineRegistry, InferenceEngine, ModelInfo,
    };
    use crate::error::{Error, Result as EngineResult};
    use async_trait::async_trait;
    use simple_ai_common::{ChatCompletionRequest, ChatCompletionResponse};

    struct FakeEngine {
        name: &'static str,
        result: std::result::Result<EngineHealth, &'static str>,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InferenceEngine for FakeEngine {
        fn engine_type(&self) -> &'static str {
            self.name
        }
        async fn health_check(&self) -> EngineResult<EngineHealth> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.result
                .clone()
                .map_err(|message| Error::Communication(message.into()))
        }
        async fn list_models(&self) -> EngineResult<Vec<ModelInfo>> {
            unimplemented!()
        }
        async fn get_model(&self, _: &str) -> EngineResult<Option<ModelInfo>> {
            unimplemented!()
        }
        async fn load_model(&self, _: &str) -> EngineResult<()> {
            unimplemented!()
        }
        async fn unload_model(&self, _: &str) -> EngineResult<()> {
            unimplemented!()
        }
        async fn chat_completion(
            &self,
            _: &str,
            _: &ChatCompletionRequest,
        ) -> EngineResult<ChatCompletionResponse> {
            unimplemented!()
        }
        async fn chat_completion_stream(
            &self,
            _: &str,
            _: &ChatCompletionRequest,
        ) -> EngineResult<ChatCompletionStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn aggregate_checks_every_engine_and_preserves_details() {
        let registry = Arc::new(EngineRegistry::new());
        let healthy_calls = Arc::new(AtomicUsize::new(0));
        let failed_calls = Arc::new(AtomicUsize::new(0));
        registry
            .register(Arc::new(FakeEngine {
                name: "healthy",
                result: Ok(EngineHealth {
                    is_healthy: true,
                    version: Some("1.2".into()),
                    models_loaded: vec!["model-a".into()],
                }),
                calls: healthy_calls.clone(),
            }))
            .await;
        registry
            .register(Arc::new(FakeEngine {
                name: "failed",
                result: Err("offline"),
                calls: failed_calls.clone(),
            }))
            .await;
        let response = run_health_check(registry, false).await.unwrap();
        assert_eq!(healthy_calls.load(Ordering::SeqCst), 1);
        assert_eq!(failed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(response.status, "ok");
        assert_eq!(response.engines.len(), 2);
        assert!(response
            .engines
            .iter()
            .any(|item| item.engine_type == "healthy"
                && item.healthy
                && item.version.as_deref() == Some("1.2")
                && item.loaded_models == ["model-a"]));
        assert!(response
            .engines
            .iter()
            .any(|item| item.engine_type == "failed"
                && !item.healthy
                && item.error.as_deref() == Some("Engine communication error: offline")));
    }

    #[tokio::test]
    async fn aggregate_preserves_unhealthy_and_empty_semantics() {
        let registry = Arc::new(EngineRegistry::new());
        let calls = Arc::new(AtomicUsize::new(0));
        registry
            .register(Arc::new(FakeEngine {
                name: "unhealthy",
                result: Ok(EngineHealth {
                    is_healthy: false,
                    version: Some("0.9".into()),
                    models_loaded: vec!["stuck".into()],
                }),
                calls: calls.clone(),
            }))
            .await;
        let failure = run_health_check(registry, false).await.unwrap_err();
        assert_eq!(failure.name.as_ref(), "engines");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(failure.error.status, "unhealthy");
        assert_eq!(failure.error.engines[0].version.as_deref(), Some("0.9"));
        assert_eq!(failure.error.engines[0].loaded_models, ["stuck"]);
        let empty = Arc::new(EngineRegistry::new());
        assert_eq!(
            run_health_check(empty.clone(), false).await.unwrap().status,
            "starting"
        );
        assert_eq!(run_health_check(empty, true).await.unwrap().status, "ok");
    }

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
