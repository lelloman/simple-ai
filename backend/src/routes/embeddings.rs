use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{ConnectInfo, State},
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{AppState, RequestEvent};
use crate::gateway::{can_request_model, ModelRequest, RouterError};
use crate::models::request::{Request, Response};
use crate::wol::WakeError;
use super::auth_helpers::{authenticate_request, extract_client_ip};

/// OpenAI-compatible embedding request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingRequest {
    /// Text input(s) to embed. Can be a single string or array of strings.
    pub input: EmbeddingInput,
    /// Model to use for embedding.
    pub model: String,
}

/// Input can be a single string or an array of strings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// POST /v1/embeddings - OpenAI-compatible embeddings endpoint
async fn create_embeddings(
    State(state): State<Arc<AppState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    let start = Instant::now();

    let (auth_user, user) = authenticate_request(&state, &headers).await?;

    // Parse and validate model request
    let model_request = ModelRequest::parse(&request.model);

    // Check permissions
    if !can_request_model(&auth_user.roles, &model_request) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Permission denied: cannot request specific models. Use class:embed-small or class:embed-large.".to_string(),
        ));
    }

    // Get model string for routing
    let model = match &model_request {
        ModelRequest::Specific(m) => m.clone(),
        ModelRequest::Class(class) => format!("class:{}", class),
    };

    let inputs = request.input.clone().into_vec();

    // Log request
    let mut req_log = Request::new(user.id.clone(), "/v1/embeddings".to_string());
    req_log.model = Some(model.clone());
    req_log.client_ip = extract_client_ip(&headers, connect_info.map(|c| c.0));

    let request_id = state.audit_logger.log_request(&req_log)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Approximate token count for usage reporting
    let prompt_tokens: u32 = inputs.iter()
        .map(|s| (s.split_whitespace().count() as u32).max(1))
        .sum();

    // Track routing metadata
    let mut runner_id: Option<String> = None;
    let mut wol_sent = false;

    // Forward to appropriate backend (gateway or direct Ollama)
    let result = if state.config.gateway.enabled {
        let initial_result = state.inference_router.embed::<EmbeddingRequest, EmbeddingResponse>(&model, &request).await;

        match initial_result {
            Ok(routed) => {
                runner_id = Some(routed.runner_id);
                Ok(routed.response)
            }
            Err(RouterError::NoRunners) if state.wake_service.is_enabled() => {
                tracing::info!("No runners available, attempting wake-on-demand for {:?}", model_request);
                wol_sent = true;

                match state.wake_service.speculative_wake_and_wait(&model_request).await {
                    Ok(wake_result) => {
                        tracing::info!(
                            "Runner {} connected after {:.1}s, retrying request",
                            wake_result.runner_id,
                            wake_result.wait_duration.as_secs_f64()
                        );
                        let retry_result = state.inference_router.embed::<EmbeddingRequest, EmbeddingResponse>(&model, &request).await;
                        match retry_result {
                            Ok(routed) => {
                                runner_id = Some(routed.runner_id);
                                Ok(routed.response)
                            }
                            Err(e) => Err(e.to_string()),
                        }
                    }
                    Err(WakeError::NoWakeableRunners) => {
                        Err("No runners available and no wakeable runners configured".to_string())
                    }
                    Err(WakeError::Timeout(secs)) => {
                        Err(format!("No runners available after waiting {}s for wake", secs))
                    }
                    Err(e) => {
                        tracing::warn!("Wake-on-demand failed: {}", e);
                        Err(format!("Failed to wake inference runners: {}", e))
                    }
                }
            }
            Err(e) => Err(e.to_string()),
        }
    } else {
        // Direct Ollama mode - check circuit breaker
        if !state.circuit_breaker.is_available("ollama") {
            Err("Circuit breaker open: Ollama backend is unavailable".to_string())
        } else {
            let result = state.ollama_client.embed(&model, &inputs).await;
            match result {
                Ok(embeddings) => {
                    state.circuit_breaker.record_success("ollama");
                    let data: Vec<EmbeddingData> = embeddings
                        .into_iter()
                        .enumerate()
                        .map(|(i, embedding)| EmbeddingData {
                            object: "embedding".to_string(),
                            embedding,
                            index: i,
                        })
                        .collect();

                    Ok(EmbeddingResponse {
                        object: "list".to_string(),
                        data,
                        model: model.clone(),
                        usage: EmbeddingUsage {
                            prompt_tokens,
                            total_tokens: prompt_tokens,
                        },
                    })
                }
                Err(e) => {
                    state.circuit_breaker.record_failure("ollama");
                    Err(e.to_string())
                }
            }
        }
    };

    // Compute model class for metrics tracking
    let model_class_str = model_request
        .effective_class(&state.config.models)
        .map(|c| c.as_str().to_string());

    let (response, resp_log) = match result {
        Ok(resp) => {
            let mut resp_log = Response::new(request_id, 200);
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.tokens_prompt = Some(prompt_tokens);
            resp_log.runner_id = runner_id;
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = model_class_str.clone();
            (resp, resp_log)
        }
        Err(e) => {
            let mut resp_log = Response::new(request_id, 500);
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.response_body = e.clone();
            resp_log.runner_id = runner_id.clone();
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = model_class_str.clone();
            let _ = state.audit_logger.log_response(&resp_log);

            let _ = state.request_events.send(RequestEvent {
                id: req_log.id.clone(),
                timestamp: req_log.timestamp.to_rfc3339(),
                user_id: req_log.user_id.clone(),
                user_email: auth_user.email.clone(),
                request_path: req_log.request_path.clone(),
                model: req_log.model.clone(),
                client_ip: req_log.client_ip.clone(),
                status: Some(500),
                latency_ms: Some(resp_log.latency_ms as i64),
                tokens_prompt: None,
                tokens_completion: None,
                runner_id,
                wol_sent,
            });

            return Err((StatusCode::INTERNAL_SERVER_ERROR, e));
        }
    };

    let _ = state.audit_logger.log_response(&resp_log);

    let _ = state.request_events.send(RequestEvent {
        id: req_log.id.clone(),
        timestamp: req_log.timestamp.to_rfc3339(),
        user_id: req_log.user_id.clone(),
        user_email: auth_user.email.clone(),
        request_path: req_log.request_path.clone(),
        model: req_log.model.clone(),
        client_ip: req_log.client_ip.clone(),
        status: Some(200),
        latency_ms: Some(resp_log.latency_ms as i64),
        tokens_prompt: Some(prompt_tokens as i64),
        tokens_completion: None,
        runner_id: resp_log.runner_id.clone(),
        wol_sent: resp_log.wol_sent,
    });

    Ok(Json(response))
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/embeddings", post(create_embeddings))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_input_single() {
        let json = r#""hello world""#;
        let input: EmbeddingInput = serde_json::from_str(json).unwrap();
        let vec = input.into_vec();
        assert_eq!(vec, vec!["hello world"]);
    }

    #[test]
    fn test_embedding_input_multiple() {
        let json = r#"["hello", "world"]"#;
        let input: EmbeddingInput = serde_json::from_str(json).unwrap();
        let vec = input.into_vec();
        assert_eq!(vec, vec!["hello", "world"]);
    }

    #[test]
    fn test_embedding_request_deserialization() {
        let json = r#"{"input": "test text", "model": "nomic-embed-text"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "nomic-embed-text");
    }

    #[test]
    fn test_embedding_request_array_input() {
        let json = r#"{"input": ["text one", "text two"], "model": "nomic-embed-text"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "nomic-embed-text");
    }

    #[test]
    fn test_embedding_response_serialization() {
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![EmbeddingData {
                object: "embedding".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            model: "nomic-embed-text".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("nomic-embed-text"));
        assert!(json.contains(r#""object":"list""#));
        assert!(json.contains("0.1"));
    }

    #[test]
    fn test_embedding_usage_serialization() {
        let usage = EmbeddingUsage {
            prompt_tokens: 10,
            total_tokens: 10,
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains(r#""prompt_tokens":10"#));
        assert!(json.contains(r#""total_tokens":10"#));
    }
}
