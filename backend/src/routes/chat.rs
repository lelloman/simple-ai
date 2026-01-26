use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use axum::{
    extract::{ConnectInfo, State},
    http::StatusCode,
    routing::post,
    Json, Router,
};
use axum::http::HeaderMap;

use crate::{AppState, RequestEvent};
use crate::auth::AuthUser;
use crate::gateway::{can_request_model, ModelRequest, RouterError};
use crate::models::chat::{ChatCompletionRequest, ChatCompletionResponse};
use crate::models::request::{Request, Response};
use crate::models::user::User;
use crate::wol::WakeError;

/// Extract client IP from headers (X-Forwarded-For, X-Real-IP) or connection info.
fn extract_client_ip(headers: &HeaderMap, addr: Option<SocketAddr>) -> Option<String> {
    // Check X-Forwarded-For first (may contain multiple IPs, take the first)
    if let Some(forwarded) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        if let Some(first_ip) = forwarded.split(',').next() {
            return Some(first_ip.trim().to_string());
        }
    }
    // Check X-Real-IP
    if let Some(real_ip) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
        return Some(real_ip.to_string());
    }
    // Fall back to connection address
    addr.map(|a| a.ip().to_string())
}

/// Authenticate a request using API key or JWT.
/// Returns (AuthUser, User) on success.
async fn authenticate_request(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(AuthUser, User), (StatusCode, String)> {
    // Extract the Authorization header
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    // Check for API key (sk-...) first
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        if token.starts_with("sk-") {
            // Validate API key
            match state.audit_logger.validate_api_key(token) {
                Ok(Some((user_id, email))) => {
                    // Find or create the user
                    let user = state.audit_logger.find_or_create_user(&user_id, email.as_deref())
                        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

                    // Check if user is enabled
                    if !user.is_enabled {
                        return Err((StatusCode::FORBIDDEN, "User is disabled".to_string()));
                    }

                    // Create AuthUser for API key users (no roles, but can use the API)
                    let auth_user = AuthUser::new(user_id, email, vec![]);
                    return Ok((auth_user, user));
                }
                Ok(None) => {
                    return Err((StatusCode::UNAUTHORIZED, "Invalid API key".to_string()));
                }
                Err(e) => {
                    return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
                }
            }
        }
    }

    // Fall back to JWT authentication
    let auth_user = state.jwks_client.authenticate(headers).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, e.to_string()))?;

    // Find or create user in database
    let user = state.audit_logger.find_or_create_user(
        &auth_user.sub,
        auth_user.email.as_deref(),
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Check if user is enabled
    if !user.is_enabled {
        return Err((StatusCode::FORBIDDEN, "User is disabled".to_string()));
    }

    Ok((auth_user, user))
}

/// POST /v1/chat/completions - OpenAI-compatible chat endpoint
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let start = Instant::now();

    // Authenticate user - try API key first, then fall back to JWT
    let (auth_user, user) = authenticate_request(&state, &headers).await?;

    // Parse and validate model request
    let model_request = match &request.model {
        Some(model_str) => ModelRequest::parse(model_str),
        None => {
            // Default: users without model:specific get class:fast
            // users with model:specific get the configured default model
            if auth_user.has_role("model:specific") {
                ModelRequest::Specific(state.config.ollama.model.clone())
            } else {
                ModelRequest::Class(crate::gateway::ModelClass::Fast)
            }
        }
    };

    // Check permissions
    if !can_request_model(&auth_user.roles, &model_request) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Permission denied: cannot request specific models. Use class:fast or class:big.".to_string(),
        ));
    }

    // Get model string for routing
    let model = match &model_request {
        ModelRequest::Specific(m) => m.clone(),
        ModelRequest::Class(class) => {
            // For class requests, use a placeholder that the router will interpret
            format!("class:{}", class)
        }
    };

    // Log request BEFORE calling Ollama
    let mut req_log = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
    req_log.request_body = serde_json::to_string(&request).unwrap_or_default();
    req_log.model = Some(model.clone());
    req_log.client_ip = extract_client_ip(&headers, connect_info.map(|c| c.0));

    let request_id = state.audit_logger.log_request(&req_log)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Emit pending request event for admin dashboard
    let _ = state.request_events.send(RequestEvent {
        id: req_log.id.clone(),
        timestamp: req_log.timestamp.to_rfc3339(),
        user_id: req_log.user_id.clone(),
        user_email: auth_user.email.clone(),
        request_path: req_log.request_path.clone(),
        model: req_log.model.clone(),
        client_ip: req_log.client_ip.clone(),
        status: None, // Pending
        latency_ms: None,
        tokens_prompt: None,
        tokens_completion: None,
        runner_id: None,
        wol_sent: false,
    });

    // Track routing metadata
    let mut runner_id: Option<String> = None;
    let mut wol_sent = false;

    // Check if batching should be used (enabled, non-streaming, has batch queue)
    let use_batching = state.batch_queue.is_some()
        && !request.stream.unwrap_or(false);

    // Forward to appropriate backend (gateway or direct Ollama)
    let result = if state.config.gateway.enabled {
        // Try gateway routing (batched or standard)
        let initial_result = if use_batching {
            let batch_queue = state.batch_queue.as_ref().unwrap();
            state.inference_router.chat_completion_batched(&model, &request, batch_queue).await
        } else {
            state.inference_router.chat_completion(&model, &request).await
        };

        match initial_result {
            Ok(routed) => {
                runner_id = Some(routed.runner_id);
                Ok(routed.response)
            }
            Err(RouterError::NoRunners) if state.wake_service.is_enabled() => {
                // No runners available, try wake-on-demand (speculative if enabled)
                tracing::info!("No runners available, attempting wake-on-demand for {:?}", model_request);
                wol_sent = true;

                match state.wake_service.speculative_wake_and_wait(&model_request).await {
                    Ok(wake_result) => {
                        tracing::info!(
                            "Runner {} connected after {:.1}s, retrying request",
                            wake_result.runner_id,
                            wake_result.wait_duration.as_secs_f64()
                        );
                        // Retry the request now that a runner is available
                        let retry_result = if use_batching {
                            let batch_queue = state.batch_queue.as_ref().unwrap();
                            state.inference_router.chat_completion_batched(&model, &request, batch_queue).await
                        } else {
                            state.inference_router.chat_completion(&model, &request).await
                        };
                        match retry_result {
                            Ok(routed) => {
                                runner_id = Some(routed.runner_id);
                                Ok(routed.response)
                            }
                            Err(e) => Err(crate::llm::OllamaError::ConnectionFailed(e.to_string())),
                        }
                    }
                    Err(WakeError::NoWakeableRunners) => {
                        Err(crate::llm::OllamaError::ConnectionFailed(
                            "No runners available and no wakeable runners configured".to_string(),
                        ))
                    }
                    Err(WakeError::Timeout(secs)) => {
                        Err(crate::llm::OllamaError::ConnectionFailed(format!(
                            "No runners available after waiting {}s for wake",
                            secs
                        )))
                    }
                    Err(e) => {
                        tracing::warn!("Wake-on-demand failed: {}", e);
                        Err(crate::llm::OllamaError::ConnectionFailed(format!(
                            "Failed to wake inference runners: {}",
                            e
                        )))
                    }
                }
            }
            Err(e) => Err(crate::llm::OllamaError::ConnectionFailed(e.to_string())),
        }
    } else {
        state.ollama_client.chat(&request, &model).await
    };

    // Compute model class for metrics tracking
    let model_class_str = model_request
        .effective_class(&state.config.models)
        .map(|c| c.as_str().to_string());

    // Log response
    let (response, resp_log) = match result {
        Ok(resp) => {
            let mut resp_log = Response::new(request_id, 200);
            resp_log.response_body = serde_json::to_string(&resp).unwrap_or_default();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.runner_id = runner_id;
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = model_class_str.clone();
            if let Some(ref usage) = resp.usage {
                resp_log.tokens_prompt = Some(usage.prompt_tokens);
                resp_log.tokens_completion = Some(usage.completion_tokens);
            }
            (resp, resp_log)
        }
        Err(e) => {
            let mut resp_log = Response::new(request_id, 500);
            resp_log.response_body = e.to_string();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.runner_id = runner_id.clone();
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = model_class_str.clone();
            let _ = state.audit_logger.log_response(&resp_log);

            // Emit request event for admin dashboard (error case)
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

            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };

    let _ = state.audit_logger.log_response(&resp_log);

    // Emit request event for admin dashboard
    let _ = state.request_events.send(RequestEvent {
        id: req_log.id.clone(),
        timestamp: req_log.timestamp.to_rfc3339(),
        user_id: req_log.user_id.clone(),
        user_email: auth_user.email.clone(),
        request_path: req_log.request_path.clone(),
        model: req_log.model.clone(),
        client_ip: req_log.client_ip.clone(),
        status: Some(resp_log.status as i32),
        latency_ms: Some(resp_log.latency_ms as i64),
        tokens_prompt: resp_log.tokens_prompt.map(|v| v as i64),
        tokens_completion: resp_log.tokens_completion.map(|v| v as i64),
        runner_id: resp_log.runner_id.clone(),
        wol_sent: resp_log.wol_sent,
    });

    Ok(Json(response))
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/chat/completions", post(chat_completions))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use crate::models::chat::{ChatCompletionRequest, ChatMessage};

    fn create_test_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            model: None,
            temperature: None,
            max_tokens: None,
            tools: None,
            stream: None,
        }
    }

    #[tokio::test]
    async fn test_chat_request_serialization() {
        let req = create_test_request();
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("Hello"));
        assert!(json.contains("user"));
    }

    #[tokio::test]
    async fn test_chat_completion_request_default_values() {
        let req = ChatCompletionRequest {
            messages: vec![],
            model: None,
            temperature: None,
            max_tokens: None,
            tools: None,
            stream: None,
        };
        assert!(req.messages.is_empty());
        assert!(req.model.is_none());
    }

    #[tokio::test]
    async fn test_chat_message_default_content() {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
        };
        assert!(msg.content.is_none());
        assert!(msg.tool_calls.is_none());
    }

    #[tokio::test]
    async fn test_chat_request_with_model_override() {
        let req = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("Hi".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            model: Some("custom-model".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(100),
            tools: None,
            stream: None,
        };
        assert_eq!(req.model, Some("custom-model".to_string()));
        assert_eq!(req.temperature, Some(0.5));
    }
}
