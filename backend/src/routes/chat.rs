use axum::body::{Body, Bytes};
use axum::http::HeaderMap;
use axum::{
    extract::{ConnectInfo, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response as AxumResponse},
    routing::post,
    Json, Router,
};
use futures_util::stream;
use simple_ai_common::InferenceMetrics;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use super::auth_helpers::{authenticate_request, extract_client_ip};
use crate::gateway::{can_request_model, ModelRequest, SchedulerError};
use crate::models::chat::ChatCompletionRequest;
use crate::models::request::{Request, Response as AuditResponse};
use crate::{AppState, RequestEvent};

struct StreamLogContext {
    state: Arc<AppState>,
    request_id: String,
    req_log: Request,
    user_email: Option<String>,
    start: Instant,
    runner_id: Option<String>,
    wol_sent: bool,
    model_class: Option<String>,
}

struct GatewayStreamState {
    response: reqwest::Response,
    buffer: String,
    pending: VecDeque<Bytes>,
    metrics: Option<InferenceMetrics>,
    log_context: Option<StreamLogContext>,
}

fn filter_gateway_stream(response: reqwest::Response, log_context: StreamLogContext) -> Body {
    let state = GatewayStreamState {
        response,
        buffer: String::new(),
        pending: VecDeque::new(),
        metrics: None,
        log_context: Some(log_context),
    };

    let stream = stream::try_unfold(state, |mut state| async move {
        loop {
            if let Some(item) = state.pending.pop_front() {
                return Ok::<_, std::io::Error>(Some((item, state)));
            }

            let Some(chunk) = state
                .response
                .chunk()
                .await
                .map_err(|e| std::io::Error::other(e.to_string()))?
            else {
                if let Some(log_context) = state.log_context.take() {
                    log_stream_response(log_context, state.metrics).await;
                }
                return Ok(None);
            };

            let text =
                std::str::from_utf8(&chunk).map_err(|e| std::io::Error::other(e.to_string()))?;
            state.buffer.push_str(text);

            while let Some(event_end) = state.buffer.find("\n\n") {
                let event = state.buffer[..event_end].to_string();
                state.buffer.drain(..event_end + 2);

                if let Some(metrics) = parse_internal_metrics_event(&event) {
                    state.metrics = Some(metrics);
                    continue;
                }

                state
                    .pending
                    .push_back(Bytes::from(format!("{}\n\n", event)));
            }
        }
    });

    Body::from_stream(stream)
}

fn parse_internal_metrics_event(event: &str) -> Option<InferenceMetrics> {
    if !event
        .lines()
        .any(|line| line.trim() == "event: simple_ai_metrics")
    {
        return None;
    }

    let data = event
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>()
        .join("\n");

    serde_json::from_str(&data).ok()
}

async fn log_stream_response(log_context: StreamLogContext, metrics: Option<InferenceMetrics>) {
    let mut resp_log = AuditResponse::new(log_context.request_id, 200);
    resp_log.response_body = "[stream]".to_string();
    resp_log.latency_ms = log_context.start.elapsed().as_millis() as u64;
    resp_log.runner_id = log_context.runner_id.clone();
    resp_log.wol_sent = log_context.wol_sent;
    resp_log.model_class = log_context.model_class;
    resp_log.inference_metrics = metrics.clone();
    if let Some(metrics) = metrics {
        resp_log.tokens_prompt = metrics.prompt_tokens;
        resp_log.tokens_completion = metrics.completion_tokens;
    }

    let _ = log_context.state.audit_logger.log_response(&resp_log);
    log_context
        .state
        .router_telemetry
        .emit(
            "request_completed",
            format!(
                "Streaming request completed with status 200 on {}",
                log_context
                    .runner_id
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string())
            ),
            Some(log_context.req_log.id.clone()),
            log_context.runner_id.clone(),
            log_context.req_log.model.clone(),
        )
        .await;

    let _ = log_context.state.request_events.send(RequestEvent {
        id: log_context.req_log.id.clone(),
        timestamp: log_context.req_log.timestamp.to_rfc3339(),
        user_id: log_context.req_log.user_id.clone(),
        user_email: log_context.user_email,
        request_path: log_context.req_log.request_path.clone(),
        model: log_context.req_log.model.clone(),
        client_ip: log_context.req_log.client_ip.clone(),
        status: Some(resp_log.status as i32),
        latency_ms: Some(resp_log.latency_ms as i64),
        tokens_prompt: resp_log.tokens_prompt.map(|v| v as i64),
        tokens_completion: resp_log.tokens_completion.map(|v| v as i64),
        runner_id: resp_log.runner_id.clone(),
        wol_sent: resp_log.wol_sent,
    });
}

/// POST /v1/chat/completions - OpenAI-compatible chat endpoint
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<AxumResponse, (StatusCode, String)> {
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
            "Permission denied: cannot request specific models. Use class:fast or class:big."
                .to_string(),
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

    let request_id = state
        .audit_logger
        .log_request(&req_log)
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
    let is_streaming = request.stream.unwrap_or(false);
    // Check if batching should be used (enabled, non-streaming, has batch queue)
    let use_batching = state.batch_queue.is_some() && !is_streaming;

    // Compute model class for metrics tracking
    let model_class_str = model_request
        .effective_class(&state.config.models)
        .map(|c| c.as_str().to_string());

    if is_streaming {
        let stream_body = if state.config.gateway.enabled {
            let scheduled = match state
                .request_scheduler
                .chat_completion_stream(&req_log.id, &model, &model_request, &request)
                .await
            {
                Ok(scheduled) => scheduled,
                Err(SchedulerError::Wake(e)) => {
                    let e = crate::llm::OllamaError::ConnectionFailed(format!(
                        "Failed to wake inference runners: {}",
                        e
                    ));
                    let mut resp_log = AuditResponse::new(request_id, 500);
                    resp_log.response_body = e.to_string();
                    resp_log.latency_ms = start.elapsed().as_millis() as u64;
                    resp_log.model_class = model_class_str.clone();
                    let _ = state.audit_logger.log_response(&resp_log);
                    return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
                }
                Err(SchedulerError::Router(e)) => {
                    let e = crate::llm::OllamaError::ConnectionFailed(e.to_string());
                    let mut resp_log = AuditResponse::new(request_id, 500);
                    resp_log.response_body = e.to_string();
                    resp_log.latency_ms = start.elapsed().as_millis() as u64;
                    resp_log.model_class = model_class_str.clone();
                    let _ = state.audit_logger.log_response(&resp_log);
                    return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
                }
            };

            state
                .wake_service
                .keepalive_runner(scheduled.runner_id.clone());
            state
                .router_telemetry
                .emit(
                    "request_dispatched",
                    format!(
                        "Streaming request dispatched to {} using {}",
                        scheduled.runner_id, scheduled.resolved_model
                    ),
                    Some(req_log.id.clone()),
                    Some(scheduled.runner_id.clone()),
                    Some(scheduled.resolved_model.clone()),
                )
                .await;

            filter_gateway_stream(
                scheduled.response,
                StreamLogContext {
                    state: state.clone(),
                    request_id: request_id.clone(),
                    req_log: req_log.clone(),
                    user_email: auth_user.email.clone(),
                    start,
                    runner_id: Some(scheduled.runner_id.clone()),
                    wol_sent: scheduled.wol_sent,
                    model_class: model_class_str.clone(),
                },
            )
        } else if !state.circuit_breaker.is_available("ollama") {
            let e = crate::llm::OllamaError::ConnectionFailed(
                "Circuit breaker open: Ollama backend is unavailable".to_string(),
            );
            let mut resp_log = AuditResponse::new(request_id, 500);
            resp_log.response_body = e.to_string();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.model_class = model_class_str.clone();
            let _ = state.audit_logger.log_response(&resp_log);
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        } else {
            let result = state.ollama_client.chat_stream(&request, &model).await;
            if result.is_ok() {
                state.circuit_breaker.record_success("ollama");
            } else {
                state.circuit_breaker.record_failure("ollama");
            }
            match result {
                Ok(stream) => {
                    let mut resp_log = AuditResponse::new(request_id.clone(), 200);
                    resp_log.response_body = "[stream]".to_string();
                    resp_log.latency_ms = start.elapsed().as_millis() as u64;
                    resp_log.runner_id = runner_id.clone();
                    resp_log.wol_sent = wol_sent;
                    resp_log.model_class = model_class_str.clone();
                    let _ = state.audit_logger.log_response(&resp_log);
                    Body::from_stream(stream)
                }
                Err(e) => {
                    let mut resp_log = AuditResponse::new(request_id, 500);
                    resp_log.response_body = e.to_string();
                    resp_log.latency_ms = start.elapsed().as_millis() as u64;
                    resp_log.model_class = model_class_str.clone();
                    let _ = state.audit_logger.log_response(&resp_log);
                    return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
                }
            }
        };

        let mut response = AxumResponse::new(stream_body);
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        response
            .headers_mut()
            .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
        return Ok(response);
    }

    let result = if state.config.gateway.enabled {
        match state
            .request_scheduler
            .chat_completion(&req_log.id, &model, &model_request, &request, use_batching)
            .await
        {
            Ok(scheduled) => {
                runner_id = Some(scheduled.runner_id.clone());
                wol_sent = scheduled.wol_sent;
                state
                    .wake_service
                    .keepalive_runner(scheduled.runner_id.clone());
                state
                    .router_telemetry
                    .emit(
                        "request_dispatched",
                        format!(
                            "Request dispatched to {} using {}",
                            scheduled.runner_id, scheduled.resolved_model
                        ),
                        Some(req_log.id.clone()),
                        Some(scheduled.runner_id.clone()),
                        Some(scheduled.resolved_model.clone()),
                    )
                    .await;
                Ok(scheduled.response)
            }
            Err(SchedulerError::Wake(e)) => Err(crate::llm::OllamaError::ConnectionFailed(
                format!("Failed to wake inference runners: {}", e),
            )),
            Err(SchedulerError::Router(e)) => {
                Err(crate::llm::OllamaError::ConnectionFailed(e.to_string()))
            }
        }
    } else if !state.circuit_breaker.is_available("ollama") {
        Err(crate::llm::OllamaError::ConnectionFailed(
            "Circuit breaker open: Ollama backend is unavailable".to_string(),
        ))
    } else {
        let result = state.ollama_client.chat(&request, &model).await;
        if result.is_ok() {
            state.circuit_breaker.record_success("ollama");
        } else {
            state.circuit_breaker.record_failure("ollama");
        }
        result
    };

    let (response, resp_log) = match result {
        Ok(resp) => {
            let metrics = resp.inference_metrics.clone();
            let client_resp = resp.strip_internal_metrics();
            let mut resp_log = AuditResponse::new(request_id, 200);
            resp_log.response_body = serde_json::to_string(&client_resp).unwrap_or_default();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.runner_id = runner_id.clone();
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = model_class_str.clone();
            resp_log.inference_metrics = metrics;
            if let Some(ref usage) = client_resp.usage {
                resp_log.tokens_prompt = Some(usage.prompt_tokens);
                resp_log.tokens_completion = Some(usage.completion_tokens);
            }
            state
                .router_telemetry
                .emit(
                    "request_completed",
                    format!(
                        "Request completed with status 200 on {}",
                        runner_id.clone().unwrap_or_else(|| "unknown".to_string())
                    ),
                    Some(req_log.id.clone()),
                    runner_id.clone(),
                    req_log.model.clone(),
                )
                .await;
            (client_resp, resp_log)
        }
        Err(e) => {
            let mut resp_log = AuditResponse::new(request_id, 500);
            resp_log.response_body = e.to_string();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.runner_id = runner_id.clone();
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = model_class_str.clone();
            let _ = state.audit_logger.log_response(&resp_log);
            state
                .router_telemetry
                .emit(
                    "request_failed",
                    format!("Request failed: {}", e),
                    Some(req_log.id.clone()),
                    runner_id.clone(),
                    req_log.model.clone(),
                )
                .await;
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

    Ok(Json(response).into_response())
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/chat/completions", post(chat_completions))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::parse_internal_metrics_event;
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

    #[test]
    fn test_parse_internal_metrics_event() {
        let event = r#"event: simple_ai_metrics
data: {"resolved_model":"model-a","engine_type":"llama_cpp","context_window":8192,"prompt_tokens":10,"completion_tokens":5}
"#;

        let metrics = parse_internal_metrics_event(event).unwrap();
        assert_eq!(metrics.resolved_model, Some("model-a".to_string()));
        assert_eq!(metrics.engine_type, Some("llama_cpp".to_string()));
        assert_eq!(metrics.context_window, Some(8192));
        assert_eq!(metrics.prompt_tokens, Some(10));
        assert_eq!(metrics.completion_tokens, Some(5));
    }
}
