use axum::body::{Body, Bytes};
use axum::http::HeaderMap;
use axum::{
    extract::{ConnectInfo, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response as AxumResponse},
    routing::post,
    Json, Router,
};
use futures_util::{stream, StreamExt};
use simple_ai_common::{
    ChatCompletionChunk, ChatCompletionRequest, InferenceMetrics, ResponseCreateRequest,
    ResponseInput, ResponseObject, ResponseOutputContent, ResponseOutputItem,
    ResponseOutputMessage,
};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use super::auth_helpers::{authenticate_request, extract_client_ip};
use crate::gateway::{can_request_model, ModelRequest, SchedulerError};
use crate::models::request::{Request, Response as AuditResponse};
use crate::{AppState, RequestEvent};

struct ResponsesStreamLogContext {
    state: Arc<AppState>,
    request_id: String,
    req_log: Request,
    user_email: Option<String>,
    start: Instant,
    runner_id: Option<String>,
    wol_sent: bool,
    model_class: Option<String>,
}

struct ResponsesStreamState {
    response: Option<reqwest::Response>,
    buffer: String,
    pending: VecDeque<Bytes>,
    metrics: Option<InferenceMetrics>,
    log_context: Option<ResponsesStreamLogContext>,
    emitted_created: bool,
    emitted_message: bool,
    message_id: String,
    collected_text: String,
    output_items: Vec<ResponseOutputItem>,
    response_id: Option<String>,
    response_model: Option<String>,
    response_created_at: Option<i64>,
}

fn sse_event(event: &str, payload: &serde_json::Value) -> Result<Bytes, std::io::Error> {
    let json = serde_json::to_string(payload).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(Bytes::from(format!("event: {event}\ndata: {json}\n\n")))
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

fn map_chat_chunk_event(
    state: &mut ResponsesStreamState,
    event: &str,
) -> Result<(), std::io::Error> {
    let data = event
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>()
        .join("\n");

    if data == "[DONE]" {
        if let (Some(response_id), Some(model), Some(created_at)) = (
            state.response_id.clone(),
            state.response_model.clone(),
            state.response_created_at,
        ) {
            let completed = serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "completed",
                    "model": model,
                    "output": state.output_items.clone(),
                    "output_text": state.collected_text.clone(),
                }
            });
            state
                .pending
                .push_back(sse_event("response.completed", &completed)?);
        }
        state.pending.push_back(Bytes::from("data: [DONE]\n\n"));
        return Ok(());
    }

    let chunk: ChatCompletionChunk =
        serde_json::from_str(&data).map_err(|e| std::io::Error::other(e.to_string()))?;

    if !state.emitted_created {
        state.response_id = Some(chunk.id.clone());
        state.response_model = Some(chunk.model.clone());
        state.response_created_at = Some(chunk.created);
        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": chunk.id,
                "object": "response",
                "created_at": chunk.created,
                "status": "in_progress",
                "model": chunk.model,
                "output": [],
                "output_text": "",
            }
        });
        state
            .pending
            .push_back(sse_event("response.created", &created)?);
        state.emitted_created = true;
    }

    let Some(choice) = chunk.choices.first() else {
        return Ok(());
    };

    if let Some(text) = &choice.delta.content {
        if !text.is_empty() {
            if !state.emitted_message {
                let message = ResponseOutputItem::Message(ResponseOutputMessage {
                    id: state.message_id.clone(),
                    role: "assistant".to_string(),
                    content: vec![ResponseOutputContent {
                        content_type: "output_text".to_string(),
                        text: String::new(),
                    }],
                });
                state.output_items.push(message.clone());
                let added = serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": state.output_items.len() - 1,
                    "item": message,
                });
                state
                    .pending
                    .push_back(sse_event("response.output_item.added", &added)?);
                state.emitted_message = true;
            }

            state.collected_text.push_str(text);
            let delta = serde_json::json!({
                "type": "response.output_text.delta",
                "item_id": state.message_id.clone(),
                "delta": text,
            });
            state
                .pending
                .push_back(sse_event("response.output_text.delta", &delta)?);
        }
    }

    if let Some(tool_calls) = &choice.delta.tool_calls {
        for tool_call in tool_calls.iter().cloned() {
            let item = simple_ai_common::ResponseOutputItem::from_tool_call(tool_call);
            state.output_items.push(item.clone());
            let added = serde_json::json!({
                "type": "response.output_item.added",
                "output_index": state.output_items.len() - 1,
                "item": item,
            });
            state
                .pending
                .push_back(sse_event("response.output_item.added", &added)?);
        }
    }

    if choice.finish_reason.is_some() && state.emitted_message {
        if let Some(ResponseOutputItem::Message(message)) = state.output_items.iter_mut().find(
            |item| matches!(item, ResponseOutputItem::Message(msg) if msg.id == state.message_id),
        ) {
            if let Some(content) = message.content.first_mut() {
                content.text = state.collected_text.clone();
            }
        }

        let done = serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": state.message_id.clone(),
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": state.collected_text.clone(),
                }]
            }
        });
        state
            .pending
            .push_back(sse_event("response.output_item.done", &done)?);
    }

    Ok(())
}

fn filter_responses_stream(
    response: reqwest::Response,
    log_context: ResponsesStreamLogContext,
) -> Body {
    let state = ResponsesStreamState {
        response: Some(response),
        buffer: String::new(),
        pending: VecDeque::new(),
        metrics: None,
        log_context: Some(log_context),
        emitted_created: false,
        emitted_message: false,
        message_id: format!("msg_{}", uuid::Uuid::new_v4()),
        collected_text: String::new(),
        output_items: Vec::new(),
        response_id: None,
        response_model: None,
        response_created_at: None,
    };

    let stream = stream::try_unfold(state, |mut state| async move {
        loop {
            if let Some(item) = state.pending.pop_front() {
                return Ok::<_, std::io::Error>(Some((item, state)));
            }

            let Some(chunk) = state
                .response
                .as_mut()
                .expect("gateway response stream should be present")
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

                map_chat_chunk_event(&mut state, &event)?;
            }
        }
    });

    Body::from_stream(stream)
}

fn filter_responses_stream_from_chat_stream(
    stream: crate::llm::ChatStream,
    log_context: ResponsesStreamLogContext,
) -> Body {
    let state = ResponsesStreamState {
        response: None,
        buffer: String::new(),
        pending: VecDeque::new(),
        metrics: None,
        log_context: Some(log_context),
        emitted_created: false,
        emitted_message: false,
        message_id: format!("msg_{}", uuid::Uuid::new_v4()),
        collected_text: String::new(),
        output_items: Vec::new(),
        response_id: None,
        response_model: None,
        response_created_at: None,
    };

    let stream = stream::try_unfold((stream, state), |(mut stream, mut state)| async move {
        loop {
            if let Some(item) = state.pending.pop_front() {
                return Ok::<_, std::io::Error>(Some((item, (stream, state))));
            }

            let Some(chunk) = stream
                .next()
                .await
                .transpose()
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

                map_chat_chunk_event(&mut state, &event)?;
            }
        }
    });

    Body::from_stream(stream)
}

async fn log_stream_response(
    log_context: ResponsesStreamLogContext,
    metrics: Option<InferenceMetrics>,
) {
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

fn build_chat_request(request: ResponseCreateRequest) -> ChatCompletionRequest {
    let ResponseCreateRequest {
        model,
        input,
        tools,
        temperature,
        max_output_tokens,
        stream,
    } = request;

    ChatCompletionRequest {
        messages: match input {
            ResponseInput::Text(text) => ResponseInput::Text(text).into_chat_messages(),
            ResponseInput::Items(items) => ResponseInput::Items(items).into_chat_messages(),
        },
        tools,
        model: Some(model),
        temperature,
        max_tokens: max_output_tokens,
        stream,
    }
}

/// POST /v1/responses - OpenAI-compatible responses endpoint.
async fn create_response(
    State(state): State<Arc<AppState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<ResponseCreateRequest>,
) -> Result<AxumResponse, (StatusCode, String)> {
    let start = Instant::now();
    let (auth_user, user) = authenticate_request(&state, &headers).await?;
    let chat_request = build_chat_request(request);

    let model_request = ModelRequest::parse(
        chat_request
            .model
            .as_deref()
            .ok_or((StatusCode::BAD_REQUEST, "model is required".to_string()))?,
    );

    if !can_request_model(&auth_user.roles, &model_request) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Permission denied: cannot request specific models. Use class:fast or class:big."
                .to_string(),
        ));
    }

    let model = match &model_request {
        ModelRequest::Specific(m) => m.clone(),
        ModelRequest::Class(class) => format!("class:{}", class),
    };

    let mut req_log = Request::new(user.id.clone(), "/v1/responses".to_string());
    req_log.request_body = serde_json::to_string(&chat_request).unwrap_or_default();
    req_log.model = Some(model.clone());
    req_log.client_ip = extract_client_ip(&headers, connect_info.map(|c| c.0));

    let request_id = state
        .audit_logger
        .log_request(&req_log)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let _ = state.request_events.send(RequestEvent {
        id: req_log.id.clone(),
        timestamp: req_log.timestamp.to_rfc3339(),
        user_id: req_log.user_id.clone(),
        user_email: auth_user.email.clone(),
        request_path: req_log.request_path.clone(),
        model: req_log.model.clone(),
        client_ip: req_log.client_ip.clone(),
        status: None,
        latency_ms: None,
        tokens_prompt: None,
        tokens_completion: None,
        runner_id: None,
        wol_sent: false,
    });

    let mut runner_id: Option<String> = None;
    let mut wol_sent = false;
    let is_streaming = chat_request.stream.unwrap_or(false);
    let use_batching = state.batch_queue.is_some() && !is_streaming;
    let model_class_str = model_request
        .effective_class(&state.config.models)
        .map(|c| c.as_str().to_string());

    if is_streaming {
        let stream_body = if state.config.gateway.enabled {
            let scheduled = match state
                .request_scheduler
                .chat_completion_stream(&req_log.id, &model, &model_request, &chat_request)
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
                        "Streaming response request dispatched to {} using {}",
                        scheduled.runner_id, scheduled.resolved_model
                    ),
                    Some(req_log.id.clone()),
                    Some(scheduled.runner_id.clone()),
                    Some(scheduled.resolved_model.clone()),
                )
                .await;

            filter_responses_stream(
                scheduled.response,
                ResponsesStreamLogContext {
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
            let result = state.ollama_client.chat_stream(&chat_request, &model).await;
            if result.is_ok() {
                state.circuit_breaker.record_success("ollama");
            } else {
                state.circuit_breaker.record_failure("ollama");
            }
            match result {
                Ok(stream) => filter_responses_stream_from_chat_stream(
                    stream,
                    ResponsesStreamLogContext {
                        state: state.clone(),
                        request_id: request_id.clone(),
                        req_log: req_log.clone(),
                        user_email: auth_user.email.clone(),
                        start,
                        runner_id: runner_id.clone(),
                        wol_sent,
                        model_class: model_class_str.clone(),
                    },
                ),
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
            .chat_completion(
                &req_log.id,
                &model,
                &model_request,
                &chat_request,
                use_batching,
            )
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
                            "Responses request dispatched to {} using {}",
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
        let result = state.ollama_client.chat(&chat_request, &model).await;
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
            let client_resp = ResponseObject::from_chat(resp.strip_internal_metrics());
            let mut resp_log = AuditResponse::new(request_id, 200);
            resp_log.response_body = serde_json::to_string(&client_resp).unwrap_or_default();
            resp_log.latency_ms = start.elapsed().as_millis() as u64;
            resp_log.runner_id = runner_id.clone();
            resp_log.wol_sent = wol_sent;
            resp_log.model_class = model_class_str.clone();
            resp_log.inference_metrics = metrics;
            if let Some(ref usage) = client_resp.usage {
                resp_log.tokens_prompt = Some(usage.input_tokens);
                resp_log.tokens_completion = Some(usage.output_tokens);
            }
            state
                .router_telemetry
                .emit(
                    "request_completed",
                    format!(
                        "Responses request completed with status 200 on {}",
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
                    format!("Responses request failed: {}", e),
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
        .route("/responses", post(create_response))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::{
        build_chat_request, map_chat_chunk_event, parse_internal_metrics_event,
        ResponsesStreamState,
    };
    use serde_json::Value;
    use simple_ai_common::{
        ResponseContent, ResponseContentPart, ResponseInputItem, ResponseTypedInputItem,
    };
    use std::collections::VecDeque;

    #[test]
    fn test_build_chat_request_from_text_input() {
        let chat = build_chat_request(simple_ai_common::ResponseCreateRequest {
            model: "test-model".to_string(),
            input: simple_ai_common::ResponseInput::Text("Hello".to_string()),
            tools: None,
            temperature: Some(0.4),
            max_output_tokens: Some(32),
            stream: Some(false),
        });

        assert_eq!(chat.model.as_deref(), Some("test-model"));
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.max_tokens, Some(32));
    }

    #[test]
    fn test_build_chat_request_from_items_input() {
        let chat = build_chat_request(simple_ai_common::ResponseCreateRequest {
            model: "class:fast".to_string(),
            input: simple_ai_common::ResponseInput::Items(vec![
                ResponseInputItem::Typed(ResponseTypedInputItem::Message {
                    role: "user".to_string(),
                    content: ResponseContent::Parts(vec![
                        ResponseContentPart {
                            part_type: "input_text".to_string(),
                            text: Some("Hello".to_string()),
                        },
                        ResponseContentPart {
                            part_type: "input_text".to_string(),
                            text: Some(" world".to_string()),
                        },
                    ]),
                    tool_call_id: None,
                }),
                ResponseInputItem::Typed(ResponseTypedInputItem::FunctionCallOutput {
                    call_id: "call_1".to_string(),
                    output: "{\"ok\":true}".to_string(),
                }),
            ]),
            tools: None,
            temperature: None,
            max_output_tokens: None,
            stream: Some(false),
        });

        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.messages[0].content.as_deref(), Some("Hello world"));
        assert_eq!(chat.messages[1].role, "tool");
        assert_eq!(chat.messages[1].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat.messages[1].content.as_deref(), Some("{\"ok\":true}"));
    }

    #[test]
    fn test_parse_internal_metrics_event() {
        let event = r#"event: simple_ai_metrics
data: {"resolved_model":"model-a","engine_type":"llama_cpp","context_window":8192,"prompt_tokens":10,"completion_tokens":5}
"#;

        let metrics = parse_internal_metrics_event(event).unwrap();
        assert_eq!(metrics.resolved_model.as_deref(), Some("model-a"));
        assert_eq!(metrics.prompt_tokens, Some(10));
    }

    #[test]
    fn test_map_chat_chunk_event_to_responses_events() {
        let mut state = ResponsesStreamState {
            response: None,
            buffer: String::new(),
            pending: VecDeque::new(),
            metrics: None,
            log_context: None,
            emitted_created: false,
            emitted_message: false,
            message_id: "msg_test".to_string(),
            collected_text: String::new(),
            output_items: Vec::new(),
            response_id: None,
            response_model: None,
            response_created_at: None,
        };

        let event = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel","tool_calls":null,"tool_call_id":null},"finish_reason":null}]}"#;
        map_chat_chunk_event(&mut state, event).unwrap();
        assert_eq!(state.pending.len(), 3);
        assert_eq!(state.collected_text, "Hel");
    }

    #[test]
    fn test_map_chat_chunk_event_with_tool_call() {
        let mut state = ResponsesStreamState {
            response: None,
            buffer: String::new(),
            pending: VecDeque::new(),
            metrics: None,
            log_context: None,
            emitted_created: false,
            emitted_message: false,
            message_id: "msg_test".to_string(),
            collected_text: String::new(),
            output_items: Vec::new(),
            response_id: None,
            response_model: None,
            response_created_at: None,
        };

        let event = serde_json::json!({
            "id":"chatcmpl-1",
            "object":"chat.completion.chunk",
            "created":123,
            "model":"test-model",
            "choices":[{
                "index":0,
                "delta":{
                    "role":"assistant",
                    "content":null,
                    "tool_calls":[{
                        "id":"call_1",
                        "type":"function",
                        "function":{"name":"lookup","arguments":"{\"city\":\"Rome\"}"}
                    }],
                    "tool_call_id":null
                },
                "finish_reason":"tool_calls"
            }]
        });

        let raw = format!("data: {}", serde_json::to_string(&event).unwrap());
        map_chat_chunk_event(&mut state, &raw).unwrap();

        assert_eq!(state.output_items.len(), 1);
        match &state.output_items[0] {
            simple_ai_common::ResponseOutputItem::FunctionCall(call) => {
                assert_eq!(call.call_id, "call_1");
                assert_eq!(call.name, "lookup");
            }
            _ => panic!("expected function call output item"),
        }

        let events: Vec<Value> = state
            .pending
            .iter()
            .filter_map(|bytes| {
                let text = std::str::from_utf8(bytes).ok()?;
                let line = text.lines().find(|line| line.starts_with("data: "))?;
                serde_json::from_str(line.trim_start_matches("data: ")).ok()
            })
            .collect();
        assert_eq!(events[0]["type"], "response.created");
        assert_eq!(events[1]["type"], "response.output_item.added");
    }

    #[test]
    fn test_map_chat_done_event_emits_completed() {
        let mut state = ResponsesStreamState {
            response: None,
            buffer: String::new(),
            pending: VecDeque::new(),
            metrics: None,
            log_context: None,
            emitted_created: true,
            emitted_message: true,
            message_id: "msg_test".to_string(),
            collected_text: "Hello".to_string(),
            output_items: vec![simple_ai_common::ResponseOutputItem::Message(
                simple_ai_common::ResponseOutputMessage {
                    id: "msg_test".to_string(),
                    role: "assistant".to_string(),
                    content: vec![simple_ai_common::ResponseOutputContent {
                        content_type: "output_text".to_string(),
                        text: "Hello".to_string(),
                    }],
                },
            )],
            response_id: Some("resp_1".to_string()),
            response_model: Some("test-model".to_string()),
            response_created_at: Some(123),
        };

        map_chat_chunk_event(&mut state, "data: [DONE]").unwrap();

        let completed = std::str::from_utf8(&state.pending[0]).unwrap();
        assert!(completed.contains("event: response.completed"));
        assert!(completed.contains("\"output_text\":\"Hello\""));
        assert_eq!(
            std::str::from_utf8(&state.pending[1]).unwrap(),
            "data: [DONE]\n\n"
        );
    }
}
