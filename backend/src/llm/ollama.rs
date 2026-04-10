use axum::body::Bytes;
use futures_util::stream::{self, Stream};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use simple_ai_common::{format_sse_chunk, format_sse_done, ChatCompletionChunk};
use std::collections::VecDeque;
use std::pin::Pin;

use crate::models::chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolCall, ToolFunction,
};

/// Client for communicating with Ollama API.
pub struct OllamaClient {
    http_client: Client,
    base_url: String,
    #[allow(dead_code)]
    default_model: String,
}

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<Bytes, OllamaError>> + Send>>;

/// Ollama chat request format.
#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaToolCall {
    #[serde(default)]
    id: Option<String>,
    function: OllamaToolFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaToolFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

/// Ollama chat response format.
#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(String),
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    #[error("Ollama error: {0}")]
    OllamaError(String),
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
}

impl OllamaClient {
    pub fn new(base_url: &str, default_model: &str) -> Self {
        Self {
            http_client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            default_model: default_model.to_string(),
        }
    }

    fn build_chat_request(
        &self,
        request: &ChatCompletionRequest,
        model: &str,
        stream: bool,
    ) -> OllamaChatRequest {
        let messages: Vec<OllamaMessage> = request
            .messages
            .iter()
            .map(|m| {
                let tool_calls = m.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| OllamaToolCall {
                            id: Some(tc.id.clone()),
                            function: OllamaToolFunction {
                                name: tc.function.name.clone(),
                                arguments: serde_json::from_str(&tc.function.arguments)
                                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                            },
                        })
                        .collect()
                });

                OllamaMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    tool_calls,
                    tool_call_id: m.tool_call_id.clone(),
                }
            })
            .collect();

        let options = if request.temperature.is_some() || request.max_tokens.is_some() {
            Some(OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens,
            })
        } else {
            None
        };

        OllamaChatRequest {
            model: model.to_string(),
            messages,
            stream,
            options,
            tools: request.tools.clone(),
        }
    }

    /// Send a chat request to Ollama and translate the response to OpenAI format.
    pub async fn chat(
        &self,
        request: &ChatCompletionRequest,
        model: &str,
    ) -> Result<ChatCompletionResponse, OllamaError> {
        let ollama_request = self.build_chat_request(request, model, false);

        let url = format!("{}/api/chat", self.base_url);

        tracing::debug!("Sending request to Ollama: {}", url);

        let response = self
            .http_client
            .post(&url)
            .json(&ollama_request)
            .send()
            .await
            .map_err(|e| OllamaError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OllamaError::OllamaError(format!("{}: {}", status, body)));
        }

        let ollama_response: OllamaChatResponse = response
            .json()
            .await
            .map_err(|e| OllamaError::InvalidResponse(e.to_string()))?;

        // Convert tool_calls from Ollama format to OpenAI format
        let tool_calls = ollama_response.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: tc.id.unwrap_or_else(|| format!("call_{}", i)),
                    call_type: "function".to_string(),
                    function: ToolFunction {
                        name: tc.function.name,
                        arguments: tc.function.arguments.to_string(),
                    },
                })
                .collect()
        });

        // Convert to OpenAI format
        let message = ChatMessage {
            role: ollama_response.message.role,
            content: ollama_response.message.content,
            tool_calls,
            tool_call_id: None,
        };

        // Determine finish_reason based on whether tool_calls are present
        let finish_reason = if ollama_response.done {
            if message.tool_calls.is_some() {
                Some("tool_calls".to_string())
            } else {
                Some("stop".to_string())
            }
        } else {
            None
        };

        let mut response = ChatCompletionResponse::new(model.to_string(), message, finish_reason);

        // Add usage info if available
        if let (Some(prompt), Some(completion)) = (
            ollama_response.prompt_eval_count,
            ollama_response.eval_count,
        ) {
            response = response.with_usage(prompt, completion);
        }

        Ok(response)
    }

    pub async fn chat_stream(
        &self,
        request: &ChatCompletionRequest,
        model: &str,
    ) -> Result<ChatStream, OllamaError> {
        let ollama_request = self.build_chat_request(request, model, true);
        let url = format!("{}/api/chat", self.base_url);

        tracing::debug!("Sending streaming request to Ollama: {}", url);

        let response = self
            .http_client
            .post(&url)
            .json(&ollama_request)
            .send()
            .await
            .map_err(|e| OllamaError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OllamaError::OllamaError(format!("{}: {}", status, body)));
        }

        struct StreamState {
            response: reqwest::Response,
            buffer: String,
            pending: VecDeque<Bytes>,
            sent_role: bool,
            emitted_done: bool,
            chunk_id: String,
            created: i64,
            model: String,
        }

        let state = StreamState {
            response,
            buffer: String::new(),
            pending: VecDeque::new(),
            sent_role: false,
            emitted_done: false,
            chunk_id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
        };

        let stream = stream::try_unfold(state, |mut state| async move {
            loop {
                if let Some(item) = state.pending.pop_front() {
                    return Ok(Some((item, state)));
                }

                let Some(chunk) = state
                    .response
                    .chunk()
                    .await
                    .map_err(|e| OllamaError::RequestFailed(e.to_string()))?
                else {
                    if !state.emitted_done {
                        state.emitted_done = true;
                        return Ok(Some((Bytes::from(format_sse_done()), state)));
                    }
                    return Ok(None);
                };

                let text = std::str::from_utf8(&chunk)
                    .map_err(|e| OllamaError::InvalidResponse(e.to_string()))?;
                state.buffer.push_str(text);

                while let Some(newline_idx) = state.buffer.find('\n') {
                    let line = state.buffer[..newline_idx].trim().to_string();
                    state.buffer.drain(..=newline_idx);

                    if line.is_empty() {
                        continue;
                    }

                    let parsed: OllamaChatResponse = serde_json::from_str(&line)
                        .map_err(|e| OllamaError::InvalidResponse(e.to_string()))?;

                    let tool_calls = parsed.message.tool_calls.map(|calls| {
                        calls
                            .into_iter()
                            .enumerate()
                            .map(|(i, tc)| ToolCall {
                                id: tc.id.unwrap_or_else(|| format!("call_{}", i)),
                                call_type: "function".to_string(),
                                function: ToolFunction {
                                    name: tc.function.name,
                                    arguments: tc.function.arguments.to_string(),
                                },
                            })
                            .collect::<Vec<_>>()
                    });

                    let has_payload = parsed
                        .message
                        .content
                        .as_ref()
                        .is_some_and(|c| !c.is_empty())
                        || tool_calls.is_some();

                    if has_payload || !state.sent_role {
                        let delta = ChatMessage {
                            role: if state.sent_role {
                                String::new()
                            } else {
                                parsed.message.role.clone()
                            },
                            content: parsed.message.content.clone(),
                            tool_calls: tool_calls.clone(),
                            tool_call_id: None,
                        };
                        let chunk = ChatCompletionChunk::new(
                            state.chunk_id.clone(),
                            state.created,
                            state.model.clone(),
                            delta,
                            None,
                        );
                        let payload = format_sse_chunk(&chunk)
                            .map_err(|e| OllamaError::InvalidResponse(e.to_string()))?;
                        state.pending.push_back(Bytes::from(payload));
                        state.sent_role = true;
                    }

                    if parsed.done {
                        let finish_reason = if tool_calls.is_some() {
                            Some("tool_calls".to_string())
                        } else {
                            Some("stop".to_string())
                        };
                        let final_chunk = ChatCompletionChunk::new(
                            state.chunk_id.clone(),
                            state.created,
                            state.model.clone(),
                            ChatMessage {
                                role: String::new(),
                                content: None,
                                tool_calls: None,
                                tool_call_id: None,
                            },
                            finish_reason,
                        );
                        let payload = format_sse_chunk(&final_chunk)
                            .map_err(|e| OllamaError::InvalidResponse(e.to_string()))?;
                        state.pending.push_back(Bytes::from(payload));
                        state.pending.push_back(Bytes::from(format_sse_done()));
                        state.emitted_done = true;
                    }
                }
            }
        });

        Ok(Box::pin(stream))
    }

    /// Send an embedding request to Ollama and return the embeddings.
    pub async fn embed(&self, model: &str, input: &[String]) -> Result<Vec<Vec<f32>>, OllamaError> {
        let ollama_request = OllamaEmbedRequest {
            model: model.to_string(),
            input: input.to_vec(),
        };

        let url = format!("{}/api/embed", self.base_url);

        tracing::debug!("Sending embed request to Ollama: {}", url);

        let response = self
            .http_client
            .post(&url)
            .json(&ollama_request)
            .send()
            .await
            .map_err(|e| OllamaError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OllamaError::OllamaError(format!("{}: {}", status, body)));
        }

        let ollama_response: OllamaEmbedResponse = response
            .json()
            .await
            .map_err(|e| OllamaError::InvalidResponse(e.to_string()))?;

        Ok(ollama_response.embeddings)
    }
}

/// Ollama embed request format (POST /api/embed).
#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
}

/// Ollama embed response format.
#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_ollama_client_new() {
        let client = OllamaClient::new("http://localhost:11434", "llama2");
        assert_eq!(client.base_url, "http://localhost:11434");
        assert_eq!(client.default_model, "llama2");
    }

    #[test]
    fn test_ollama_client_url_trailing_slash_handled() {
        let client = OllamaClient::new("http://localhost:11434/", "llama2");
        assert_eq!(client.base_url, "http://localhost:11434");
        assert!(!client.base_url.ends_with("/"));
    }

    #[test]
    fn test_ollama_client_full_url_construction() {
        let client = OllamaClient::new("http://localhost:11434", "llama2");
        let full_url = format!("{}/api/chat", client.base_url);
        assert_eq!(full_url, "http://localhost:11434/api/chat");
    }

    #[test]
    fn test_ollama_chat_request_serialization() {
        let request = OllamaChatRequest {
            model: "llama2".to_string(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            options: None,
            tools: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains(r#""model":"llama2""#));
        assert!(json.contains(r#""role":"user""#));
        assert!(json.contains(r#""content":"Hello""#));
        assert!(json.contains(r#""stream":false"#));
    }

    #[test]
    fn test_ollama_chat_request_with_options() {
        let request = OllamaChatRequest {
            model: "llama2".to_string(),
            messages: vec![],
            stream: false,
            options: Some(OllamaOptions {
                temperature: Some(0.7),
                num_predict: Some(100),
            }),
            tools: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains(r#""temperature":0.7"#));
        assert!(json.contains(r#""num_predict":100"#));
    }

    #[test]
    fn test_ollama_chat_request_options_none() {
        let request = OllamaChatRequest {
            model: "llama2".to_string(),
            messages: vec![],
            stream: false,
            options: None,
            tools: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("temperature"));
        assert!(!json.contains("num_predict"));
    }

    #[test]
    fn test_ollama_response_deserialization() {
        let json = r#"{
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "done": true,
            "prompt_eval_count": 10,
            "eval_count": 5
        }"#;
        let response: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.message.role, "assistant");
        assert_eq!(response.message.content, Some("Hello!".to_string()));
        assert!(response.done);
        assert_eq!(response.prompt_eval_count, Some(10));
        assert_eq!(response.eval_count, Some(5));
    }

    #[test]
    fn test_ollama_response_without_usage() {
        let json = r#"{
            "message": {
                "role": "assistant",
                "content": "Hi"
            },
            "done": false
        }"#;
        let response: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.message.content, Some("Hi".to_string()));
        assert!(!response.done);
        assert!(response.prompt_eval_count.is_none());
        assert!(response.eval_count.is_none());
    }

    #[test]
    fn test_ollama_tool_call_serialization() {
        let tool_call = OllamaToolCall {
            id: Some("call_1".to_string()),
            function: OllamaToolFunction {
                name: "get_weather".to_string(),
                arguments: serde_json::json!({"location": "NYC"}),
            },
        };
        let json = serde_json::to_string(&tool_call).unwrap();
        assert!(json.contains(r#""id":"call_1""#));
        assert!(json.contains(r#""name":"get_weather""#));
    }

    #[test]
    fn test_ollama_error_request_failed() {
        let error = OllamaError::RequestFailed("connection refused".to_string());
        assert!(error.to_string().contains("HTTP request failed"));
    }

    #[test]
    fn test_ollama_error_invalid_response() {
        let error = OllamaError::InvalidResponse("invalid json".to_string());
        assert!(error.to_string().contains("Invalid response"));
    }

    #[test]
    fn test_ollama_error_ollama_error() {
        let error = OllamaError::OllamaError("model not found".to_string());
        assert!(error.to_string().contains("Ollama error"));
    }

    #[test]
    fn test_ollama_chat_response_with_tool_calls() {
        let json = r#"{
            "message": {
                "role": "assistant",
                "content": "Let me check...",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "NYC"}
                        }
                    }
                ]
            },
            "done": true,
            "prompt_eval_count": 15,
            "eval_count": 8
        }"#;
        let response: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert!(response.message.tool_calls.is_some());
        let calls = response.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, Some("call_1".to_string()));
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_ollama_message_with_null_content() {
        let json = r#"{"role": "assistant", "content": null}"#;
        let message: OllamaMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.content, None);
    }

    #[test]
    fn test_ollama_message_without_content_field() {
        let json = r#"{"role": "assistant"}"#;
        let message: OllamaMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.content, None);
    }

    #[test]
    fn test_ollama_options_serialization() {
        let options = OllamaOptions {
            temperature: Some(0.5),
            num_predict: Some(256),
        };
        let json = serde_json::to_string(&options).unwrap();
        assert!(json.contains(r#""temperature":0.5"#));
        assert!(json.contains(r#""num_predict":256"#));
    }

    #[test]
    fn test_ollama_chat_request_with_tools() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        })];
        let request = OllamaChatRequest {
            model: "llama2".to_string(),
            messages: vec![],
            stream: false,
            options: None,
            tools: Some(tools),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("get_weather"));
    }
}
