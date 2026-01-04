use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::models::chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolCall, ToolFunction};

/// Client for communicating with Ollama API.
pub struct OllamaClient {
    http_client: Client,
    base_url: String,
    #[allow(dead_code)]
    default_model: String,
}

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

#[derive(Debug, Serialize)]
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
}

impl OllamaClient {
    pub fn new(base_url: &str, default_model: &str) -> Self {
        Self {
            http_client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            default_model: default_model.to_string(),
        }
    }

    /// Send a chat request to Ollama and translate the response to OpenAI format.
    pub async fn chat(
        &self,
        request: &ChatCompletionRequest,
        model: &str,
    ) -> Result<ChatCompletionResponse, OllamaError> {
        // Convert messages to Ollama format
        let messages: Vec<OllamaMessage> = request.messages.iter()
            .map(|m| {
                // Convert tool_calls from OpenAI format to Ollama format
                let tool_calls = m.tool_calls.as_ref().map(|calls| {
                    calls.iter().map(|tc| OllamaToolCall {
                        id: Some(tc.id.clone()),
                        function: OllamaToolFunction {
                            name: tc.function.name.clone(),
                            arguments: serde_json::from_str(&tc.function.arguments)
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                        },
                    }).collect()
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

        let ollama_request = OllamaChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
            options,
            tools: request.tools.clone(),
        };

        let url = format!("{}/api/chat", self.base_url);

        tracing::debug!("Sending request to Ollama: {}", url);

        let response = self.http_client
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
            calls.into_iter().enumerate().map(|(i, tc)| {
                ToolCall {
                    id: tc.id.unwrap_or_else(|| format!("call_{}", i)),
                    call_type: "function".to_string(),
                    function: ToolFunction {
                        name: tc.function.name,
                        arguments: tc.function.arguments.to_string(),
                    },
                }
            }).collect()
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

        let mut response = ChatCompletionResponse::new(
            model.to_string(),
            message,
            finish_reason,
        );

        // Add usage info if available
        if let (Some(prompt), Some(completion)) =
            (ollama_response.prompt_eval_count, ollama_response.eval_count)
        {
            response = response.with_usage(prompt, completion);
        }

        Ok(response)
    }
}
