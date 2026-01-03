use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::models::chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage};

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
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
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
    content: String,
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
            .map(|m| OllamaMessage {
                role: m.role.clone(),
                content: m.content.clone().unwrap_or_default(),
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

        // Convert to OpenAI format
        let message = ChatMessage {
            role: ollama_response.message.role,
            content: Some(ollama_response.message.content),
            tool_calls: None,
            tool_call_id: None,
        };

        let finish_reason = if ollama_response.done {
            Some("stop".to_string())
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
