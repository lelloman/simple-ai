//! Ollama inference engine implementation.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use simple_ai_common::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolCall, ToolFunction,
};

use super::{EngineHealth, InferenceEngine, ModelInfo};
use crate::error::{Error, Result};

/// Ollama inference engine.
///
/// Communicates with an Ollama server to provide inference capabilities.
pub struct OllamaEngine {
    http_client: Client,
    base_url: String,
    batch_size: u32,
}

impl OllamaEngine {
    pub fn new(base_url: &str) -> Self {
        Self::with_batch_size(base_url, 1)
    }

    pub fn with_batch_size(base_url: &str, batch_size: u32) -> Self {
        Self {
            http_client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            batch_size,
        }
    }
}

// ============================================================================
// Ollama API types
// ============================================================================

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

/// Response from /api/tags endpoint.
#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelInfo {
    name: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    modified_at: Option<String>,
    #[serde(default)]
    details: Option<OllamaModelDetails>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelDetails {
    #[serde(default)]
    parameter_size: Option<String>,
    #[serde(default)]
    quantization_level: Option<String>,
}

/// Request to generate with keep_alive for model management (Phase 2).
#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

// ============================================================================
// InferenceEngine implementation
// ============================================================================

#[async_trait]
impl InferenceEngine for OllamaEngine {
    fn engine_type(&self) -> &'static str {
        "ollama"
    }

    fn batch_size(&self) -> u32 {
        self.batch_size
    }

    async fn health_check(&self) -> Result<EngineHealth> {
        let url = format!("{}/api/tags", self.base_url);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        if response.status().is_success() {
            let tags: OllamaTagsResponse = response
                .json()
                .await
                .map_err(|e| Error::Communication(e.to_string()))?;

            Ok(EngineHealth {
                is_healthy: true,
                version: None,
                models_loaded: tags.models.iter().map(|m| m.name.clone()).collect(),
            })
        } else {
            Err(Error::EngineNotAvailable(format!(
                "Ollama returned {}",
                response.status()
            )))
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        if !response.status().is_success() {
            return Err(Error::Communication(format!(
                "Ollama returned {}",
                response.status()
            )));
        }

        let tags: OllamaTagsResponse = response
            .json()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        let models = tags
            .models
            .into_iter()
            .map(|m| {
                let parameter_count = m.details.as_ref().and_then(|d| {
                    d.parameter_size.as_ref().and_then(|s| parse_parameter_size(s))
                });
                let quantization = m
                    .details
                    .as_ref()
                    .and_then(|d| d.quantization_level.clone());

                ModelInfo {
                    id: m.name.clone(),
                    name: m.name,
                    size_bytes: m.size,
                    parameter_count,
                    context_length: None, // Ollama doesn't expose this in /api/tags
                    quantization,
                    modified_at: m.modified_at,
                }
            })
            .collect();

        Ok(models)
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        let models = self.list_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }

    async fn load_model(&self, model_id: &str) -> Result<()> {
        // Ollama loads models lazily, but we can "warm" them by sending
        // a minimal request with keep_alive to ensure they stay loaded.
        let url = format!("{}/api/generate", self.base_url);

        let request = OllamaGenerateRequest {
            model: model_id.to_string(),
            prompt: "".to_string(),
            stream: false,
            keep_alive: Some("10m".to_string()),
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::LoadFailed(format!(
                "Failed to load model {}: {}",
                model_id, body
            )));
        }

        tracing::info!("Model {} loaded/warmed", model_id);
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        // Ollama unloads models by setting keep_alive to 0
        let url = format!("{}/api/generate", self.base_url);

        let request = OllamaGenerateRequest {
            model: model_id.to_string(),
            prompt: "".to_string(),
            stream: false,
            keep_alive: Some("0".to_string()),
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Communication(format!(
                "Failed to unload model {}: {}",
                model_id, body
            )));
        }

        tracing::info!("Model {} unloaded", model_id);
        Ok(())
    }

    async fn chat_completion(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        // Convert messages to Ollama format
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
                                    .unwrap_or_else(|e| {
                                        tracing::warn!(
                                            "Failed to parse tool arguments for '{}': {}. Input: {}",
                                            tc.function.name,
                                            e,
                                            tc.function.arguments
                                        );
                                        serde_json::Value::Object(serde_json::Map::new())
                                    }),
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

        let ollama_request = OllamaChatRequest {
            model: model_id.to_string(),
            messages,
            stream: false,
            options,
            tools: request.tools.clone(),
        };

        let url = format!("{}/api/chat", self.base_url);

        tracing::debug!("Sending chat request to Ollama: {} model={}", url, model_id);

        let response = self
            .http_client
            .post(&url)
            .json(&ollama_request)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::InferenceFailed(format!("{}: {}", status, body)));
        }

        let ollama_response: OllamaChatResponse = response
            .json()
            .await
            .map_err(|e| Error::InferenceFailed(e.to_string()))?;

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

        let message = ChatMessage {
            role: ollama_response.message.role,
            content: ollama_response.message.content,
            tool_calls,
            tool_call_id: None,
        };

        let finish_reason = if ollama_response.done {
            if message.tool_calls.is_some() {
                Some("tool_calls".to_string())
            } else {
                Some("stop".to_string())
            }
        } else {
            None
        };

        let mut response =
            ChatCompletionResponse::new(model_id.to_string(), message, finish_reason);

        if let (Some(prompt), Some(completion)) =
            (ollama_response.prompt_eval_count, ollama_response.eval_count)
        {
            response = response.with_usage(prompt, completion);
        }

        Ok(response)
    }
}

/// Parse parameter size strings like "7B", "70B", "1.5B" into actual counts.
fn parse_parameter_size(s: &str) -> Option<u64> {
    let s = s.trim().to_uppercase();
    let (num_str, multiplier) = if s.ends_with('B') {
        (&s[..s.len() - 1], 1_000_000_000u64)
    } else if s.ends_with('M') {
        (&s[..s.len() - 1], 1_000_000u64)
    } else {
        return None;
    };

    num_str.parse::<f64>().ok().map(|n| (n * multiplier as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_parameter_size() {
        assert_eq!(parse_parameter_size("7B"), Some(7_000_000_000));
        assert_eq!(parse_parameter_size("70B"), Some(70_000_000_000));
        assert_eq!(parse_parameter_size("1.5B"), Some(1_500_000_000));
        assert_eq!(parse_parameter_size("500M"), Some(500_000_000));
        assert_eq!(parse_parameter_size("invalid"), None);
    }

    #[test]
    fn test_ollama_engine_url_normalization() {
        let engine = OllamaEngine::new("http://localhost:11434/");
        assert_eq!(engine.base_url, "http://localhost:11434");
    }
}
