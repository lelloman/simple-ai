//! Ollama inference engine implementation.

use async_trait::async_trait;
use futures_util::stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use simple_ai_common::{
    format_sse_chunk, format_sse_done, format_sse_metrics, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, InferenceMetrics, ToolCall,
    ToolFunction,
};
use simple_server::web::body::Bytes;
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;

use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::error::{Error, Result};

/// Ollama inference engine.
///
/// Communicates with an Ollama server to provide inference capabilities.
pub struct OllamaEngine {
    http_client: Client,
    base_url: String,
    batch_size: u32,
    model_context_cache: RwLock<HashMap<String, Option<u32>>>,
}

impl OllamaEngine {
    pub fn with_batch_size(base_url: &str, batch_size: u32) -> Self {
        Self {
            http_client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            batch_size,
            model_context_cache: RwLock::new(HashMap::new()),
        }
    }

    // Keep the completion warmup for existing chat models. Embedding-only
    // models reject generate before Ollama's empty-prompt lifecycle handling.
    async fn set_model_keep_alive(&self, model_id: &str, keep_alive: &str) -> Result<()> {
        let response = self
            .http_client
            .post(format!("{}/api/generate", self.base_url))
            .json(&OllamaGenerateRequest {
                model: model_id.to_string(),
                prompt: String::new(),
                stream: false,
                keep_alive: Some(keep_alive.to_string()),
            })
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let body = response.text().await.unwrap_or_default();
        let unsupported = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error")?.as_str().map(str::to_owned))
            .is_some_and(|error| error.ends_with("does not support generate"));
        if status != reqwest::StatusCode::BAD_REQUEST || !unsupported {
            return Err(Error::LoadFailed(format!("Ollama generate lifecycle model={model_id} keep_alive={keep_alive}: {status}: {body}")));
        }
        // Empty input loads without computing a vector; keep_alive=0 unloads.
        // Do not cache capability or success: a failed load must be retryable.
        let response = self
            .http_client
            .post(format!("{}/api/embed", self.base_url))
            .json(&serde_json::json!({"model": model_id, "input": [], "keep_alive": keep_alive}))
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::LoadFailed(format!(
                "Ollama embed lifecycle model={model_id} keep_alive={keep_alive}: {status}: {body}"
            )));
        }
        Ok(())
    }

    fn build_chat_request(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
        stream: bool,
        context_length: Option<u32>,
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
                    content: m.content.as_ref().and_then(|content| content.text_only().ok()),
                    tool_calls,
                    tool_call_id: m.tool_call_id.clone(),
                }
            })
            .collect();

        let options = if request.temperature.is_some()
            || request.max_tokens.is_some()
            || context_length.is_some()
        {
            Some(OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens,
                num_ctx: context_length,
            })
        } else {
            None
        };

        OllamaChatRequest {
            model: model_id.to_string(),
            messages,
            stream,
            options,
            tools: request.tools.clone(),
        }
    }

    async fn cached_model_context_length(&self, model_id: &str) -> Option<u32> {
        if let Some(context_length) = self.model_context_cache.read().await.get(model_id) {
            return *context_length;
        }

        let context_length = match self.fetch_model_context_length(model_id).await {
            Ok(value) => value,
            Err(e) => {
                tracing::debug!(
                    "Could not fetch Ollama context length for {}: {}",
                    model_id,
                    e
                );
                None
            }
        };

        self.model_context_cache
            .write()
            .await
            .insert(model_id.to_string(), context_length);

        context_length
    }

    async fn fetch_model_context_length(&self, model_id: &str) -> Result<Option<u32>> {
        let url = format!("{}/api/show", self.base_url);
        let request = OllamaShowRequest {
            model: model_id.to_string(),
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let show: OllamaShowResponse = response
            .json()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        Ok(extract_ollama_context_length(&show.model_info))
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
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
}

#[derive(Debug, Serialize)]
struct OllamaShowRequest {
    model: String,
}

#[derive(Debug, Deserialize)]
struct OllamaShowResponse {
    #[serde(default)]
    model_info: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
    #[serde(default)]
    total_duration: Option<u64>,
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
    #[serde(default)]
    digest: Option<String>,
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

        let mut models = Vec::new();
        for m in tags.models {
            let parameter_count = m.details.as_ref().and_then(|d| {
                d.parameter_size
                    .as_ref()
                    .and_then(|s| parse_parameter_size(s))
            });
            let quantization = m
                .details
                .as_ref()
                .and_then(|d| d.quantization_level.clone());

            let context_length = self.cached_model_context_length(&m.name).await;

            models.push(ModelInfo {
                revision: m.digest,
                id: m.name.clone(),
                name: m.name,
                size_bytes: m.size,
                parameter_count,
                context_length,
                quantization,
                modified_at: m.modified_at,
                reasoning: None,
            });
        }

        Ok(models)
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        let models = self.list_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }

    async fn load_model(&self, model_id: &str) -> Result<()> {
        self.set_model_keep_alive(model_id, "10m").await?;
        tracing::info!("Model {} loaded/warmed", model_id);
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        self.set_model_keep_alive(model_id, "0").await?;
        tracing::info!("Model {} unloaded", model_id);
        Ok(())
    }

    async fn chat_completion(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        if request.has_images() {
            return Err(Error::NotSupported(
                "image content is not supported by the Ollama adapter".to_string(),
            ));
        }
        let context_length = self.cached_model_context_length(model_id).await;
        let ollama_request = self.build_chat_request(model_id, request, false, context_length);

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
            content: ollama_response.message.content.map(Into::into),
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

        let prompt_tokens = ollama_response.prompt_eval_count;
        let completion_tokens = ollama_response.eval_count;
        if let (Some(prompt), Some(completion)) = (
            ollama_response.prompt_eval_count,
            ollama_response.eval_count,
        ) {
            response = response.with_usage(prompt, completion);
        }

        response = response.with_inference_metrics(
            InferenceMetrics {
                resolved_model: Some(model_id.to_string()),
                engine_type: Some(self.engine_type().to_string()),
                context_window: context_length,
                prompt_tokens,
                completion_tokens,
                prompt_eval_ms: nanos_to_ms(ollama_response.prompt_eval_duration),
                completion_eval_ms: nanos_to_ms(ollama_response.eval_duration),
                total_inference_ms: nanos_to_ms(ollama_response.total_duration),
                ..Default::default()
            }
            .with_computed_rates(),
        );

        Ok(response)
    }

    async fn chat_completion_stream(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        if request.has_images() {
            return Err(Error::NotSupported(
                "image content is not supported by the Ollama adapter".to_string(),
            ));
        }
        let context_length = self.cached_model_context_length(model_id).await;
        let ollama_request = self.build_chat_request(model_id, request, true, context_length);
        let url = format!("{}/api/chat", self.base_url);

        tracing::debug!(
            "Sending streaming chat request to Ollama: {} model={}",
            url,
            model_id
        );

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

        struct StreamState {
            response: reqwest::Response,
            buffer: String,
            pending: VecDeque<Bytes>,
            sent_role: bool,
            emitted_done: bool,
            chunk_id: String,
            created: i64,
            model: String,
            context_length: Option<u32>,
        }

        let state = StreamState {
            response,
            buffer: String::new(),
            pending: VecDeque::new(),
            sent_role: false,
            emitted_done: false,
            chunk_id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            created: chrono::Utc::now().timestamp(),
            model: model_id.to_string(),
            context_length,
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
                    .map_err(|e| Error::Communication(e.to_string()))?
                else {
                    if !state.emitted_done {
                        state.emitted_done = true;
                        return Ok(Some((Bytes::from(format_sse_done()), state)));
                    }
                    return Ok(None);
                };

                let text = std::str::from_utf8(&chunk)
                    .map_err(|e| Error::InferenceFailed(e.to_string()))?;
                state.buffer.push_str(text);

                while let Some(newline_idx) = state.buffer.find('\n') {
                    let line = state.buffer[..newline_idx].trim().to_string();
                    state.buffer.drain(..=newline_idx);

                    if line.is_empty() {
                        continue;
                    }

                    let parsed: OllamaChatResponse = serde_json::from_str(&line)
                        .map_err(|e| Error::InferenceFailed(e.to_string()))?;

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
                            content: parsed.message.content.clone().map(Into::into),
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
                            .map_err(|e| Error::InferenceFailed(e.to_string()))?;
                        state.pending.push_back(Bytes::from(payload));
                        state.sent_role = true;
                    }

                    if parsed.done {
                        let metrics = InferenceMetrics {
                            resolved_model: Some(state.model.clone()),
                            engine_type: Some("ollama".to_string()),
                            context_window: state.context_length,
                            prompt_tokens: parsed.prompt_eval_count,
                            completion_tokens: parsed.eval_count,
                            prompt_eval_ms: nanos_to_ms(parsed.prompt_eval_duration),
                            completion_eval_ms: nanos_to_ms(parsed.eval_duration),
                            total_inference_ms: nanos_to_ms(parsed.total_duration),
                            ..Default::default()
                        }
                        .with_computed_rates();
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
                            .map_err(|e| Error::InferenceFailed(e.to_string()))?;
                        state.pending.push_back(Bytes::from(payload));
                        let payload = format_sse_metrics(&metrics)
                            .map_err(|e| Error::InferenceFailed(e.to_string()))?;
                        state.pending.push_back(Bytes::from(payload));
                        state.pending.push_back(Bytes::from(format_sse_done()));
                        state.emitted_done = true;
                    }
                }
            }
        });

        Ok(Box::pin(stream))
    }

    async fn embed(&self, model_id: &str, input: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.base_url);

        let request = OllamaEmbedRequest {
            model: model_id.to_string(),
            input: input.to_vec(),
        };

        tracing::debug!(
            "Sending embed request to Ollama: {} model={}",
            url,
            model_id
        );

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::InferenceFailed(format!("{}: {}", status, body)));
        }

        let embed_response: OllamaEmbedResponse = response
            .json()
            .await
            .map_err(|e| Error::InferenceFailed(e.to_string()))?;

        Ok(embed_response.embeddings)
    }
}

fn nanos_to_ms(value: Option<u64>) -> Option<u64> {
    value.map(|nanos| nanos / 1_000_000)
}

/// Ollama embed request (POST /api/embed).
#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
}

/// Ollama embed response.
#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
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

    num_str
        .parse::<f64>()
        .ok()
        .map(|n| (n * multiplier as f64) as u64)
}

fn extract_ollama_context_length(model_info: &HashMap<String, serde_json::Value>) -> Option<u32> {
    model_info.iter().find_map(|(key, value)| {
        if key == "context_length" || key.ends_with(".context_length") {
            value.as_u64().and_then(|v| u32::try_from(v).ok())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn embedding_only_registry_acquisition_retry_and_unload() {
        use crate::engine::registry::EngineRegistry;
        use std::sync::Arc;
        for model in ["qwen3-embedding:0.6b", "bge-m3:567m"] {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api/tags"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "models": [{"name": model}]
                })))
                .mount(&server)
                .await;
            Mock::given(path("/api/generate"))
                .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                    "error": format!("{model} does not support generate")
                })))
                .expect(4)
                .mount(&server)
                .await;
            let engine = Arc::new(OllamaEngine::with_batch_size(&server.uri(), 8));
            let registry = EngineRegistry::new();
            registry.register(engine.clone()).await;
            registry
                .set_engine_resources(HashMap::from([("ollama".into(), "gpu:0".into())]))
                .await;
            let warmup = serde_json::json!({"model": model, "input": [], "keep_alive": "10m"});
            let failed = Mock::given(path("/api/embed"))
                .and(body_json(warmup.clone()))
                .respond_with(ResponseTemplate::new(500).set_body_string("out of memory"))
                .expect(1)
                .mount_as_scoped(&server)
                .await;
            let error = registry
                .acquire_model(model)
                .await
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains(model) && error.contains("out of memory"));
            drop(failed);
            Mock::given(path("/api/embed"))
                .and(body_json(warmup))
                .respond_with(
                    ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings": []})),
                )
                .expect(2)
                .mount(&server)
                .await;
            Mock::given(path("/api/embed"))
                .and(body_json(
                    serde_json::json!({"model": model, "input": ["profile"]}),
                ))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({"embeddings": [vec![0.5; 1024]]})),
                )
                .expect(1)
                .mount(&server)
                .await;
            let lease = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                registry.acquire_model(model),
            )
            .await
            .unwrap()
            .unwrap();
            let vectors = lease
                .engine
                .embed(&lease.engine_model, &["profile".into()])
                .await
                .unwrap();
            assert_eq!(vectors, vec![vec![0.5; 1024]]);
            drop(lease);
            // Reacquisition also runs warmup under the shared resource lease.
            drop(registry.acquire_model(model).await.unwrap());
            Mock::given(path("/api/embed"))
                .and(body_json(
                    serde_json::json!({"model": model, "input": [], "keep_alive": "0"}),
                ))
                .respond_with(ResponseTemplate::new(200))
                .expect(1)
                .mount(&server)
                .await;
            engine.unload_model(model).await.unwrap();
        }
    }

    #[tokio::test]
    async fn chat_lifecycle_retains_generate_and_other_errors_do_not_fallback() {
        let server = MockServer::start().await;
        let engine = OllamaEngine::with_batch_size(&server.uri(), 1);
        Mock::given(path("/api/generate"))
            .and(body_json(serde_json::json!({
                "model": "chat", "prompt": "", "stream": false, "keep_alive": "10m"
            })))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(path("/api/generate"))
            .and(body_json(serde_json::json!({
                "model": "chat", "prompt": "", "stream": false, "keep_alive": "0"
            })))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(path("/api/generate"))
            .and(body_json(serde_json::json!({
                "model": "missing", "prompt": "", "stream": false, "keep_alive": "10m"
            })))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(path("/api/embed"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;
        engine.load_model("chat").await.unwrap();
        engine.unload_model("chat").await.unwrap();
        assert!(engine.load_model("missing").await.is_err());
    }

    #[test]
    fn test_parse_parameter_size() {
        assert_eq!(parse_parameter_size("7B"), Some(7_000_000_000));
        assert_eq!(parse_parameter_size("70B"), Some(70_000_000_000));
        assert_eq!(parse_parameter_size("1.5B"), Some(1_500_000_000));
        assert_eq!(parse_parameter_size("500M"), Some(500_000_000));
        assert_eq!(parse_parameter_size("invalid"), None);
    }

    #[test]
    fn test_extract_ollama_context_length() {
        let mut model_info = HashMap::new();
        model_info.insert(
            "llama.context_length".to_string(),
            serde_json::Value::from(131072),
        );

        assert_eq!(extract_ollama_context_length(&model_info), Some(131072));
    }

    #[test]
    fn test_ollama_engine_url_normalization() {
        let engine = OllamaEngine::with_batch_size("http://localhost:11434/", 1);
        assert_eq!(engine.base_url, "http://localhost:11434");
    }
}
