//! OpenAI-compatible chat completion types.

use serde::{Deserialize, Serialize};

/// Inference-only performance metadata used internally by SimpleAI.
///
/// These timings describe model evaluation and intentionally exclude gateway
/// queueing, wake-on-LAN, model load, and model unload time.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct InferenceMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_eval_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_inference_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_per_sec: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_per_sec: Option<f64>,
}

impl InferenceMetrics {
    pub fn tokens_per_second(tokens: Option<u32>, duration_ms: Option<u64>) -> Option<f64> {
        let tokens = tokens?;
        let duration_ms = duration_ms?;
        if duration_ms == 0 {
            return None;
        }
        Some(tokens as f64 * 1000.0 / duration_ms as f64)
    }

    pub fn with_computed_rates(mut self) -> Self {
        self.prompt_tokens_per_sec =
            Self::tokens_per_second(self.prompt_tokens, self.prompt_eval_ms);
        self.completion_tokens_per_sec =
            Self::tokens_per_second(self.completion_tokens, self.completion_eval_ms);
        self
    }
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(
        default,
        rename = "_simple_ai_metrics",
        skip_serializing_if = "Option::is_none"
    )]
    pub inference_metrics: Option<InferenceMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI-compatible streaming chat completion chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

impl ChatCompletionChunk {
    pub fn new(
        id: String,
        created: i64,
        model: String,
        delta: ChatMessage,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created,
            model,
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
        }
    }
}

pub fn format_sse_chunk(chunk: &ChatCompletionChunk) -> Result<String, serde_json::Error> {
    Ok(format!("data: {}\n\n", serde_json::to_string(chunk)?))
}

pub fn format_sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}

pub fn format_sse_metrics(metrics: &InferenceMetrics) -> Result<String, serde_json::Error> {
    Ok(format!(
        "event: simple_ai_metrics\ndata: {}\n\n",
        serde_json::to_string(metrics)?
    ))
}

impl ChatCompletionResponse {
    pub fn new(model: String, message: ChatMessage, finish_reason: Option<String>) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: now,
            model,
            choices: vec![Choice {
                index: 0,
                message,
                finish_reason,
            }],
            usage: None,
            inference_metrics: None,
        }
    }

    pub fn with_usage(mut self, prompt_tokens: u32, completion_tokens: u32) -> Self {
        self.usage = Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        });
        self
    }

    pub fn with_inference_metrics(mut self, metrics: InferenceMetrics) -> Self {
        self.inference_metrics = Some(metrics);
        self
    }

    pub fn strip_internal_metrics(mut self) -> Self {
        self.inference_metrics = None;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_request_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "Hello"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert!(req.tools.is_none());
        assert!(req.model.is_none());
        assert!(req.temperature.is_none());
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn test_chat_completion_request_with_all_fields() {
        let req = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Some(vec![]),
            model: Some("gpt-4".to_string()),
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: Some(false),
        };
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.model, Some("gpt-4".to_string()));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn test_chat_message_default_content() {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
        };
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_chat_completion_response_new() {
        let message = ChatMessage {
            role: "assistant".to_string(),
            content: Some("Hello!".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let response = ChatCompletionResponse::new(
            "test-model".to_string(),
            message,
            Some("stop".to_string()),
        );
        assert!(response.id.starts_with("chatcmpl-"));
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.model, "test-model");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[0].message.role, "assistant");
        assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_chat_completion_response_with_usage() {
        let message = ChatMessage {
            role: "assistant".to_string(),
            content: Some("Hello!".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let response = ChatCompletionResponse::new(
            "test-model".to_string(),
            message,
            Some("stop".to_string()),
        )
        .with_usage(10, 5);

        assert!(response.usage.is_some());
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_inference_metrics_tokens_per_second() {
        assert_eq!(
            InferenceMetrics::tokens_per_second(Some(50), Some(2_000)),
            Some(25.0)
        );
        assert_eq!(InferenceMetrics::tokens_per_second(Some(50), Some(0)), None);
        assert_eq!(InferenceMetrics::tokens_per_second(None, Some(1_000)), None);
    }

    #[test]
    fn test_internal_metrics_can_be_stripped() {
        let message = ChatMessage {
            role: "assistant".to_string(),
            content: Some("Hello!".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let response =
            ChatCompletionResponse::new("model".to_string(), message, Some("stop".to_string()))
                .with_inference_metrics(InferenceMetrics {
                    resolved_model: Some("model".to_string()),
                    engine_type: Some("test".to_string()),
                    context_window: Some(8192),
                    ..Default::default()
                });

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("_simple_ai_metrics"));

        let stripped = response.strip_internal_metrics();
        let json = serde_json::to_string(&stripped).unwrap();
        assert!(!json.contains("_simple_ai_metrics"));
    }

    #[test]
    fn test_usage_total_tokens_calculation() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }

    #[test]
    fn test_finish_reason_none_when_not_done() {
        let message = ChatMessage {
            role: "assistant".to_string(),
            content: Some("Streaming...".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let response = ChatCompletionResponse::new("model".to_string(), message, None);
        assert_eq!(response.choices[0].finish_reason, None);
    }

    #[test]
    fn test_tool_call_serialization() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                arguments: r#"{"location": "NYC"}"#.to_string(),
            },
        };
        let json = serde_json::to_string(&tool_call).unwrap();
        assert!(json.contains(r#""id":"call_123""#));
        assert!(json.contains(r#""type":"function""#));
        assert!(json.contains(r#""name":"get_weather""#));
    }

    #[test]
    fn test_tool_call_with_empty_arguments() {
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: ToolFunction {
                name: "no_args".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let json = serde_json::to_string(&tool_call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.function.name, "no_args");
        assert_eq!(deserialized.function.arguments, "{}");
    }

    #[test]
    fn test_request_serde_roundtrip() {
        let original = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("What is 2+2?".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            model: Some("llama2".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(50),
            tools: None,
            stream: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, original.model);
        assert_eq!(deserialized.temperature, original.temperature);
        assert_eq!(deserialized.max_tokens, original.max_tokens);
        assert_eq!(deserialized.messages.len(), original.messages.len());
    }

    #[test]
    fn test_chat_message_with_tool_call_id() {
        let msg = ChatMessage {
            role: "tool".to_string(),
            content: Some("Result: 42".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
        };
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_chat_message_with_tool_calls() {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: ToolFunction {
                    name: "get_time".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
        };
        assert!(msg.content.is_none());
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    }
}
