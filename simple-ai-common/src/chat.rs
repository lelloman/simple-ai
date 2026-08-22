//! OpenAI-compatible chat completion types.

use serde::{Deserialize, Serialize};

/// Model-native reasoning depth requested for a completion.
///
/// `none` disables reasoning. The remaining values are interpreted by models
/// whose chat templates support native reasoning effort.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Default,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Default => "default",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Reasoning controls supported by one concrete model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningCapabilities {
    /// Effort values accepted by this model's chat template.
    #[serde(default)]
    pub supported_efforts: Vec<ReasoningEffort>,
    /// Whether this runtime/model combination accepts a hard thinking budget.
    #[serde(default)]
    pub supports_thinking_budget: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_thinking_budget_tokens: Option<i32>,
}

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
    /// Model-native reasoning depth. This is distinct from a token budget.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Per-request hard limit for reasoning tokens. Zero disables thinking;
    /// `-1` means unrestricted for runtimes that support that convention.
    #[serde(default)]
    pub thinking_budget_tokens: Option<i32>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: Option<bool>,
}

impl ChatCompletionRequest {
    pub fn has_images(&self) -> bool {
        self.messages
            .iter()
            .filter_map(|message| message.content.as_ref())
            .any(ChatContent::has_images)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ChatContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl From<String> for ChatContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ChatContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl ChatContent {
    pub fn text_only(&self) -> Result<String, &'static str> {
        match self {
            Self::Text(text) => Ok(text.clone()),
            Self::Parts(parts) => {
                let mut text = String::new();
                for part in parts {
                    match part {
                        ChatContentPart::Text { text: part } => text.push_str(part),
                        ChatContentPart::ImageUrl { .. } => {
                            return Err("image content is not supported by this engine")
                        }
                    }
                }
                Ok(text)
            }
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(_) => None,
        }
    }

    pub fn has_images(&self) -> bool {
        matches!(self, Self::Parts(parts) if parts.iter().any(|part| matches!(part, ChatContentPart::ImageUrl { .. })))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<ChatContent>,
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
    pub delta: ChatCompletionDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Incremental fields emitted by an OpenAI-compatible chat stream.
///
/// Unlike a complete `ChatMessage`, the role is commonly present only in the
/// first event. Some reasoning models emit intermediate tokens through the
/// llama.cpp-compatible `reasoning_content` extension.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl From<ChatMessage> for ChatCompletionDelta {
    fn from(message: ChatMessage) -> Self {
        Self {
            role: Some(message.role),
            content: message
                .content
                .and_then(|content| content.as_text().map(str::to_owned)),
            reasoning_content: None,
            tool_calls: message.tool_calls,
            tool_call_id: message.tool_call_id,
        }
    }
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
                delta: delta.into(),
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
        assert!(req.reasoning_effort.is_none());
        assert!(req.thinking_budget_tokens.is_none());
    }

    #[test]
    fn test_chat_message_accepts_openai_text_content_parts() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello, "},
                {"type": "text", "text": "Qwen!"}
            ]
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        assert_eq!(
            message.content.as_ref().unwrap().text_only().unwrap(),
            "Hello, Qwen!"
        );
    }

    #[test]
    fn test_chat_message_accepts_image_content_parts() {
        let json = r#"{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "example"}}]
        }"#;

        let message = serde_json::from_str::<ChatMessage>(json).unwrap();
        assert!(message.content.as_ref().unwrap().has_images());
        assert!(message.content.as_ref().unwrap().text_only().is_err());
    }

    #[test]
    fn test_chat_completion_request_with_all_fields() {
        let req = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("Hello".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Some(vec![]),
            model: Some("gpt-4".to_string()),
            temperature: Some(0.7),
            max_tokens: Some(100),
            reasoning_effort: Some(ReasoningEffort::High),
            thinking_budget_tokens: Some(2048),
            stream: Some(false),
        };
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.model, Some("gpt-4".to_string()));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(req.thinking_budget_tokens, Some(2048));
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
            content: Some("Hello!".into()),
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
            content: Some("Hello!".into()),
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
            content: Some("Hello!".into()),
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
            content: Some("Streaming...".into()),
            tool_calls: None,
            tool_call_id: None,
        };
        let response = ChatCompletionResponse::new("model".to_string(), message, None);
        assert_eq!(response.choices[0].finish_reason, None);
    }

    #[test]
    fn test_stream_delta_accepts_reasoning_without_repeated_role() {
        let json = r#"{
            "id":"chatcmpl-1",
            "object":"chat.completion.chunk",
            "created":123,
            "model":"qwen3.8-27b",
            "choices":[{
                "index":0,
                "delta":{"reasoning_content":"We need"},
                "finish_reason":null
            }]
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let delta = &chunk.choices[0].delta;

        assert!(delta.role.is_none());
        assert!(delta.content.is_none());
        assert_eq!(delta.reasoning_content.as_deref(), Some("We need"));
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
                content: Some("What is 2+2?".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            model: Some("llama2".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(50),
            reasoning_effort: Some(ReasoningEffort::Medium),
            thinking_budget_tokens: Some(1024),
            tools: None,
            stream: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, original.model);
        assert_eq!(deserialized.temperature, original.temperature);
        assert_eq!(deserialized.max_tokens, original.max_tokens);
        assert_eq!(deserialized.reasoning_effort, original.reasoning_effort);
        assert_eq!(
            deserialized.thinking_budget_tokens,
            original.thinking_budget_tokens
        );
        assert_eq!(deserialized.messages.len(), original.messages.len());
    }

    #[test]
    fn test_chat_message_with_tool_call_id() {
        let msg = ChatMessage {
            role: "tool".to_string(),
            content: Some("Result: 42".into()),
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
