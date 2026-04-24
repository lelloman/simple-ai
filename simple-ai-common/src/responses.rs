//! OpenAI-compatible Responses API types.

use serde::{Deserialize, Serialize};

use crate::{ChatCompletionResponse, ChatMessage, ToolCall};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCreateRequest {
    pub model: String,
    pub input: ResponseInput,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default, rename = "max_output_tokens")]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseInputItem {
    Typed(ResponseTypedInputItem),
    Message(ResponseInputMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseTypedInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: ResponseContent,
        #[serde(default)]
        tool_call_id: Option<String>,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseInputMessage {
    pub role: String,
    pub content: ResponseContent,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseContent {
    Text(String),
    Parts(Vec<ResponseContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub status: String,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    pub output_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message(ResponseOutputMessage),
    #[serde(rename = "function_call")]
    FunctionCall(ResponseFunctionCall),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseOutputMessage {
    pub id: String,
    pub role: String,
    pub content: Vec<ResponseOutputContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseOutputContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFunctionCall {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

impl ResponseContent {
    pub fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts.into_iter().filter_map(|part| part.text).collect(),
        }
    }
}

impl ResponseInput {
    pub fn into_chat_messages(self) -> Vec<ChatMessage> {
        match self {
            Self::Text(text) => vec![ChatMessage {
                role: "user".to_string(),
                content: Some(text),
                tool_calls: None,
                tool_call_id: None,
            }],
            Self::Items(items) => items
                .into_iter()
                .map(ResponseInputItem::into_chat_message)
                .collect(),
        }
    }
}

impl ResponseInputItem {
    pub fn into_chat_message(self) -> ChatMessage {
        match self {
            Self::Typed(ResponseTypedInputItem::Message {
                role,
                content,
                tool_call_id,
            }) => ChatMessage {
                role,
                content: Some(content.into_text()),
                tool_calls: None,
                tool_call_id,
            },
            Self::Typed(ResponseTypedInputItem::FunctionCallOutput { call_id, output }) => {
                ChatMessage {
                    role: "tool".to_string(),
                    content: Some(output),
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                }
            }
            Self::Message(message) => ChatMessage {
                role: message.role,
                content: Some(message.content.into_text()),
                tool_calls: None,
                tool_call_id: message.tool_call_id,
            },
        }
    }
}

impl ResponseObject {
    pub fn from_chat(response: ChatCompletionResponse) -> Self {
        let mut output = Vec::new();
        let mut output_text = String::new();

        if let Some(choice) = response.choices.first() {
            if let Some(text) = &choice.message.content {
                output_text = text.clone();
                if !text.is_empty() {
                    output.push(ResponseOutputItem::Message(ResponseOutputMessage {
                        id: format!("msg_{}", uuid::Uuid::new_v4()),
                        role: choice.message.role.clone(),
                        content: vec![ResponseOutputContent {
                            content_type: "output_text".to_string(),
                            text: text.clone(),
                        }],
                    }));
                }
            }

            if let Some(tool_calls) = &choice.message.tool_calls {
                output.extend(
                    tool_calls
                        .iter()
                        .cloned()
                        .map(ResponseOutputItem::from_tool_call),
                );
            }
        }

        Self {
            id: response.id,
            object: "response".to_string(),
            created_at: response.created,
            status: "completed".to_string(),
            model: response.model,
            output,
            output_text,
            usage: response.usage.map(|usage| ResponseUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            }),
        }
    }
}

impl ResponseOutputItem {
    pub fn from_tool_call(tool_call: ToolCall) -> Self {
        Self::FunctionCall(ResponseFunctionCall {
            id: tool_call.id.clone(),
            call_id: tool_call.id,
            name: tool_call.function.name,
            arguments: tool_call.function.arguments,
            status: "completed".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatCompletionResponse, ToolFunction, Usage};

    #[test]
    fn test_response_text_input_maps_to_user_message() {
        let messages = ResponseInput::Text("Hello".to_string()).into_chat_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_function_call_output_maps_to_tool_message() {
        let messages = ResponseInput::Items(vec![ResponseInputItem::Typed(
            ResponseTypedInputItem::FunctionCallOutput {
                call_id: "call_123".to_string(),
                output: "42".to_string(),
            },
        )])
        .into_chat_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "tool");
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(messages[0].content.as_deref(), Some("42"));
    }

    #[test]
    fn test_response_object_from_chat_response() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 123,
            model: "test-model".to_string(),
            choices: vec![crate::Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some("Hi".to_string()),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: ToolFunction {
                            name: "lookup".to_string(),
                            arguments: "{\"q\":\"hi\"}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 2,
                total_tokens: 7,
            }),
            inference_metrics: None,
        };

        let object = ResponseObject::from_chat(response);
        assert_eq!(object.object, "response");
        assert_eq!(object.status, "completed");
        assert_eq!(object.output_text, "Hi");
        assert_eq!(object.output.len(), 2);
        assert_eq!(object.usage.unwrap().input_tokens, 5);
    }
}
