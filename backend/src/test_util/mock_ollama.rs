use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct MockOllamaResponse {
    pub message: MockOllamaMessage,
    pub done: bool,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub eval_count: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MockOllamaMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<MockOllamaToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MockOllamaToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub function: MockOllamaToolFunction,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MockOllamaToolFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl MockOllamaResponse {
    pub fn simple_text(content: &str) -> Self {
        Self {
            message: MockOllamaMessage {
                role: "assistant".to_string(),
                content: Some(content.to_string()),
                tool_calls: None,
            },
            done: true,
            prompt_eval_count: Some(10),
            eval_count: Some(content.split_whitespace().count() as u32),
        }
    }

    pub fn with_tool_calls(name: &str, args: HashMap<&str, &str>) -> Self {
        Self {
            message: MockOllamaMessage {
                role: "assistant".to_string(),
                content: Some("Thinking...".to_string()),
                tool_calls: Some(vec![MockOllamaToolCall {
                    id: Some("call_1".to_string()),
                    function: MockOllamaToolFunction {
                        name: name.to_string(),
                        arguments: serde_json::to_value(args).unwrap(),
                    },
                }]),
            },
            done: true,
            prompt_eval_count: Some(10),
            eval_count: Some(5),
        }
    }

    pub fn json() -> serde_json::Value {
        serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "done": true,
            "prompt_eval_count": 10,
            "eval_count": 5
        })
    }

    pub fn error_json(message: &str) -> serde_json::Value {
        serde_json::json!({
            "error": message
        })
    }
}
