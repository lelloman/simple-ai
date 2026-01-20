//! Integration tests for the inference-runner HTTP API.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use tower::ServiceExt;

// Note: These tests require the inference-runner to expose its internals for testing.
// For now, we test what we can without a running Ollama instance.

#[tokio::test]
async fn test_health_endpoint() {
    // Build a minimal router with just the health endpoint
    let app = Router::new()
        .route("/health", axum::routing::get(health_handler));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

async fn health_handler() -> (StatusCode, axum::Json<serde_json::Value>) {
    (StatusCode::OK, axum::Json(serde_json::json!({"status": "ok"})))
}

#[test]
fn test_capability_serialization_roundtrip() {
    use simple_ai_common::Capability;

    for cap in Capability::ALL {
        let json = serde_json::to_string(&cap).unwrap();
        let parsed: Capability = serde_json::from_str(&json).unwrap();
        assert_eq!(cap, parsed);
    }
}

#[test]
fn test_chat_request_minimal() {
    use simple_ai_common::ChatCompletionRequest;

    let json = r#"{
        "model": "llama3.2:3b",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }"#;

    let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.model, Some("llama3.2:3b".to_string()));
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].role, "user");
}

#[test]
fn test_chat_request_with_tools() {
    use simple_ai_common::ChatCompletionRequest;

    let json = r#"{
        "model": "llama3.2:3b",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {}
                }
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }"#;

    let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(request.tools.is_some());
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.max_tokens, Some(100));
}
