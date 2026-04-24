use axum::body::{to_bytes, Body, Bytes};
use http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use simple_ai_backend::audit::AuditLogger;
use simple_ai_backend::auth::JwksClient;
use simple_ai_backend::llm::OllamaClient;
use simple_ai_backend::{
    routes, AppState, Config, InferenceRouter, RequestScheduler, ResponseContent,
    ResponseCreateRequest, ResponseInput, ResponseInputItem, ResponseTypedInputItem,
    RouterTelemetry, RunnerRegistry, WakeService,
};
use std::sync::Arc;
use tower::ServiceExt;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn create_test_state(
    ollama_base_url: &str,
) -> Result<Arc<AppState>, Box<dyn std::error::Error>> {
    let config = Config {
        host: "0.0.0.0".to_string(),
        port: 8080,
        ollama: simple_ai_backend::config::OllamaConfig {
            base_url: ollama_base_url.to_string(),
            model: "llama3".to_string(),
        },
        oidc: simple_ai_backend::config::OidcConfig {
            issuer: "https://example.com".to_string(),
            audience: "".to_string(),
            role_claim_path: "roles".to_string(),
            admin_role: "admin".to_string(),
            admin_users: vec![],
        },
        database: simple_ai_backend::config::DatabaseConfig {
            url: ":memory:".to_string(),
        },
        logging: simple_ai_backend::config::LoggingConfig {
            level: "info".to_string(),
        },
        cors: simple_ai_backend::config::CorsConfig {
            origins: "*".to_string(),
        },
        language: simple_ai_backend::config::LanguageConfig {
            model_path: "models/lid.176.bin".to_string(),
        },
        gateway: simple_ai_backend::config::GatewayConfig::default(),
        wol: simple_ai_backend::config::WolConfig::default(),
        models: simple_ai_backend::config::ModelsConfig::default(),
        routing: simple_ai_backend::config::RoutingConfig::default(),
    };

    let mock_server = MockServer::start().await;

    #[derive(Deserialize, Serialize)]
    struct OidcConfig {
        jwks_uri: String,
    }

    Mock::given(method("GET"))
        .and(path("/.well-known/openid-configuration"))
        .respond_with(ResponseTemplate::new(200).set_body_json(OidcConfig {
            jwks_uri: format!("{}/.well-known/jwks.json", mock_server.uri()),
        }))
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path("/.well-known/jwks.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "keys": [{
                "kid": "test-key",
                "kty": "RSA",
                "alg": "RS256",
                "n": "test",
                "e": "AQAB"
            }]
        })))
        .mount(&mock_server)
        .await;

    let mock_oidc_config = simple_ai_backend::config::OidcConfig {
        issuer: format!("{}/", mock_server.uri()),
        audience: "test".to_string(),
        role_claim_path: "roles".to_string(),
        admin_role: "admin".to_string(),
        admin_users: vec![],
    };

    let jwks_client = JwksClient::new(&mock_oidc_config).await?;
    let ollama_client = OllamaClient::new(&config.ollama.base_url, &config.ollama.model);
    let audit_logger = Arc::new(AuditLogger::new(&config.database.url)?);
    let runner_registry = Arc::new(RunnerRegistry::new());
    let inference_router = Arc::new(InferenceRouter::new(
        runner_registry.clone(),
        config.models.clone(),
        config.routing.clone(),
        audit_logger.clone(),
    ));
    let wol_config = config.wol.clone();
    let wake_service = Arc::new(WakeService::new(
        runner_registry.clone(),
        audit_logger.clone(),
        config.gateway.clone(),
        config.wol.clone(),
        config.models.clone(),
        config.routing.clone(),
    ));
    let router_telemetry = Arc::new(RouterTelemetry::new());
    let request_scheduler = Arc::new(RequestScheduler::new(
        inference_router.clone(),
        runner_registry.clone(),
        wake_service.clone(),
        router_telemetry.clone(),
        None,
        config.routing.clone(),
    ));
    let (request_events_tx, _) = tokio::sync::broadcast::channel(64);

    Ok(Arc::new(AppState {
        config,
        jwks_client,
        ollama_client,
        audit_logger,
        lang_detector: tokio::sync::Mutex::new(fasttext::FastText::default()),
        runner_registry,
        inference_router,
        request_scheduler,
        wol_config,
        wake_service,
        request_events: request_events_tx,
        router_telemetry,
        batch_queue: None,
        batch_dispatcher: None,
        circuit_breaker: std::sync::Arc::new(simple_ai_backend::CircuitBreaker::new(0, 30)),
    }))
}

async fn create_api_key(state: &Arc<AppState>) -> String {
    let user = state
        .audit_logger
        .find_or_create_user("user-1", Some("user@example.com"))
        .unwrap();
    let (_, secret) = state
        .audit_logger
        .create_api_key(&user.id, "Integration Test Key")
        .unwrap();
    secret
}

async fn send_request(
    app: &axum::Router,
    method: http::Method,
    uri: &str,
    api_key: Option<&str>,
    body: Option<Bytes>,
) -> http::Response<Body> {
    let mut req_builder = http::Request::builder().method(method).uri(uri);

    if body.is_some() {
        req_builder = req_builder.header("Content-Type", "application/json");
    }
    if let Some(api_key) = api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
    }

    let req = req_builder
        .body(if let Some(b) = body {
            Body::from(b)
        } else {
            Body::empty()
        })
        .unwrap();

    app.clone().oneshot(req).await.unwrap()
}

#[tokio::test]
async fn test_responses_non_streaming_success_maps_text_output() {
    let ollama = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({
            "model": "class:fast",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": {
                "role": "assistant",
                "content": "Hi there"
            },
            "done": true,
            "prompt_eval_count": 10,
            "eval_count": 2
        })))
        .mount(&ollama)
        .await;

    let state = create_test_state(&ollama.uri()).await.unwrap();
    let api_key = create_api_key(&state).await;
    let app = axum::Router::new().nest("/v1", routes::responses::router(state));

    let request = ResponseCreateRequest {
        model: "class:fast".to_string(),
        input: ResponseInput::Text("Hello".to_string()),
        tools: None,
        temperature: None,
        max_output_tokens: None,
        stream: Some(false),
    };

    let response = send_request(
        &app,
        http::Method::POST,
        "/v1/responses",
        Some(&api_key),
        Some(Bytes::from(serde_json::to_vec(&request).unwrap())),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "response");
    assert_eq!(json["status"], "completed");
    assert_eq!(json["model"], "class:fast");
    assert_eq!(json["output_text"], "Hi there");
    assert_eq!(json["output"][0]["type"], "message");
    assert_eq!(json["usage"]["input_tokens"], 10);
    assert_eq!(json["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn test_responses_non_streaming_success_maps_tool_calls() {
    let ollama = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": {
                "role": "assistant",
                "content": "Thinking...",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": {"city": "Rome"}
                    }
                }]
            },
            "done": true,
            "prompt_eval_count": 10,
            "eval_count": 5
        })))
        .mount(&ollama)
        .await;

    let state = create_test_state(&ollama.uri()).await.unwrap();
    let api_key = create_api_key(&state).await;
    let app = axum::Router::new().nest("/v1", routes::responses::router(state));

    let request = ResponseCreateRequest {
        model: "class:fast".to_string(),
        input: ResponseInput::Text("What's the weather?".to_string()),
        tools: Some(vec![json!({
            "type":"function",
            "function":{"name":"lookup_weather","parameters":{"type":"object"}}
        })]),
        temperature: None,
        max_output_tokens: None,
        stream: Some(false),
    };

    let response = send_request(
        &app,
        http::Method::POST,
        "/v1/responses",
        Some(&api_key),
        Some(Bytes::from(serde_json::to_vec(&request).unwrap())),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["output"][0]["type"], "message");
    assert_eq!(json["output"][1]["type"], "function_call");
    assert_eq!(json["output"][1]["call_id"], "call_1");
    assert_eq!(json["output"][1]["name"], "lookup_weather");
}

#[tokio::test]
async fn test_responses_accepts_function_call_output_input() {
    let ollama = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_json(json!({
            "model": "class:fast",
            "messages": [
                {"role": "user", "content": "Use the tool"},
                {"role": "tool", "content": "{\"temp\":21}", "tool_call_id": "call_1"}
            ],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": {"role": "assistant", "content": "It is 21C"},
            "done": true,
            "prompt_eval_count": 6,
            "eval_count": 3
        })))
        .mount(&ollama)
        .await;

    let state = create_test_state(&ollama.uri()).await.unwrap();
    let api_key = create_api_key(&state).await;
    let app = axum::Router::new().nest("/v1", routes::responses::router(state));

    let request = ResponseCreateRequest {
        model: "class:fast".to_string(),
        input: ResponseInput::Items(vec![
            ResponseInputItem::Typed(ResponseTypedInputItem::Message {
                role: "user".to_string(),
                content: ResponseContent::Text("Use the tool".to_string()),
                tool_call_id: None,
            }),
            ResponseInputItem::Typed(ResponseTypedInputItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: "{\"temp\":21}".to_string(),
            }),
        ]),
        tools: None,
        temperature: None,
        max_output_tokens: None,
        stream: Some(false),
    };

    let response = send_request(
        &app,
        http::Method::POST,
        "/v1/responses",
        Some(&api_key),
        Some(Bytes::from(serde_json::to_vec(&request).unwrap())),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["output_text"], "It is 21C");
}

#[tokio::test]
async fn test_responses_streaming_emits_responses_events_and_filters_internal_metrics() {
    let ollama = MockServer::start().await;
    let stream_body = concat!(
        "{\"message\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"done\":false}\n",
        "{\"message\":{\"role\":\"assistant\",\"content\":\"lo\"},\"done\":false}\n",
        "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,",
        "\"prompt_eval_count\":10,\"eval_count\":2,\"prompt_eval_duration\":1000000,",
        "\"eval_duration\":2000000,\"total_duration\":3000000}\n"
    );
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/x-ndjson")
                .set_body_raw(stream_body, "application/x-ndjson"),
        )
        .mount(&ollama)
        .await;

    let state = create_test_state(&ollama.uri()).await.unwrap();
    let api_key = create_api_key(&state).await;
    let app = axum::Router::new().nest("/v1", routes::responses::router(state));

    let request = ResponseCreateRequest {
        model: "class:fast".to_string(),
        input: ResponseInput::Text("Hello".to_string()),
        tools: None,
        temperature: None,
        max_output_tokens: None,
        stream: Some(true),
    };

    let response = send_request(
        &app,
        http::Method::POST,
        "/v1/responses",
        Some(&api_key),
        Some(Bytes::from(serde_json::to_vec(&request).unwrap())),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("event: response.created"));
    assert!(text.contains("event: response.output_item.added"));
    assert!(text.contains("event: response.output_text.delta"));
    assert!(text.contains("event: response.output_item.done"));
    assert!(text.contains("event: response.completed"));
    assert!(text.contains("\"delta\":\"Hel\""));
    assert!(text.contains("\"delta\":\"lo\""));
    assert!(text.contains("\"output_text\":\"Hello\""));
    assert!(text.contains("data: [DONE]"));
    assert!(!text.contains("simple_ai_metrics"));
}

#[tokio::test]
async fn test_responses_rejects_specific_model_for_api_key_user_without_roles() {
    let ollama = MockServer::start().await;
    let state = create_test_state(&ollama.uri()).await.unwrap();
    let api_key = create_api_key(&state).await;
    let app = axum::Router::new().nest("/v1", routes::responses::router(state));

    let request = ResponseCreateRequest {
        model: "llama3".to_string(),
        input: ResponseInput::Text("Hello".to_string()),
        tools: None,
        temperature: None,
        max_output_tokens: None,
        stream: Some(false),
    };

    let response = send_request(
        &app,
        http::Method::POST,
        "/v1/responses",
        Some(&api_key),
        Some(Bytes::from(serde_json::to_vec(&request).unwrap())),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("Permission denied"));
}

#[tokio::test]
async fn test_responses_surfaces_upstream_error() {
    let ollama = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(500).set_body_string("backend exploded"))
        .mount(&ollama)
        .await;

    let state = create_test_state(&ollama.uri()).await.unwrap();
    let api_key = create_api_key(&state).await;
    let app = axum::Router::new().nest("/v1", routes::responses::router(state));

    let request = ResponseCreateRequest {
        model: "class:fast".to_string(),
        input: ResponseInput::Text("Hello".to_string()),
        tools: None,
        temperature: None,
        max_output_tokens: None,
        stream: Some(false),
    };

    let response = send_request(
        &app,
        http::Method::POST,
        "/v1/responses",
        Some(&api_key),
        Some(Bytes::from(serde_json::to_vec(&request).unwrap())),
    )
    .await;

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("500 Internal Server Error"));
}
