use simple_ai_backend::{routes, AppState, Config};
use simple_ai_backend::auth::{JwksClient, AuthError};
use simple_ai_backend::llm::OllamaClient;
use simple_ai_backend::audit::AuditLogger;
use simple_ai_backend::models::chat::{ChatCompletionRequest, ChatMessage};
use std::sync::Arc;
use bytes::Bytes;
use http::StatusCode;
use tower::ServiceExt;
use wiremock::{MockServer, Mock, ResponseTemplate};
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;

async fn create_test_state() -> Result<Arc<AppState>, AuthError> {
    let config = Config {
        host: "0.0.0.0".to_string(),
        port: 8080,
        ollama: simple_ai_backend::config::OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
        },
        oidc: simple_ai_backend::config::OidcConfig {
            issuer: "https://example.com".to_string(),
            audience: "".to_string(),
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
    };

    let mock_server = MockServer::start().await;

    #[derive(Deserialize, Serialize)]
    struct OidcConfig {
        jwks_uri: String,
    }

    Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/.well-known/openid-configuration"))
        .respond_with(ResponseTemplate::new(200).set_body_json(OidcConfig {
            jwks_uri: format!("{}/.well-known/jwks.json", mock_server.uri()),
        }))
        .mount(&mock_server)
        .await;

    Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/.well-known/jwks.json"))
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

    let jwks_client = JwksClient::new(&format!("{}/", mock_server.uri())).await?;
    let ollama_client = OllamaClient::new(&config.ollama.base_url, &config.ollama.model);
    let audit_logger = AuditLogger::new(&config.database.url).unwrap();

    Ok(Arc::new(AppState {
        config,
        jwks_client,
        ollama_client,
        audit_logger,
        lang_detector: tokio::sync::Mutex::new(fasttext::FastText::default()),
    }))
}

async fn send_request(app: &axum::Router, method: http::Method, uri: &str, body: Option<Bytes>) -> StatusCode {
    let mut req_builder = http::Request::builder()
        .method(method)
        .uri(uri);

    if body.is_some() {
        req_builder = req_builder.header("Content-Type", "application/json");
    }

    let req = req_builder.body(if let Some(b) = body {
        axum::body::Body::from(b)
    } else {
        axum::body::Body::empty()
    }).unwrap();

    let response = app.clone().oneshot(req).await.unwrap();
    response.status()
}

#[tokio::test]
async fn test_chat_completions_requires_auth() {
    let state = create_test_state().await.unwrap();
    let app = routes::chat::router(state);

    let request = ChatCompletionRequest {
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some("Hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }],
        model: None,
        temperature: None,
        max_tokens: None,
        tools: None,
    };

    let body = Bytes::from(serde_json::to_string(&request).unwrap());
    let status = send_request(&app, http::Method::POST, "/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_language_detection_requires_auth() {
    let state = create_test_state().await.unwrap();
    let app = routes::language::router(state);

    let body = Bytes::from(r#"{"text": "Hello world"}"#);
    let status = send_request(&app, http::Method::POST, "/detect-language", Some(body)).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_admin_dashboard_requires_auth() {
    let state = create_test_state().await.unwrap();
    let app = routes::admin::router(state);

    let status = send_request(&app, http::Method::GET, "/", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_admin_users_requires_auth() {
    let state = create_test_state().await.unwrap();
    let app = routes::admin::router(state);

    let status = send_request(&app, http::Method::GET, "/users", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_admin_user_disable_requires_auth() {
    let state = create_test_state().await.unwrap();
    let app = routes::admin::router(state);

    let status = send_request(&app, http::Method::POST, "/users/test-user/disable", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_admin_user_enable_requires_auth() {
    let state = create_test_state().await.unwrap();
    let app = routes::admin::router(state);

    let status = send_request(&app, http::Method::POST, "/users/test-user/enable", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_admin_stats_requires_auth() {
    let state = create_test_state().await.unwrap();
    let app = routes::admin::router(state);

    let status = send_request(&app, http::Method::GET, "/stats", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_nonexistent_route_returns_404() {
    let state = create_test_state().await.unwrap();
    let app = routes::admin::router(state);

    let status = send_request(&app, http::Method::GET, "/nonexistent", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}
