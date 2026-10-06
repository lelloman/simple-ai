use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use simple_ai_backend::{
    audit::AuditLogger,
    auth::{AuthError, JwksClient},
    llm::OllamaClient,
    routes, AppState, Config, InferenceRouter, RequestScheduler, RouterTelemetry, RunnerRegistry,
    WakeService,
};
use simple_ai_common::{CommandResponse, RunnerHealth, RunnerStatus};
use simple_server::web::{
    http::{Request, StatusCode},
    Body,
};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn create_test_state() -> Result<Arc<AppState>, AuthError> {
    let mut config = Config {
        host: "0.0.0.0".to_string(),
        port: 8080,
        ollama: simple_ai_backend::config::OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
        },
        oidc: simple_ai_backend::config::OidcConfig {
            issuer: "https://example.com".to_string(),
            audience: "".to_string(),
            additional_audiences: vec![],
            android_client_id: None,
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
        trusted_proxies: vec![],
    };

    let mock_server = MockServer::start().await;

    #[derive(Deserialize, Serialize)]
    struct OidcConfig {
        jwks_uri: String,
    }

    Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path(
            "/.well-known/openid-configuration",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(OidcConfig {
            jwks_uri: format!("{}/.well-known/jwks.json", mock_server.uri()),
        }))
        .mount(&mock_server)
        .await;

    Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/.well-known/jwks.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(
                serde_json::from_str::<serde_json::Value>(include_str!(
                    "fixtures/test-oidc-jwks.json"
                ))
                .unwrap(),
            ),
        )
        .mount(&mock_server)
        .await;

    // Create OIDC config with mock server as issuer
    let mock_oidc_config = simple_ai_backend::config::OidcConfig {
        issuer: format!("{}/", mock_server.uri()),
        audience: "test".to_string(),
        additional_audiences: vec![],
        android_client_id: None,
        role_claim_path: "roles".to_string(),
        admin_role: "admin".to_string(),
        admin_users: vec![],
    };

    let jwks_client = JwksClient::new(&mock_oidc_config).await?;
    config.oidc = mock_oidc_config;
    let ollama_client = OllamaClient::new(&config.ollama.base_url, &config.ollama.model);
    let audit_logger = Arc::new(AuditLogger::new(&config.database.url).unwrap());
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
        lan_local: Default::default(),
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
        request_cancellations: std::sync::Arc::new(
            simple_ai_backend::RequestCancellationRegistry::new(),
        ),
        router_telemetry,
        batch_queue: None,
        batch_dispatcher: None,
        circuit_breaker: std::sync::Arc::new(simple_ai_backend::CircuitBreaker::new(0, 30)),
    }))
}

fn token(state: &AppState, roles: &[&str], audience: &str) -> String {
    use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some("test-key".into());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    encode(
        &header,
        &json!({"sub":"sse-test-user", "iss":state.config.oidc.issuer,
        "aud":audience, "exp":now+600, "iat":now, "roles":roles}),
        &EncodingKey::from_rsa_pem(include_bytes!("fixtures/test-oidc-private.pem")).unwrap(),
    )
    .unwrap()
}
fn status(health: RunnerHealth) -> RunnerStatus {
    RunnerStatus {
        health,
        capabilities: vec![],
        engines: vec![],
        metrics: None,
        model_aliases: Default::default(),
    }
}
async fn register(state: &AppState, id: &str) {
    state
        .runner_registry
        .register(
            id.into(),
            "雪\nrunner".into(),
            None,
            status(RunnerHealth::Healthy),
            None,
            tokio::sync::mpsc::channel(4).0,
            None,
        )
        .await;
}

#[tokio::test]
async fn admin_sse_auth_and_named_events_over_real_http() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let state = create_test_state().await.unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = simple_server::lifecycle::Shutdown::new();
        let server = tokio::spawn(simple_server::web::serve(
            listener,
            routes::admin::router(state.clone()),
            stop.clone(),
        ));
        let client = reqwest::Client::new();
        for (credential, expected) in [
            ("invalid".into(), StatusCode::UNAUTHORIZED),
            (token(&state, &["user"], "test"), StatusCode::FORBIDDEN),
            (
                token(&state, &["admin"], "wrong-audience"),
                StatusCode::UNAUTHORIZED,
            ),
        ] {
            let response = client
                .get(format!("http://{addr}/runners/events"))
                .query(&[("token", credential)])
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
            assert_ne!(
                response.headers().get("content-type").map(|v| v.as_bytes()),
                Some(b"text/event-stream".as_slice())
            );
        }
        let mut response = client
            .get(format!("http://{addr}/runners/events"))
            .query(&[("token", token(&state, &["admin"], "test"))])
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        assert_eq!(response.headers()["cache-control"], "no-cache");
        register(&state, "runner-sse").await;
        state
            .runner_registry
            .update_status("runner-sse", status(RunnerHealth::Degraded))
            .await;
        state.runner_registry.emit_command_response(
            "runner-sse",
            &CommandResponse {
                request_id: "cmd-1".into(),
                success: false,
                error: Some("雪\nfailed".into()),
                status: None,
            },
        );
        state.runner_registry.unregister("runner-sse").await;
        let mut bytes = Vec::new();
        while bytes.windows(2).filter(|v| *v == b"\n\n").count() < 4 {
            bytes.extend_from_slice(&response.chunk().await.unwrap().expect("live SSE stream"));
        }
        let text = std::str::from_utf8(&bytes).unwrap();
        let frames: Vec<_> = text.split("\n\n").filter(|v| !v.is_empty()).collect();
        assert_eq!(frames.len(), 4);
        for (frame, (name, kind)) in frames.iter().zip([
            ("runner_connected", "connected"),
            ("runner_status_changed", "status_changed"),
            ("runner_command_completed", "command_completed"),
            ("runner_disconnected", "disconnected"),
        ]) {
            let mut lines = frame.lines();
            assert_eq!(lines.next().unwrap(), format!("event: {name}"));
            let payload: serde_json::Value =
                serde_json::from_str(lines.next().unwrap().strip_prefix("data: ").unwrap())
                    .unwrap();
            assert!(lines.next().is_none());
            assert_eq!(payload["type"], kind);
            assert_eq!(payload["runner_id"], "runner-sse");
            if kind == "connected" {
                assert_eq!(payload["name"], "雪\nrunner");
            }
            if kind == "command_completed" {
                assert_eq!(payload["error"], "雪\nfailed");
            }
        }
        drop(response);
        stop.request();
        server.await.unwrap().unwrap();
    })
    .await
    .expect("admin SSE HTTP/disconnect contract timed out");
}

#[tokio::test]
async fn admin_sse_keepalive_and_lagged_broadcast_contract() {
    use futures_util::FutureExt;
    let state = create_test_state().await.unwrap();
    let response = routes::admin::router(state.clone())
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/runners/events?token={}",
                    token(&state, &["admin"], "test")
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(14)).await;
    assert!(body.next().now_or_never().is_none());
    tokio::time::advance(Duration::from_secs(1)).await;
    assert_eq!(body.next().await.unwrap().unwrap().as_ref(), b":\n\n");
    // Overflow the bounded registry channel while the body is unpolled.
    // Existing policy skips lag errors and resumes with the retained events.
    for i in 0..70 {
        register(&state, &format!("runner-{i}")).await;
    }
    let bytes = body.next().await.unwrap().unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.starts_with("event: runner_connected\ndata: "));
    assert!(!text.contains("event: error"));
    assert!(text.contains("runner-6"));
    tokio::time::advance(Duration::from_secs(14)).await;
    // Drain the remaining retained events, which continue resetting idle time.
    for _ in 0..63 {
        body.next().await.unwrap().unwrap();
    }
    assert!(body.next().now_or_never().is_none());
    tokio::time::advance(Duration::from_secs(15)).await;
    assert_eq!(body.next().await.unwrap().unwrap().as_ref(), b":\n\n");
    drop(body);
    tokio::time::resume();
}
