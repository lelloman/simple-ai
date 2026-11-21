//! `/v1/chat/completions` must not wait on audit persistence before enqueue.
use simple_ai_backend::{
    config::Config, test_util, BatchDispatcher, BatchQueue, BatchQueueConfig, RequestScheduler,
};
use simple_ai_common::{
    ChatCompletionResponse, ChatMessage, EngineStatus, ModelInfo, RunnerHealth, RunnerStatus,
};
use simple_server::web::{routing::post, Json, Router};
use std::{sync::Arc, time::Duration};

const CONCURRENCY: usize = 64;
const RUNNER_DELAY: Duration = Duration::from_millis(200);

fn runner_status() -> RunnerStatus {
    RunnerStatus {
        health: RunnerHealth::Healthy,
        capabilities: vec![],
        engines: vec![EngineStatus {
            engine_type: "test".into(),
            resource_group: None,
            is_healthy: true,
            version: None,
            loaded_models: vec!["model-a".into()],
            available_models: vec![ModelInfo {
                revision: None,
                id: "model-a".into(),
                name: "model-a".into(),
                size_bytes: None,
                parameter_count: None,
                context_length: None,
                quantization: None,
                modified_at: None,
                reasoning: None,
            }],
            error: None,
            batch_size: 256,
            prompt_cache: None,
        }],
        metrics: None,
        model_aliases: Default::default(),
    }
}

async fn mock_oidc() -> wiremock::MockServer {
    let oidc = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::path(
        "/.well-known/openid-configuration",
    ))
    .respond_with(
        wiremock::ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"jwks_uri": format!("{}/jwks", oidc.uri())})),
    )
    .mount(&oidc)
    .await;
    wiremock::Mock::given(wiremock::matchers::path("/jwks"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(
                serde_json::from_str::<serde_json::Value>(include_str!(
                    "fixtures/test-oidc-jwks.json"
                ))
                .unwrap(),
            ),
        )
        .mount(&oidc)
        .await;
    oidc
}

fn count(conn: &rusqlite::Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

#[tokio::test]
async fn concurrent_chat_requests_do_not_wait_on_a_stalled_audit_database() {
    let oidc = mock_oidc().await;
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("audit.db");
    let mut config: Config = test_util::test_config();
    config.oidc.issuer = oidc.uri();
    config.database.url = format!("sqlite:{}", db_path.display());
    config.gateway.enabled = true;
    let mut state = test_util::create_test_state_with_config(config).await;

    let queue = Arc::new(BatchQueue::new(BatchQueueConfig::default()));
    let dispatcher = Arc::new(
        BatchDispatcher::new(
            queue.clone(),
            state.runner_registry.clone(),
            state.inference_router.clone(),
            state.router_telemetry.clone(),
        )
        .with_wake_service(state.wake_service.clone()),
    );
    state.request_scheduler = Arc::new(RequestScheduler::new(
        state.inference_router.clone(),
        state.runner_registry.clone(),
        state.wake_service.clone(),
        state.router_telemetry.clone(),
        Some(queue.clone()),
        state.config.routing.clone(),
    ));
    state.batch_queue = Some(queue);
    state.batch_dispatcher = Some(dispatcher.clone());

    state
        .audit_logger
        .find_or_create_user("tagger-user", None)
        .unwrap();
    let (_, api_key) = state
        .audit_logger
        .create_api_key(
            "tagger-user",
            "bulk-tagger",
            &["model:specific".to_string()],
        )
        .unwrap();
    let audit = state.audit_logger.clone();

    let runner = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            tokio::time::sleep(RUNNER_DELAY).await;
            Json(ChatCompletionResponse::new(
                "model-a".into(),
                ChatMessage {
                    role: "assistant".into(),
                    content: Some("ok".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Some("stop".into()),
            ))
        }),
    );
    let runner_listener = simple_server::http::bind("127.0.0.1:0").await.unwrap();
    let runner_addr = runner_listener.local_addr().unwrap();
    let stop = simple_server::lifecycle::Shutdown::new();
    let runner_server = tokio::spawn(simple_server::web::serve(
        runner_listener,
        runner,
        stop.clone(),
    ));
    let (tx, _rx) = tokio::sync::mpsc::channel(32);
    state
        .runner_registry
        .register(
            "runner".into(),
            "runner".into(),
            None,
            runner_status(),
            Some(format!("http://{runner_addr}")),
            tx,
            None,
        )
        .await;
    let dispatch_stop = stop.clone();
    let dispatch_task =
        tokio::spawn(async move { dispatcher.run_until_shutdown(dispatch_stop).await });

    let app = Router::new()
        .nest(
            "/v1",
            simple_ai_backend::routes::chat::router(Arc::new(state)),
        )
        .layer(simple_server::web::middleware::from_fn(
            simple_ai_backend::logging::request_logger,
        ));
    let listener = simple_server::http::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(simple_server::web::serve_with_connect_info(
        listener,
        app,
        stop.clone(),
    ));

    // Another writer holds SQLite's write lock for longer than the whole run,
    // standing in for a disk that stalls every commit.
    let blocker = rusqlite::Connection::open(&db_path).unwrap();
    blocker.execute_batch("BEGIN IMMEDIATE").unwrap();

    let client = reqwest::Client::new();
    let started = std::time::Instant::now();
    let responses = futures_util::future::join_all((0..CONCURRENCY).map(|_| {
        let client = client.clone();
        let api_key = api_key.clone();
        async move {
            client
                .post(format!("http://{addr}/v1/chat/completions"))
                .bearer_auth(api_key)
                .json(&serde_json::json!({
                    "model": "model-a",
                    "messages": [{"role": "user", "content": "tag this"}]
                }))
                .send()
                .await
                .unwrap()
                .status()
        }
    }))
    .await;
    let elapsed = started.elapsed();
    assert!(
        responses.iter().all(|status| status.is_success()),
        "{responses:?}"
    );
    assert!(
        elapsed < Duration::from_millis(1500),
        "{CONCURRENCY} requests with a {RUNNER_DELAY:?} runner took {elapsed:?} while the audit \
         database was write-locked"
    );

    // Once the stall ends every audit row is persisted.
    blocker.execute_batch("ROLLBACK").unwrap();
    audit.flush();
    let reader = rusqlite::Connection::open(&db_path).unwrap();
    assert_eq!(
        count(&reader, "SELECT COUNT(*) FROM requests"),
        CONCURRENCY as i64
    );
    assert_eq!(
        count(&reader, "SELECT COUNT(*) FROM responses WHERE status = 200"),
        CONCURRENCY as i64
    );
    assert!(
        count(
            &reader,
            "SELECT COUNT(*) FROM api_keys WHERE last_used_at IS NOT NULL"
        ) >= 1
    );
    assert_eq!(
        reader
            .query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
            .unwrap(),
        "wal"
    );

    stop.request();
    dispatch_task.await.unwrap();
    server.await.unwrap().unwrap();
    runner_server.await.unwrap().unwrap();
}
