//! Regression for a host connecting only after idle-manager's wake retry.
use simple_ai_backend::{
    audit::AuditLogger,
    config::{GatewayConfig, ModelsConfig, RoutingConfig, WolConfig},
    gateway::{ModelRequest, RunnerRegistry},
    wol::{WakeError, WakeService},
};
use simple_ai_common::protocol::{RunnerHealth, RunnerStatus};
use std::{sync::Arc, time::Duration};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

async fn fixture(
    budget: u64,
) -> (
    MockServer,
    tempfile::TempDir,
    Arc<RunnerRegistry>,
    tokio::task::JoinHandle<Result<simple_ai_backend::wol::WakeResult, WakeError>>,
) {
    let http = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/nodes/fixture/wake"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&http)
        .await;
    let dir = tempfile::tempdir().unwrap();
    let audit = Arc::new(AuditLogger::new(dir.path().join("audit.db").to_str().unwrap()).unwrap());
    audit
        .upsert_runner(
            "fixture",
            "Fixture",
            Some("00:11:22:33:44:55"),
            None,
            Some(&["code:smart".into()]),
        )
        .unwrap();
    let registry = Arc::new(RunnerRegistry::new());
    let gateway = GatewayConfig {
        wake_timeout_secs: budget,
        idle_manager_url: Some(http.uri()),
        auto_wake_enabled: true,
        ..Default::default()
    };
    let service = WakeService::new(
        registry.clone(),
        audit,
        gateway,
        WolConfig::default(),
        ModelsConfig::default(),
        RoutingConfig::default(),
    );
    let task = tokio::spawn(async move {
        service
            .wake_and_wait(&ModelRequest::Specific("code:smart".into()))
            .await
    });
    // Complete real HTTP I/O before switching the wait to Tokio's virtual clock.
    tokio::time::timeout(Duration::from_secs(3), async {
        while http.received_requests().await.unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!task.is_finished());
    (http, dir, registry, task)
}

#[tokio::test]
async fn retry_connection_after_211_seconds_survives_default_deadline() {
    assert_eq!(GatewayConfig::default().wake_timeout_secs, 480);
    let (_http, _dir, registry, task) = fixture(GatewayConfig::default().wake_timeout_secs).await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(211)).await;
    tokio::task::yield_now().await;
    assert!(
        !task.is_finished(),
        "deadline must allow the observed second wake attempt"
    );
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    registry
        .register(
            "fixture".into(),
            "Fixture".into(),
            None,
            RunnerStatus {
                health: RunnerHealth::Healthy,
                capabilities: vec![],
                engines: vec![],
                metrics: None,
                model_aliases: Default::default(),
            },
            None,
            tx,
            None,
        )
        .await;
    assert_eq!(task.await.unwrap().unwrap().runner_id, "fixture");
    tokio::time::resume();
}

#[tokio::test]
async fn old_two_minute_budget_reproduces_failure_before_retry() {
    let (_http, _dir, _registry, task) = fixture(120).await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(121)).await;
    assert!(matches!(task.await.unwrap(), Err(WakeError::Timeout(120))));
    tokio::time::resume();
}

#[tokio::test]
async fn new_deadline_is_still_bounded_when_runner_never_connects() {
    let (_http, _dir, _registry, task) = fixture(480).await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(481)).await;
    assert!(matches!(task.await.unwrap(), Err(WakeError::Timeout(480))));
    tokio::time::resume();
}
