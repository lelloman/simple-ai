use futures_util::{SinkExt, StreamExt};
use simple_ai_backend::{
    gateway::{ws_handler, WsState},
    AuditLogger, RunnerRegistry,
};
use simple_ai_common::{GatewayMessage, RunnerMessage, RunnerRegistration, RunnerStatus};
use simple_server::web::{routing::get, Router};
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::test]
async fn gateway_auth_registration_ping_and_disconnect_over_tcp() {
    timeout(Duration::from_secs(10), async {
        let registry = Arc::new(RunnerRegistry::new());
        let state = Arc::new(WsState {
            registry: registry.clone(),
            auth_token: "transport-secret".into(),
            audit_logger: Arc::new(AuditLogger::new(":memory:").unwrap()),
            batch_dispatcher: None,
        });
        let app = Router::new().route("/ws", get(ws_handler)).with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            simple_server::web::serve_with_connect_info(listener, app, simple_server::lifecycle::Shutdown::new()).await.unwrap();
        });
        let registration = |token: &str| RunnerRegistration::new(
            "transport-runner".into(), "Transport Runner".into(), None, 18080,
            token.into(), RunnerStatus::starting(),
        );
        let (mut rejected, response) = connect_async(format!("ws://{addr}/ws")).await.unwrap();
        assert_eq!(response.status(), 101);
        rejected.send(Message::Text(serde_json::to_string(&RunnerMessage::Register(registration("wrong"))).unwrap())).await.unwrap();
        let error: GatewayMessage = serde_json::from_str(rejected.next().await.unwrap().unwrap().to_text().unwrap()).unwrap();
        assert!(matches!(error, GatewayMessage::Error { code, .. } if code == "AUTH_FAILED"));
        assert_eq!(registry.count().await, 0);

        let (mut socket, _) = connect_async(format!("ws://{addr}/ws")).await.unwrap();
        socket.send(Message::Text(serde_json::to_string(&RunnerMessage::Register(registration("transport-secret"))).unwrap())).await.unwrap();
        let ack: GatewayMessage = serde_json::from_str(socket.next().await.unwrap().unwrap().to_text().unwrap()).unwrap();
        assert!(matches!(ack, GatewayMessage::RegisterAck { runner_id } if runner_id == "transport-runner"));
        while registry.count().await == 0 { tokio::task::yield_now().await; }
        assert_eq!(registry.get("transport-runner").await.unwrap().http_base_url.as_deref(), Some("http://127.0.0.1:18080"));
        socket.send(Message::Ping(vec![1, 2, 3])).await.unwrap();
        assert_eq!(socket.next().await.unwrap().unwrap(), Message::Pong(vec![1, 2, 3]));
        socket.close(None).await.unwrap();
        while registry.count().await != 0 { tokio::task::yield_now().await; }
        server.abort();
        let _ = server.await;
    }).await.expect("WebSocket transport test timed out");
}

#[tokio::test]
async fn replaced_or_unknown_connection_closes_on_next_status_message() {
    timeout(Duration::from_secs(10), async {
        let registry = Arc::new(RunnerRegistry::new());
        let state = Arc::new(WsState {
            registry: registry.clone(),
            auth_token: "secret".into(),
            audit_logger: Arc::new(AuditLogger::new(":memory:").unwrap()),
            batch_dispatcher: None,
        });
        let app = Router::new()
            .route("/ws", get(ws_handler))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            simple_server::web::serve_with_connect_info(
                listener,
                app,
                simple_server::lifecycle::Shutdown::new(),
            )
            .await
            .unwrap();
        });

        for message in [
            RunnerMessage::Heartbeat(RunnerStatus::starting()),
            RunnerMessage::StatusUpdate(RunnerStatus::starting()),
        ] {
            for replacement in [false, true] {
                let (mut socket, _) = connect_async(format!("ws://{addr}/ws")).await.unwrap();
                let registration = RunnerRegistration::new(
                    "runner".into(),
                    "Runner".into(),
                    None,
                    18080,
                    "secret".into(),
                    RunnerStatus::starting(),
                );
                socket
                    .send(Message::Text(
                        serde_json::to_string(&RunnerMessage::Register(registration)).unwrap(),
                    ))
                    .await
                    .unwrap();
                assert!(matches!(
                    socket.next().await.unwrap().unwrap(),
                    Message::Text(_)
                ));
                // Ping orders observation after registration, whose ack precedes insertion.
                socket.send(Message::Ping(vec![])).await.unwrap();
                assert_eq!(socket.next().await.unwrap().unwrap(), Message::Pong(vec![]));
                let first = registry.get("runner").await.unwrap().generation;
                let (tx, _rx) = tokio::sync::mpsc::channel(32);
                let second = if replacement {
                    Some(
                        registry
                            .register(
                                "runner".into(),
                                "Replacement".into(),
                                None,
                                RunnerStatus::starting(),
                                None,
                                tx,
                                None,
                            )
                            .await,
                    )
                } else {
                    assert!(registry
                        .unregister_connection("runner", first)
                        .await
                        .is_some());
                    None
                };
                let mut events = registry.subscribe_events();
                socket
                    .send(Message::Text(serde_json::to_string(&message).unwrap()))
                    .await
                    .unwrap();
                assert!(matches!(
                    socket.next().await.unwrap().unwrap(),
                    Message::Close(_)
                ));
                // Draining the socket waits for the server handler to finish its exit path.
                while let Some(frame) = socket.next().await {
                    if frame.is_err() {
                        break;
                    }
                }
                assert!(matches!(
                    events.try_recv(),
                    Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                ));
                if let Some(second) = second {
                    assert_eq!(registry.get("runner").await.unwrap().generation, second);
                    assert!(registry
                        .unregister_connection("runner", second)
                        .await
                        .is_some());
                } else {
                    assert!(registry.get("runner").await.is_none());
                }
            }
        }
        server.abort();
        let _ = server.await;
    })
    .await
    .expect("stale WebSocket did not close on its next heartbeat/status");
}
