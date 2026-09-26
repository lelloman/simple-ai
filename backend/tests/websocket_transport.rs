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
