//! WebSocket client for gateway connection.

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::time::{interval, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use simple_ai_common::{
    CommandResponse, GatewayMessage, RunnerMessage, RunnerRegistration, RunnerStatus,
};

use crate::config::GatewayConfig;
use crate::engine::EngineRegistry;

use super::StatusCollector;

/// Gateway WebSocket client.
///
/// Handles connection lifecycle, heartbeats, and message processing.
pub struct GatewayClient {
    config: GatewayConfig,
    runner_id: String,
    runner_name: String,
    machine_type: Option<String>,
    mac_address: Option<String>,
    http_port: u16,
    status_collector: Arc<StatusCollector>,
    engine_registry: Arc<EngineRegistry>,
}

impl GatewayClient {
    // Preserve the constructor API without introducing a parameter object.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: GatewayConfig,
        runner_id: String,
        runner_name: String,
        machine_type: Option<String>,
        mac_address: Option<String>,
        http_port: u16,
        status_collector: Arc<StatusCollector>,
        engine_registry: Arc<EngineRegistry>,
    ) -> Self {
        Self {
            config,
            runner_id,
            runner_name,
            machine_type,
            mac_address,
            http_port,
            status_collector,
            engine_registry,
        }
    }

    /// Start the client with automatic reconnection.
    ///
    /// Reconnects on connection loss until application shutdown.
    pub async fn run_until_shutdown(&self, stop: simple_server::lifecycle::Shutdown) {
        tokio::select! {
            biased;
            _ = stop.requested() => {},
            _ = self.run() => {},
        }
        tracing::info!("Gateway client stopped");
    }

    async fn run(&self) {
        loop {
            tracing::info!("Connecting to gateway at {}", self.config.ws_url);

            match self.connect_and_run().await {
                Ok(()) => {
                    tracing::info!("Gateway connection closed normally");
                }
                Err(e) => {
                    tracing::error!("Gateway connection error: {}", e);
                }
            }

            tracing::info!(
                "Reconnecting in {} seconds...",
                self.config.reconnect_delay_secs
            );
            tokio::time::sleep(Duration::from_secs(self.config.reconnect_delay_secs)).await;
        }
    }

    /// Connect to gateway and run message loop.
    async fn connect_and_run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (ws_stream, _) =
            timeout(Duration::from_secs(30), connect_async(&self.config.ws_url)).await??;
        let (mut write, mut read) = ws_stream.split();

        // Never probe engines before registration. The first heartbeat reports readiness.
        let status = RunnerStatus::starting();
        let mut registration = RunnerRegistration::new(
            self.runner_id.clone(),
            self.runner_name.clone(),
            self.machine_type.clone(),
            self.http_port,
            self.config.auth_token.clone(),
            status,
        );
        registration.mac_address = self.mac_address.clone();
        let msg = RunnerMessage::Register(registration);
        let json = serde_json::to_string(&msg)?;
        timeout(Duration::from_secs(5), write.send(Message::Text(json))).await??;
        tracing::info!("Sent registration to gateway");

        // Wait for registration ack with timeout
        let ack_timeout = Duration::from_secs(10);
        match timeout(ack_timeout, read.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                let response: GatewayMessage = serde_json::from_str(&text)?;
                match response {
                    GatewayMessage::RegisterAck { runner_id } => {
                        tracing::info!("Registration acknowledged for {}", runner_id);
                    }
                    GatewayMessage::Error { code, message } => {
                        return Err(format!("Registration failed: {} - {}", code, message).into());
                    }
                    _ => {
                        return Err("Unexpected response to registration".into());
                    }
                }
            }
            Ok(Some(Ok(_))) => {
                return Err("Expected text message for registration ack".into());
            }
            Ok(Some(Err(e))) => {
                return Err(format!("WebSocket error during registration: {}", e).into());
            }
            Ok(None) => {
                return Err("Connection closed during registration".into());
            }
            Err(_) => {
                return Err("Registration acknowledgment timeout".into());
            }
        }

        self.engine_registry.drain.arm(None);

        // Create channel for outbound messages
        let (tx, mut rx) = mpsc::channel::<RunnerMessage>(32);

        // Heartbeats belong to this connection future, so cancellation cannot
        // leave a detached status collector running after shutdown/reconnect.
        self.message_loop(&mut write, &mut read, &mut rx, tx.clone())
            .await
    }

    async fn message_loop<S, R>(
        &self,
        write: &mut S,
        read: &mut R,
        rx: &mut mpsc::Receiver<RunnerMessage>,
        tx: mpsc::Sender<RunnerMessage>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        S: SinkExt<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
        R: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
    {
        // These workers are owned by the connection future, not detached tasks.
        // Dropping the transport cancels slow probes and pending command handling.
        let (commands, mut pending_commands) = mpsc::channel::<String>(32);
        let heartbeats = async {
            let mut heartbeat = interval(Duration::from_secs(self.config.heartbeat_interval_secs));
            loop {
                heartbeat.tick().await;
                let status = self.status_collector.collect().await;
                tx.send(RunnerMessage::Heartbeat(status)).await?;
            }
            #[allow(unreachable_code)]
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        };
        let command_worker = async {
            while let Some(text) = pending_commands.recv().await {
                if let Err(e) = self.handle_gateway_message(&text, &tx).await {
                    tracing::error!("Error handling gateway message: {}", e);
                }
            }
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        };
        let transport = async {
            loop {
                tokio::select! {
                    Some(msg) = rx.recv() => {
                        let json = serde_json::to_string(&msg)?;
                        timeout(Duration::from_secs(10), write.send(Message::Text(json))).await??;
                        if let RunnerMessage::CommandResponse(response) = &msg {
                            if response.success {
                                self.engine_registry.drain.arm(Some(&response.request_id));
                            }
                        }
                        tracing::debug!("Sent message to gateway: {:?}", std::mem::discriminant(&msg));
                    }
                    result = read.next() => {
                        let Some(result) = result else { return Ok(()); };
                        match result? {
                            Message::Text(text) => {
                                // Never stop reading the socket behind a slow command.
                                commands.try_send(text)?;
                            }
                            Message::Ping(data) => {
                                timeout(Duration::from_secs(10), write.send(Message::Pong(data))).await??;
                            }
                            Message::Close(_) => {
                                tracing::info!("Gateway sent close frame");
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                }
            }
        };
        tokio::select! {
            result = transport => result,
            result = heartbeats => result,
            result = command_worker => result,
        }
    }

    async fn handle_gateway_message(
        &self,
        text: &str,
        tx: &mpsc::Sender<RunnerMessage>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let msg: GatewayMessage = serde_json::from_str(text)?;
        tracing::debug!(
            "Received gateway message: {:?}",
            std::mem::discriminant(&msg)
        );

        match msg {
            GatewayMessage::Drain { action, request_id } => {
                let result = self.engine_registry.drain.begin(action, &request_id);
                let status = self.status_collector.collect().await;
                tx.send(RunnerMessage::CommandResponse(CommandResponse {
                    request_id,
                    success: result.is_ok(),
                    error: result.err().map(str::to_string),
                    status: Some(status),
                }))
                .await?;
            }
            GatewayMessage::Ping { timestamp } => {
                tracing::debug!("Received ping with timestamp {}", timestamp);
                // Respond with current status
                let status = self.status_collector.collect().await;
                tx.send(RunnerMessage::StatusUpdate(status)).await?;
            }

            GatewayMessage::RequestStatus { request_id } => {
                let status = self.status_collector.collect().await;
                let response = CommandResponse {
                    request_id,
                    success: true,
                    error: None,
                    status: Some(status),
                };
                tx.send(RunnerMessage::CommandResponse(response)).await?;
            }

            GatewayMessage::LoadModel {
                model_id,
                request_id,
            } => {
                tracing::info!(%request_id, model = %model_id, "Gateway command triggering model load");
                let result = self.load_model(&model_id, &request_id).await;
                let status = self.status_collector.collect().await;
                let response = CommandResponse {
                    request_id,
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                    status: Some(status),
                };
                tx.send(RunnerMessage::CommandResponse(response)).await?;
            }

            GatewayMessage::UnloadModel {
                model_id,
                request_id,
            } => {
                tracing::info!(%request_id, model = %model_id, "Gateway command triggering model unload");
                let result = self.unload_model(&model_id).await;
                let status = self.status_collector.collect().await;
                let response = CommandResponse {
                    request_id,
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                    status: Some(status),
                };
                tx.send(RunnerMessage::CommandResponse(response)).await?;
            }

            GatewayMessage::RegisterAck { runner_id } => {
                // Unexpected after initial registration - could indicate protocol issue
                tracing::warn!(
                    "Received unexpected RegisterAck for {} after already registered",
                    runner_id
                );
            }

            GatewayMessage::Error { code, message } => {
                tracing::error!("Gateway error: {} - {}", code, message);
            }
        }

        Ok(())
    }

    async fn load_model(
        &self,
        model_id: &str,
        request_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _work = self
            .engine_registry
            .drain
            .admit()
            .ok_or("Runner is draining")?;
        // Resolve canonical name to local engine name if aliased
        let local_id = self.status_collector.resolve_to_local(model_id);
        let local_id = local_id.as_str();

        tracing::debug!("Loading model: {} (local: {})", model_id, local_id);

        // Find an engine that has this model
        if self
            .engine_registry
            .resolve_engine_for_model(local_id)
            .await
            .is_some()
        {
            drop(
                self.engine_registry
                    .acquire_model_for_request(local_id, request_id)
                    .await?,
            );
            tracing::info!("Loaded model: {} (local: {})", model_id, local_id);
            return Ok(());
        }

        // Model not found in any engine's catalog - try loading via each engine
        let engines = self.engine_registry.all().await;
        if engines.is_empty() {
            return Err(format!("Cannot load model '{}': no engines registered", model_id).into());
        }

        let mut last_error = None;
        for engine in engines {
            match engine.load_model(local_id).await {
                Ok(()) => {
                    tracing::info!(
                        "Loaded model {} (local: {}) via {}",
                        model_id,
                        local_id,
                        engine.engine_type()
                    );
                    return Ok(());
                }
                Err(e) => {
                    tracing::debug!(
                        "Engine {} failed to load model '{}' (local: '{}'): {}",
                        engine.engine_type(),
                        model_id,
                        local_id,
                        e
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(format!(
            "Failed to load model '{}' (local: '{}'): {}",
            model_id,
            local_id,
            last_error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown error".to_string())
        )
        .into())
    }

    async fn unload_model(
        &self,
        model_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _work = self
            .engine_registry
            .drain
            .admit()
            .ok_or("Runner is draining")?;
        // Resolve canonical name to local engine name if aliased
        let local_id = self.status_collector.resolve_to_local(model_id);
        let local_id = local_id.as_str();

        if let Some((engine, engine_model)) = self
            .engine_registry
            .resolve_engine_for_model(local_id)
            .await
        {
            if engine.unload_model(&engine_model).await.is_ok() {
                tracing::info!(
                    "Unloaded model {} (local: {}) from {}",
                    model_id,
                    local_id,
                    engine.engine_type()
                );
                return Ok(());
            }
        }
        // Not an error if model wasn't loaded
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AliasesConfig, ApiConfig, CapabilitiesConfig, Config, EnginesConfig, OcrConfig,
        RunnerConfig,
    };
    use crate::engine::EngineRegistry;
    use simple_ai_common::{RunnerStatus, PROTOCOL_VERSION};

    fn test_config() -> Config {
        Config {
            runner: RunnerConfig {
                id: "test-runner".to_string(),
                name: "Test Runner".to_string(),
                machine_type: None,
                mac_address: None,
            },
            api: ApiConfig::default(),
            gateway: None,
            engines: EnginesConfig::default(),
            capabilities: CapabilitiesConfig::default(),
            aliases: AliasesConfig::default(),
            model_routes: Default::default(),
            co_resident_models: Default::default(),
            engine_resources: Default::default(),
            ocr: OcrConfig::default(),
        }
    }

    fn test_gateway_config() -> GatewayConfig {
        GatewayConfig {
            ws_url: "ws://localhost:8081/ws/runners".to_string(),
            auth_token: "test-token".to_string(),
            reconnect_delay_secs: 5,
            heartbeat_interval_secs: 30,
        }
    }

    async fn stalled_engine_client(ws_url: String) -> (GatewayClient, wiremock::MockServer) {
        use wiremock::{matchers::any, Mock, MockServer, ResponseTemplate};

        let engine = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(60)))
            .mount(&engine)
            .await;
        let config = test_config();
        let registry = Arc::new(EngineRegistry::new());
        registry
            .register(Arc::new(crate::engine::OllamaEngine::with_batch_size(
                &engine.uri(),
                1,
            )))
            .await;
        let collector = Arc::new(StatusCollector::new(config, registry.clone(), false));
        let mut gateway = test_gateway_config();
        gateway.ws_url = ws_url;
        gateway.reconnect_delay_secs = 1;
        (
            GatewayClient::new(
                gateway,
                "test-runner".into(),
                "Test Runner".into(),
                None,
                None,
                8080,
                collector,
                registry,
            ),
            engine,
        )
    }

    async fn receive_registration<S>(socket: &mut tokio_tungstenite::WebSocketStream<S>)
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        let message = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let RunnerMessage::Register(registration) = serde_json::from_str(&message).unwrap() else {
            panic!("expected registration");
        };
        assert_eq!(registration.runner_id, "test-runner");
        assert_eq!(
            registration.status.health,
            simple_ai_common::RunnerHealth::Starting
        );
    }

    #[tokio::test]
    async fn registration_does_not_wait_for_stalled_health_endpoint() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (client, engine) =
            stalled_engine_client(format!("ws://{}", listener.local_addr().unwrap())).await;
        let server = async {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            timeout(
                Duration::from_millis(500),
                receive_registration(&mut socket),
            )
            .await
            .expect("registration must not wait for the 60-second health endpoint");
            assert!(engine.received_requests().await.unwrap().is_empty());
        };
        tokio::select! {
            _ = client.run() => panic!("client unexpectedly stopped"),
            result = timeout(Duration::from_secs(5), server) => result.unwrap(),
        }
    }

    #[tokio::test]
    async fn first_heartbeat_reports_readiness_after_registration_ack() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (client, engine) =
            stalled_engine_client(format!("ws://{}", listener.local_addr().unwrap())).await;
        engine.reset().await;
        wiremock::Mock::given(wiremock::matchers::any())
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"models": []})),
            )
            .mount(&engine)
            .await;
        let server = async {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            receive_registration(&mut socket).await;
            socket
                .send(Message::Text(
                    serde_json::to_string(&GatewayMessage::RegisterAck {
                        runner_id: "test-runner".into(),
                    })
                    .unwrap(),
                ))
                .await
                .unwrap();
            let message = socket.next().await.unwrap().unwrap().into_text().unwrap();
            let RunnerMessage::Heartbeat(status) = serde_json::from_str(&message).unwrap() else {
                panic!("expected immediate heartbeat");
            };
            assert_eq!(status.health, simple_ai_common::RunnerHealth::Healthy);
            assert_eq!(status.engines.len(), 1);
            assert!(status.engines[0].is_healthy);
        };
        tokio::select! {
            _ = client.run() => panic!("client unexpectedly stopped"),
            result = timeout(Duration::from_secs(5), server) => result.unwrap(),
        }
    }

    #[tokio::test]
    async fn reconnects_after_close_before_registration_and_during_stalled_status() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (client, engine) =
            stalled_engine_client(format!("ws://{}", listener.local_addr().unwrap())).await;
        let server = async {
            // Reproduce a gateway closing before it consumes registration.
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            socket.close(None).await.unwrap();
            drop(socket);

            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            receive_registration(&mut socket).await;
            let ack = serde_json::to_string(&GatewayMessage::RegisterAck {
                runner_id: "test-runner".into(),
            })
            .unwrap();
            socket.send(Message::Text(ack.clone())).await.unwrap();
            // Ensure status collection really is blocked in an HTTP health probe.
            while engine.received_requests().await.unwrap().is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            socket.close(None).await.unwrap();
            drop(socket);

            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            receive_registration(&mut socket).await;
            socket.send(Message::Text(ack)).await.unwrap();
            // A ping response proves the acknowledged connection is usable even
            // with another health probe stalled.
            socket.send(Message::Ping(vec![1, 2, 3])).await.unwrap();
            assert_eq!(
                socket.next().await.unwrap().unwrap(),
                Message::Pong(vec![1, 2, 3])
            );
        };
        tokio::select! {
            _ = client.run() => panic!("client unexpectedly stopped"),
            result = timeout(Duration::from_secs(8), server) => result.unwrap(),
        }
    }

    #[tokio::test]
    async fn reconnects_when_registration_ack_never_arrives() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (client, _engine) =
            stalled_engine_client(format!("ws://{}", listener.local_addr().unwrap())).await;
        let server = async {
            let (stream, _) = listener.accept().await.unwrap();
            let mut silent = tokio_tungstenite::accept_async(stream).await.unwrap();
            receive_registration(&mut silent).await;
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            receive_registration(&mut socket).await;
        };
        tokio::select! {
            _ = client.run() => panic!("client unexpectedly stopped"),
            result = timeout(Duration::from_secs(15), server) => result.unwrap(),
        }
    }

    #[test]
    fn test_gateway_client_creation() {
        let config = test_config();
        let gateway_config = test_gateway_config();
        let registry = Arc::new(EngineRegistry::new());
        let status_collector = Arc::new(StatusCollector::new(
            config.clone(),
            registry.clone(),
            false,
        ));

        let client = GatewayClient::new(
            gateway_config,
            config.runner.id.clone(),
            config.runner.name.clone(),
            config.runner.machine_type.clone(),
            None,
            8080,
            status_collector,
            registry,
        );

        assert_eq!(client.runner_id, "test-runner");
        assert_eq!(client.runner_name, "Test Runner");
        assert_eq!(client.http_port, 8080);
    }

    #[test]
    fn test_runner_registration_creation() {
        let status = RunnerStatus::starting();
        let reg = RunnerRegistration::new(
            "runner-1".to_string(),
            "Runner One".to_string(),
            Some("gpu".to_string()),
            8080,
            "token".to_string(),
            status,
        );

        assert_eq!(reg.runner_id, "runner-1");
        assert_eq!(reg.http_port, 8080);
        assert_eq!(reg.protocol_version, PROTOCOL_VERSION);
    }
}
