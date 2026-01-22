//! WebSocket client for gateway connection.

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::time::{interval, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use simple_ai_common::{CommandResponse, GatewayMessage, RunnerMessage, RunnerRegistration};

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
    /// This runs indefinitely, reconnecting on connection loss.
    pub async fn run(&self) {
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
        let (ws_stream, _) = connect_async(&self.config.ws_url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Send registration
        let status = self.status_collector.collect().await;
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
        write.send(Message::Text(json)).await?;
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

        // Create channel for outbound messages
        let (tx, mut rx) = mpsc::channel::<RunnerMessage>(32);

        // Spawn heartbeat task
        let heartbeat_interval = Duration::from_secs(self.config.heartbeat_interval_secs);
        let heartbeat_tx = tx.clone();
        let status_collector = self.status_collector.clone();
        let heartbeat_handle = tokio::spawn(async move {
            let mut ticker = interval(heartbeat_interval);
            loop {
                ticker.tick().await;
                let status = status_collector.collect().await;
                if heartbeat_tx
                    .send(RunnerMessage::Heartbeat(status))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });

        // Main message loop
        let result = self
            .message_loop(&mut write, &mut read, &mut rx, tx.clone())
            .await;

        // Clean up
        heartbeat_handle.abort();

        result
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
        loop {
            tokio::select! {
                // Handle outbound messages
                Some(msg) = rx.recv() => {
                    let json = serde_json::to_string(&msg)?;
                    write.send(Message::Text(json)).await?;
                    tracing::debug!("Sent message to gateway: {:?}", std::mem::discriminant(&msg));
                }

                // Handle inbound messages
                Some(result) = read.next() => {
                    match result {
                        Ok(Message::Text(text)) => {
                            if let Err(e) = self.handle_gateway_message(&text, &tx).await {
                                tracing::error!("Error handling gateway message: {}", e);
                            }
                        }
                        Ok(Message::Ping(data)) => {
                            write.send(Message::Pong(data)).await?;
                        }
                        Ok(Message::Close(_)) => {
                            tracing::info!("Gateway sent close frame");
                            return Ok(());
                        }
                        Ok(_) => {} // Ignore other message types
                        Err(e) => {
                            return Err(format!("WebSocket error: {}", e).into());
                        }
                    }
                }

                else => {
                    return Ok(());
                }
            }
        }
    }

    async fn handle_gateway_message(
        &self,
        text: &str,
        tx: &mpsc::Sender<RunnerMessage>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let msg: GatewayMessage = serde_json::from_str(text)?;
        tracing::debug!("Received gateway message: {:?}", std::mem::discriminant(&msg));

        match msg {
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

            GatewayMessage::LoadModel { model_id, request_id } => {
                let result = self.load_model(&model_id).await;
                let status = self.status_collector.collect().await;
                let response = CommandResponse {
                    request_id,
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                    status: Some(status),
                };
                tx.send(RunnerMessage::CommandResponse(response)).await?;
            }

            GatewayMessage::UnloadModel { model_id, request_id } => {
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

    async fn load_model(&self, model_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Find an engine that has this model
        if let Some(engine) = self.engine_registry.find_engine_for_model(model_id).await {
            engine.load_model(model_id).await?;
            tracing::info!("Loaded model: {}", model_id);
            return Ok(());
        }

        // Model not found in any engine's catalog - try loading via each engine
        let engines = self.engine_registry.all().await;
        if engines.is_empty() {
            return Err(format!("Cannot load model '{}': no engines registered", model_id).into());
        }

        let mut last_error = None;
        for engine in engines {
            match engine.load_model(model_id).await {
                Ok(()) => {
                    tracing::info!("Loaded model {} via {}", model_id, engine.engine_type());
                    return Ok(());
                }
                Err(e) => {
                    tracing::debug!(
                        "Engine {} failed to load model '{}': {}",
                        engine.engine_type(),
                        model_id,
                        e
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(format!(
            "Failed to load model '{}': {}",
            model_id,
            last_error.map(|e| e.to_string()).unwrap_or_else(|| "unknown error".to_string())
        ).into())
    }

    async fn unload_model(&self, model_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Try each engine
        for engine in self.engine_registry.all().await {
            if engine.unload_model(model_id).await.is_ok() {
                tracing::info!("Unloaded model {} from {}", model_id, engine.engine_type());
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
    use crate::config::{ApiConfig, CapabilitiesConfig, Config, EnginesConfig, RunnerConfig};
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

    #[test]
    fn test_gateway_client_creation() {
        let config = test_config();
        let gateway_config = test_gateway_config();
        let registry = Arc::new(EngineRegistry::new());
        let status_collector = Arc::new(StatusCollector::new(config.clone(), registry.clone()));

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
