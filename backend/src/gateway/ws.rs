//! WebSocket handler for runner connections.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        ConnectInfo, State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

use simple_ai_common::{GatewayMessage, RunnerMessage, RunnerRegistration, PROTOCOL_VERSION};

use crate::audit::AuditLogger;
use super::{BatchDispatcher, RunnerRegistry};

/// Shared state for WebSocket connections.
pub struct WsState {
    pub registry: Arc<RunnerRegistry>,
    pub auth_token: String,
    pub audit_logger: Arc<AuditLogger>,
    /// Optional batch dispatcher for cache invalidation on runner changes.
    pub batch_dispatcher: Option<Arc<BatchDispatcher>>,
}

/// WebSocket upgrade handler.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<WsState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    tracing::info!("Runner connection attempt from {}", addr);
    ws.on_upgrade(move |socket| handle_runner(socket, state, addr))
}

/// Handle an individual runner connection.
async fn handle_runner(socket: WebSocket, state: Arc<WsState>, addr: SocketAddr) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Wait for registration message
    let registration_timeout = Duration::from_secs(10);
    let registration = match timeout(registration_timeout, ws_rx.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => match serde_json::from_str::<RunnerMessage>(&text) {
            Ok(RunnerMessage::Register(reg)) => reg,
            Ok(_) => {
                tracing::warn!("Expected Register message from {}, got different type", addr);
                let _ = send_error(&mut ws_tx, "PROTOCOL_ERROR", "Expected Register message").await;
                return;
            }
            Err(e) => {
                tracing::warn!("Failed to parse registration from {}: {}", addr, e);
                let _ = send_error(&mut ws_tx, "PARSE_ERROR", &e.to_string()).await;
                return;
            }
        },
        Ok(Some(Ok(_))) => {
            tracing::warn!("Expected text message for registration from {}", addr);
            let _ = send_error(&mut ws_tx, "PROTOCOL_ERROR", "Expected text message").await;
            return;
        }
        Ok(Some(Err(e))) => {
            tracing::warn!("WebSocket error during registration from {}: {}", addr, e);
            return;
        }
        Ok(None) => {
            tracing::info!("Connection closed before registration from {}", addr);
            return;
        }
        Err(_) => {
            tracing::warn!("Registration timeout from {}", addr);
            let _ = send_error(&mut ws_tx, "TIMEOUT", "Registration timeout").await;
            return;
        }
    };

    // Validate registration
    if let Err(msg) = validate_registration(&registration, &state.auth_token) {
        tracing::warn!(
            "Registration validation failed for {} from {}: {}",
            registration.runner_id,
            addr,
            msg
        );
        let _ = send_error(&mut ws_tx, "AUTH_FAILED", &msg).await;
        return;
    }

    tracing::info!(
        "Runner {} ({}) registered from {}",
        registration.runner_id,
        registration.runner_name,
        addr
    );

    // Send registration acknowledgment
    let ack = GatewayMessage::RegisterAck {
        runner_id: registration.runner_id.clone(),
    };
    if let Err(e) = send_message(&mut ws_tx, &ack).await {
        tracing::error!("Failed to send RegisterAck: {}", e);
        return;
    }

    // Create channel for outbound messages to this runner
    let (tx, mut rx) = mpsc::channel::<GatewayMessage>(32);

    // Derive HTTP base URL from socket address and registered port
    let http_base_url = format!("http://{}:{}", addr.ip(), registration.http_port);

    // Use MAC address from registration if provided, otherwise try ARP lookup
    let mac_address = if let Some(ref mac) = registration.mac_address {
        tracing::info!("Runner {} provided MAC address: {}", registration.runner_id, mac);
        Some(mac.clone())
    } else {
        // Fall back to ARP lookup (works for non-Docker deployments)
        let arp_mac = crate::arp::lookup_mac(&addr.ip());
        if let Some(ref mac) = arp_mac {
            tracing::info!("Discovered MAC {} for runner {} via ARP ({})", mac, registration.runner_id, addr.ip());
        } else {
            tracing::debug!("No MAC address for runner {} - not provided and not found in ARP cache", registration.runner_id);
        }
        arp_mac
    };

    // Register the runner
    state
        .registry
        .register(
            registration.runner_id.clone(),
            registration.runner_name.clone(),
            registration.machine_type.clone(),
            registration.status.clone(),
            Some(http_base_url),
            tx,
            mac_address.clone(),
        )
        .await;

    // Invalidate batch size cache since a new runner connected
    if let Some(ref dispatcher) = state.batch_dispatcher {
        dispatcher.invalidate_cache().await;
    }

    // Extract available models from runner status
    let available_models: Vec<String> = registration
        .status
        .engines
        .iter()
        .flat_map(|e| e.available_models.iter().map(|m| m.id.clone()))
        .collect();

    // Persist runner to database for WOL and offline tracking
    if let Err(e) = state.audit_logger.upsert_runner(
        &registration.runner_id,
        &registration.runner_name,
        mac_address.as_deref(),
        registration.machine_type.as_deref(),
        Some(&available_models),
    ) {
        tracing::warn!("Failed to persist runner {}: {}", registration.runner_id, e);
    }

    let runner_id = registration.runner_id.clone();

    // Main message loop
    loop {
        tokio::select! {
            // Outbound messages (from gateway to runner)
            Some(msg) = rx.recv() => {
                if let Err(e) = send_message(&mut ws_tx, &msg).await {
                    tracing::error!("Failed to send message to {}: {}", runner_id, e);
                    break;
                }
            }

            // Inbound messages (from runner to gateway)
            Some(result) = ws_rx.next() => {
                match result {
                    Ok(Message::Text(text)) => {
                        if let Err(e) = handle_runner_message(&text, &runner_id, &state.registry).await {
                            tracing::error!("Error handling message from {}: {}", runner_id, e);
                        }
                    }
                    Ok(Message::Ping(data)) => {
                        if let Err(e) = ws_tx.send(Message::Pong(data)).await {
                            tracing::error!("Failed to send pong to {}: {}", runner_id, e);
                            break;
                        }
                    }
                    Ok(Message::Close(_)) => {
                        tracing::info!("Runner {} sent close frame", runner_id);
                        break;
                    }
                    Ok(_) => {} // Ignore binary, pong, etc.
                    Err(e) => {
                        tracing::error!("WebSocket error from {}: {}", runner_id, e);
                        break;
                    }
                }
            }

            else => break,
        }
    }

    // Unregister the runner
    state.registry.unregister(&runner_id).await;

    // Invalidate batch size cache since a runner disconnected
    if let Some(ref dispatcher) = state.batch_dispatcher {
        dispatcher.invalidate_cache().await;
    }

    tracing::info!("Runner {} disconnected", runner_id);
}

/// Validate a runner registration.
fn validate_registration(reg: &RunnerRegistration, expected_token: &str) -> Result<(), String> {
    // Check auth token
    if reg.auth_token != expected_token {
        return Err("Invalid auth token".to_string());
    }

    // Check protocol version
    if reg.protocol_version != PROTOCOL_VERSION {
        return Err(format!(
            "Protocol version mismatch: expected {}, got {}",
            PROTOCOL_VERSION, reg.protocol_version
        ));
    }

    // Check runner ID is not empty
    if reg.runner_id.is_empty() {
        return Err("Runner ID cannot be empty".to_string());
    }

    Ok(())
}

/// Handle a message from a runner.
async fn handle_runner_message(
    text: &str,
    runner_id: &str,
    registry: &RunnerRegistry,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let msg: RunnerMessage = serde_json::from_str(text)?;

    match msg {
        RunnerMessage::Heartbeat(status) => {
            tracing::debug!("Heartbeat from {}: {:?}", runner_id, status.health);
            registry.update_status(runner_id, status).await;
        }
        RunnerMessage::StatusUpdate(status) => {
            tracing::debug!("Status update from {}: {:?}", runner_id, status.health);
            registry.update_status(runner_id, status).await;
        }
        RunnerMessage::CommandResponse(response) => {
            tracing::debug!(
                "Command response from {}: request_id={}, success={}",
                runner_id,
                response.request_id,
                response.success
            );
            // Update status if included
            if let Some(status) = response.status {
                registry.update_status(runner_id, status).await;
            }
            // TODO: Route response back to waiting request (for load/unload commands)
        }
        RunnerMessage::Register(_) => {
            tracing::warn!(
                "Unexpected Register message from {} after already registered",
                runner_id
            );
        }
    }

    Ok(())
}

/// Send a GatewayMessage over WebSocket.
async fn send_message<S>(
    sink: &mut S,
    msg: &GatewayMessage,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    S: SinkExt<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let json = serde_json::to_string(msg)?;
    sink.send(Message::Text(json)).await?;
    Ok(())
}

/// Send an error message over WebSocket.
async fn send_error<S>(
    sink: &mut S,
    code: &str,
    message: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    S: SinkExt<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let msg = GatewayMessage::Error {
        code: code.to_string(),
        message: message.to_string(),
    };
    send_message(sink, &msg).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use simple_ai_common::RunnerStatus;

    #[test]
    fn test_validate_registration_success() {
        let status = RunnerStatus::starting();
        let reg = RunnerRegistration::new(
            "runner-1".to_string(),
            "Test Runner".to_string(),
            Some("gpu".to_string()),
            8080,
            "secret-token".to_string(),
            status,
        );

        let result = validate_registration(&reg, "secret-token");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_registration_bad_token() {
        let status = RunnerStatus::starting();
        let reg = RunnerRegistration::new(
            "runner-1".to_string(),
            "Test".to_string(),
            None,
            8080,
            "wrong-token".to_string(),
            status,
        );

        let result = validate_registration(&reg, "correct-token");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid auth token"));
    }

    #[test]
    fn test_validate_registration_empty_id() {
        let status = RunnerStatus::starting();
        let reg = RunnerRegistration::new(
            "".to_string(),
            "Test".to_string(),
            None,
            8080,
            "token".to_string(),
            status,
        );

        let result = validate_registration(&reg, "token");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Runner ID cannot be empty"));
    }
}
