//! WebSocket handler for runner connections.

use std::net::SocketAddr;
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use simple_server::web::ws::{Message, WebSocket, WebSocketUpgrade};
use simple_server::web::{
    extract::{ConnectInfo, State},
    response::IntoResponse,
};
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

use simple_ai_common::{GatewayMessage, RunnerMessage, RunnerRegistration, PROTOCOL_VERSION};

use super::{BatchDispatcher, RunnerRegistry};
use crate::audit::AuditLogger;
use crate::runner_state::RunnerState;

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
                tracing::warn!(
                    "Expected Register message from {}, got different type",
                    addr
                );
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
        tracing::info!(
            "Runner {} provided MAC address: {}",
            registration.runner_id,
            mac
        );
        Some(mac.clone())
    } else {
        // Fall back to ARP lookup (works for non-Docker deployments)
        let arp_mac = crate::arp::lookup_mac(&addr.ip());
        if let Some(ref mac) = arp_mac {
            tracing::info!(
                "Discovered MAC {} for runner {} via ARP ({})",
                mac,
                registration.runner_id,
                addr.ip()
            );
        } else {
            tracing::debug!(
                "No MAC address for runner {} - not provided and not found in ARP cache",
                registration.runner_id
            );
        }
        arp_mac
    };

    // Register the runner
    let generation = state
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

    persist_runner_status(
        &state.audit_logger,
        &registration,
        mac_address.as_deref(),
        &registration.status,
    );

    let runner_id = registration.runner_id.clone();

    // A disabled or draining runner must not keep models loaded, including
    // after a reconnect or restart.
    if state.registry.state_of(&runner_id).await != RunnerState::Enabled {
        state.registry.enforce_runner_state(&runner_id);
    }

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
                        match handle_runner_message(&text, &runner_id, generation, &state.registry).await {
                            Ok(false) => {
                                tracing::warn!(runner = %runner_id, generation, "Runner connection is no longer registered; closing for reconnect");
                                let _ = ws_tx.send(Message::Close(None)).await;
                                break;
                            }
                            Err(e) => tracing::error!("Error handling message from {}: {}", runner_id, e),
                            Ok(true) => {}
                        }
                        if let Some(runner) = state.registry.get(&runner_id).await {
                            persist_runner_status(&state.audit_logger, &registration, mac_address.as_deref(), &runner.status);
                        }
                        if state.registry.state_of(&runner_id).await != RunnerState::Enabled {
                            state.registry.enforce_runner_state(&runner_id);
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
    if state
        .registry
        .unregister_connection(&runner_id, generation)
        .await
        .is_some()
    {
        tracing::info!("Runner {} disconnected", runner_id);
    }
}

/// Keep the offline wake catalog once readiness arrives, rather than replacing
/// it with the deliberately empty catalog in a prompt starting registration.
fn persist_runner_status(
    audit: &AuditLogger,
    registration: &RunnerRegistration,
    mac_address: Option<&str>,
    status: &simple_ai_common::RunnerStatus,
) {
    let available_models = if status.health == simple_ai_common::RunnerHealth::Starting {
        None
    } else {
        let mut models: Vec<String> = status
            .engines
            .iter()
            .flat_map(|e| e.available_models.iter().map(|m| m.id.clone()))
            .chain(status.model_aliases.keys().cloned())
            .collect();
        models.sort();
        models.dedup();
        Some(models)
    };
    if let Err(e) = audit.upsert_runner(
        &registration.runner_id,
        &registration.runner_name,
        mac_address,
        registration.machine_type.as_deref(),
        available_models.as_deref(),
    ) {
        tracing::warn!("Failed to persist runner {}: {}", registration.runner_id, e);
    }
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

/// Handle a message; false tells the socket loop to close and force reconnect.
async fn handle_runner_message(
    text: &str,
    runner_id: &str,
    generation: u64,
    registry: &RunnerRegistry,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    if registry
        .get(runner_id)
        .await
        .is_none_or(|runner| runner.generation != generation)
    {
        return Ok(false);
    }
    let msg: RunnerMessage = serde_json::from_str(text)?;

    match msg {
        RunnerMessage::Heartbeat(status) => {
            tracing::debug!("Heartbeat from {}: {:?}", runner_id, status.health);
            if !registry
                .update_connection_status(runner_id, generation, status)
                .await
            {
                return Ok(false);
            }
        }
        RunnerMessage::StatusUpdate(status) => {
            tracing::debug!("Status update from {}: {:?}", runner_id, status.health);
            if !registry
                .update_connection_status(runner_id, generation, status)
                .await
            {
                return Ok(false);
            }
        }
        RunnerMessage::CommandResponse(response) => {
            tracing::debug!(
                "Command response from {}: request_id={}, success={}",
                runner_id,
                response.request_id,
                response.success
            );
            // Update status if included
            if let Some(status) = response.status.as_ref() {
                if !registry
                    .update_connection_status(runner_id, generation, status.clone())
                    .await
                {
                    return Ok(false);
                }
            }
            registry.emit_command_response(runner_id, &response);
        }
        RunnerMessage::Register(_) => {
            tracing::warn!(
                "Unexpected Register message from {} after already registered",
                runner_id
            );
        }
    }

    Ok(true)
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
    sink.send(Message::Text(json.into())).await?;
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

    #[tokio::test]
    async fn stale_exit_preserves_replacement_and_status_requires_reconnect() {
        let registry = RunnerRegistry::new();
        let (tx, _rx) = mpsc::channel(32);
        let first = registry
            .register(
                "runner".into(),
                "First".into(),
                None,
                RunnerStatus::starting(),
                None,
                tx.clone(),
                None,
            )
            .await;
        let second = registry
            .register(
                "runner".into(),
                "Second".into(),
                None,
                RunnerStatus::starting(),
                None,
                tx,
                None,
            )
            .await;
        assert_ne!(first, second);
        let mut events = registry.subscribe_events();
        assert!(registry
            .unregister_connection("runner", first)
            .await
            .is_none());
        assert_eq!(registry.get("runner").await.unwrap().generation, second);
        assert!(matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
        let mut stale_status = RunnerStatus::starting();
        stale_status.health = simple_ai_common::RunnerHealth::Healthy;
        assert!(
            !registry
                .update_connection_status("runner", first, stale_status.clone())
                .await
        );
        assert_eq!(
            registry.get("runner").await.unwrap().status.health,
            simple_ai_common::RunnerHealth::Starting
        );
        for message in [
            RunnerMessage::Heartbeat(stale_status.clone()),
            RunnerMessage::StatusUpdate(stale_status),
        ] {
            let text = serde_json::to_string(&message).unwrap();
            // A false result causes the handler to send Close and exit.
            assert!(!handle_runner_message(&text, "runner", first, &registry)
                .await
                .unwrap());
            assert!(handle_runner_message(&text, "runner", second, &registry)
                .await
                .unwrap());
        }
        assert!(registry
            .unregister_connection("runner", second)
            .await
            .is_some());
        for message in [
            RunnerMessage::Heartbeat(RunnerStatus::starting()),
            RunnerMessage::StatusUpdate(RunnerStatus::starting()),
        ] {
            let text = serde_json::to_string(&message).unwrap();
            assert!(!handle_runner_message(&text, "runner", second, &registry)
                .await
                .unwrap());
        }
        assert!(matches!(
            events.recv().await.unwrap(),
            super::super::registry::RunnerEvent::StatusChanged { .. }
        ));
        assert!(matches!(
            events.recv().await.unwrap(),
            super::super::registry::RunnerEvent::Disconnected { .. }
        ));
        assert!(registry
            .unregister_connection("runner", second)
            .await
            .is_none());
        assert!(matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn starting_registration_preserves_catalog_until_status_reports_readiness() {
        let audit = AuditLogger::new(":memory:").unwrap();
        audit
            .upsert_runner(
                "fixture",
                "Fixture",
                None,
                None,
                Some(&["old-model".into()]),
            )
            .unwrap();
        let registration = RunnerRegistration::new(
            "fixture".into(),
            "Fixture".into(),
            None,
            8080,
            "token".into(),
            RunnerStatus::starting(),
        );
        persist_runner_status(&audit, &registration, None, &registration.status);
        assert_eq!(
            audit
                .get_runner("fixture")
                .unwrap()
                .unwrap()
                .available_models,
            vec!["old-model"]
        );
        let mut ready = RunnerStatus::starting();
        ready.health = simple_ai_common::RunnerHealth::Healthy;
        ready
            .model_aliases
            .insert("new-model".into(), "local-model".into());
        persist_runner_status(&audit, &registration, None, &ready);
        assert_eq!(
            audit
                .get_runner("fixture")
                .unwrap()
                .unwrap()
                .available_models,
            vec!["new-model"]
        );
        ready.model_aliases.clear();
        persist_runner_status(&audit, &registration, None, &ready);
        assert!(audit
            .get_runner("fixture")
            .unwrap()
            .unwrap()
            .available_models
            .is_empty());
    }

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
