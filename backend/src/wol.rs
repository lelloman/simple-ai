//! Wake-on-LAN (WOL) implementation and WakeService.
//!
//! Sends magic packets to wake up offline machines.
//! Supports direct UDP broadcast or sending via a bouncer service (for Docker).
//! WakeService provides wake-on-demand functionality for the gateway.

use std::net::UdpSocket;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::time::timeout;

use crate::audit::{AuditLogger, RunnerRecord};
use crate::config::{GatewayConfig, WolConfig};
use crate::config::ModelsConfig;
use crate::gateway::{classify_model, ModelRequest, RunnerEvent, RunnerRegistry};

/// Errors that can occur during WOL operations.
#[derive(Debug, thiserror::Error)]
pub enum WolError {
    #[error("Invalid MAC address format: {0}")]
    InvalidMacAddress(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Bouncer error: {0}")]
    BouncerError(String),
}

/// Parse a MAC address string (AA:BB:CC:DD:EE:FF) into bytes.
fn parse_mac_address(mac: &str) -> Result<[u8; 6], WolError> {
    let parts: Vec<&str> = mac.split(':').collect();
    if parts.len() != 6 {
        return Err(WolError::InvalidMacAddress(format!(
            "Expected 6 octets separated by ':', got {}",
            parts.len()
        )));
    }

    let mut bytes = [0u8; 6];
    for (i, part) in parts.iter().enumerate() {
        bytes[i] = u8::from_str_radix(part, 16).map_err(|_| {
            WolError::InvalidMacAddress(format!("Invalid hex octet: {}", part))
        })?;
    }

    Ok(bytes)
}

/// Build a Wake-on-LAN magic packet.
///
/// The magic packet consists of:
/// - 6 bytes of 0xFF
/// - 16 repetitions of the target MAC address (96 bytes)
///
/// Total: 102 bytes
fn build_magic_packet(mac_bytes: &[u8; 6]) -> [u8; 102] {
    let mut packet = [0u8; 102];

    // First 6 bytes are 0xFF
    for byte in packet.iter_mut().take(6) {
        *byte = 0xFF;
    }

    // Repeat MAC address 16 times
    for i in 0..16 {
        let offset = 6 + (i * 6);
        packet[offset..offset + 6].copy_from_slice(mac_bytes);
    }

    packet
}

/// Send a Wake-on-LAN magic packet to wake a machine.
///
/// # Arguments
/// * `mac_address` - Target MAC address in format AA:BB:CC:DD:EE:FF
/// * `broadcast_addr` - Broadcast address to send to (e.g., "255.255.255.255")
/// * `port` - UDP port (typically 9 or 7)
///
/// # Example
/// ```ignore
/// send_wol("AA:BB:CC:DD:EE:FF", "255.255.255.255", 9)?;
/// ```
pub fn send_wol(mac_address: &str, broadcast_addr: &str, port: u16) -> Result<(), WolError> {
    let mac_bytes = parse_mac_address(mac_address)?;
    let packet = build_magic_packet(&mac_bytes);

    let socket = UdpSocket::bind("0.0.0.0:0")
        .map_err(|e| WolError::NetworkError(format!("Failed to bind socket: {}", e)))?;

    socket
        .set_broadcast(true)
        .map_err(|e| WolError::NetworkError(format!("Failed to enable broadcast: {}", e)))?;

    let dest = format!("{}:{}", broadcast_addr, port);
    socket
        .send_to(&packet, &dest)
        .map_err(|e| WolError::NetworkError(format!("Failed to send packet: {}", e)))?;

    tracing::info!(
        "Sent WOL magic packet for MAC {} to {}:{}",
        mac_address,
        broadcast_addr,
        port
    );

    Ok(())
}

/// Send WOL via bouncer service (TCP).
///
/// # Arguments
/// * `bouncer_addr` - Address of the bouncer service (e.g., "localhost:9999")
/// * `mac_address` - Target MAC address in format AA:BB:CC:DD:EE:FF
pub async fn send_wol_via_bouncer(
    bouncer_addr: &str,
    mac_address: &str,
    _broadcast_addr: &str,
) -> Result<(), WolError> {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;

    // Strip protocol prefix if present
    let addr = bouncer_addr
        .trim_start_matches("tcp://")
        .trim_start_matches("http://");

    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|e| WolError::BouncerError(format!("Failed to connect to bouncer at {}: {}", addr, e)))?;

    stream
        .write_all(format!("{}\n", mac_address).as_bytes())
        .await
        .map_err(|e| WolError::BouncerError(format!("Failed to send MAC to bouncer: {}", e)))?;

    tracing::info!("Sent WOL via bouncer for MAC {}", mac_address);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mac_address_valid() {
        let mac = parse_mac_address("AA:BB:CC:DD:EE:FF").unwrap();
        assert_eq!(mac, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_parse_mac_address_lowercase() {
        let mac = parse_mac_address("aa:bb:cc:dd:ee:ff").unwrap();
        assert_eq!(mac, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_parse_mac_address_mixed_case() {
        let mac = parse_mac_address("Aa:Bb:Cc:Dd:Ee:Ff").unwrap();
        assert_eq!(mac, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_parse_mac_address_zeros() {
        let mac = parse_mac_address("00:00:00:00:00:00").unwrap();
        assert_eq!(mac, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_parse_mac_address_invalid_too_short() {
        let result = parse_mac_address("AA:BB:CC:DD:EE");
        assert!(matches!(result, Err(WolError::InvalidMacAddress(_))));
    }

    #[test]
    fn test_parse_mac_address_invalid_too_long() {
        let result = parse_mac_address("AA:BB:CC:DD:EE:FF:00");
        assert!(matches!(result, Err(WolError::InvalidMacAddress(_))));
    }

    #[test]
    fn test_parse_mac_address_invalid_hex() {
        let result = parse_mac_address("GG:BB:CC:DD:EE:FF");
        assert!(matches!(result, Err(WolError::InvalidMacAddress(_))));
    }

    #[test]
    fn test_parse_mac_address_wrong_delimiter() {
        let result = parse_mac_address("AA-BB-CC-DD-EE-FF");
        assert!(matches!(result, Err(WolError::InvalidMacAddress(_))));
    }

    #[test]
    fn test_build_magic_packet() {
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let packet = build_magic_packet(&mac);

        // Check length
        assert_eq!(packet.len(), 102);

        // Check first 6 bytes are 0xFF
        assert_eq!(&packet[0..6], &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

        // Check MAC is repeated 16 times
        for i in 0..16 {
            let offset = 6 + (i * 6);
            assert_eq!(&packet[offset..offset + 6], &mac);
        }
    }

    #[test]
    fn test_wol_error_display() {
        let err = WolError::InvalidMacAddress("test".to_string());
        assert!(err.to_string().contains("Invalid MAC address"));

        let err = WolError::NetworkError("test".to_string());
        assert!(err.to_string().contains("Network error"));
    }
}

// ============================================================================
// WakeService - Wake-on-demand functionality
// ============================================================================

/// Errors that can occur during wake operations.
#[derive(Debug, thiserror::Error)]
pub enum WakeError {
    #[error("No wakeable runners configured")]
    NoWakeableRunners,
    #[error("Failed to wake runners: {0}")]
    WakeFailed(String),
    #[error("Timeout waiting for runner to connect after {0}s")]
    Timeout(u64),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Idle manager error: {0}")]
    IdleManagerError(String),
}

/// Result of a successful wake operation.
#[derive(Debug)]
pub struct WakeResult {
    /// ID of the runner that connected.
    pub runner_id: String,
    /// How long we waited for the runner to connect.
    pub wait_duration: Duration,
    /// Number of runners we attempted to wake.
    pub runners_woken: usize,
}

/// Service for waking offline runners on demand.
pub struct WakeService {
    registry: Arc<RunnerRegistry>,
    audit_logger: Arc<AuditLogger>,
    gateway_config: GatewayConfig,
    wol_config: WolConfig,
    models_config: ModelsConfig,
    http_client: reqwest::Client,
}

impl WakeService {
    /// Create a new WakeService.
    pub fn new(
        registry: Arc<RunnerRegistry>,
        audit_logger: Arc<AuditLogger>,
        gateway_config: GatewayConfig,
        wol_config: WolConfig,
        models_config: ModelsConfig,
    ) -> Self {
        Self {
            registry,
            audit_logger,
            gateway_config,
            wol_config,
            models_config,
            http_client: reqwest::Client::new(),
        }
    }

    /// Check if auto-wake is enabled.
    pub fn is_enabled(&self) -> bool {
        self.gateway_config.auto_wake_enabled
    }

    /// Find runners that are offline but have MAC addresses (wakeable).
    ///
    /// Returns runners from the database that:
    /// 1. Have a MAC address configured
    /// 2. Are NOT currently connected (not in live registry)
    /// 3. Have models matching the requested model/class (if specified)
    pub async fn find_wakeable_runners(
        &self,
        model_request: Option<&ModelRequest>,
    ) -> Result<Vec<RunnerRecord>, WakeError> {
        // Get all runners from the database
        let all_runners = self
            .audit_logger
            .get_all_runners()
            .map_err(|e| WakeError::DatabaseError(e.to_string()))?;

        // Get currently connected runner IDs
        let connected = self.registry.all().await;
        let connected_ids: std::collections::HashSet<_> =
            connected.iter().map(|r| r.id.as_str()).collect();

        // Filter to runners that are offline and have MAC addresses
        let mut wakeable: Vec<RunnerRecord> = all_runners
            .into_iter()
            .filter(|r| r.mac_address.is_some() && !connected_ids.contains(r.id.as_str()))
            .collect();

        // Further filter by model request if specified
        if let Some(request) = model_request {
            wakeable = wakeable
                .into_iter()
                .filter(|r| self.runner_matches_request(r, request))
                .collect();
        }

        // Sort by last_seen_at descending (most recently seen first)
        wakeable.sort_by(|a, b| b.last_seen_at.cmp(&a.last_seen_at));

        Ok(wakeable)
    }

    /// Check if a runner has models matching the request.
    fn runner_matches_request(&self, runner: &RunnerRecord, request: &ModelRequest) -> bool {
        match request {
            ModelRequest::Specific(model_id) => {
                // Check if runner has this specific model
                runner.available_models.iter().any(|m| m == model_id)
            }
            ModelRequest::Class(class) => {
                // Check if runner has any model of this class
                runner
                    .available_models
                    .iter()
                    .any(|m| classify_model(m, &self.models_config) == Some(*class))
            }
        }
    }

    /// Find the best runner to wake for a model request.
    ///
    /// Returns the most recently seen runner that:
    /// 1. Is offline
    /// 2. Has a MAC address
    /// 3. Has models matching the request
    pub async fn find_best_runner_to_wake(
        &self,
        model_request: &ModelRequest,
    ) -> Result<Option<RunnerRecord>, WakeError> {
        let wakeable = self.find_wakeable_runners(Some(model_request)).await?;
        Ok(wakeable.into_iter().next())
    }

    /// Wake a single runner by MAC address.
    ///
    /// Tries idle-manager first (if configured), falls back to direct WOL.
    async fn wake_runner(&self, runner: &RunnerRecord) -> Result<(), WakeError> {
        let mac = runner
            .mac_address
            .as_ref()
            .ok_or_else(|| WakeError::WakeFailed("No MAC address".to_string()))?;

        // Try idle-manager first if configured
        if let Some(ref idle_manager_url) = self.gateway_config.idle_manager_url {
            match self.wake_via_idle_manager(idle_manager_url, &runner.id).await {
                Ok(()) => {
                    tracing::info!(
                        "Woke runner {} via idle-manager",
                        runner.id
                    );
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to wake {} via idle-manager ({}), falling back to direct WOL",
                        runner.id,
                        e
                    );
                }
            }
        }

        // Fall back to direct WOL
        if let Some(ref bouncer_url) = self.wol_config.bouncer_url {
            send_wol_via_bouncer(bouncer_url, mac, &self.wol_config.broadcast_address)
                .await
                .map_err(|e| WakeError::WakeFailed(e.to_string()))?;
        } else {
            send_wol(mac, &self.wol_config.broadcast_address, self.wol_config.port)
                .map_err(|e| WakeError::WakeFailed(e.to_string()))?;
        }

        tracing::info!("Sent WOL packet to runner {} (MAC: {})", runner.id, mac);
        Ok(())
    }

    /// Wake runner via idle-manager service.
    async fn wake_via_idle_manager(
        &self,
        idle_manager_url: &str,
        runner_id: &str,
    ) -> Result<(), WakeError> {
        let url = format!("{}/wake/{}", idle_manager_url.trim_end_matches('/'), runner_id);

        let response = self
            .http_client
            .post(&url)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| WakeError::IdleManagerError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(WakeError::IdleManagerError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        Ok(())
    }

    /// Wake a runner that can handle the model request and wait for it to connect.
    ///
    /// This is the main entry point for wake-on-demand.
    ///
    /// # Algorithm:
    /// 1. Find the best wakeable runner for the model request
    /// 2. Subscribe to registry connection events BEFORE waking
    /// 3. Wake the runner
    /// 4. Wait for it to connect or timeout
    pub async fn wake_and_wait(&self, model_request: &ModelRequest) -> Result<WakeResult, WakeError> {
        let start = std::time::Instant::now();

        // 1. Find the best runner for this request
        let runner = self
            .find_best_runner_to_wake(model_request)
            .await?
            .ok_or(WakeError::NoWakeableRunners)?;

        tracing::info!(
            "Found wakeable runner {} for {:?} (models: {:?})",
            runner.id,
            model_request,
            runner.available_models
        );

        // 2. Subscribe BEFORE waking to avoid race conditions
        let mut rx = self.registry.subscribe_events();

        // 3. Wake the runner
        self.wake_runner(&runner).await?;

        tracing::info!("Sent wake signal to runner {}, waiting for connection...", runner.id);

        // 4. Wait for a runner to connect (filter for Connected events only)
        let timeout_duration = Duration::from_secs(self.gateway_config.wake_timeout_secs);
        let runner_id = timeout(timeout_duration, async {
            loop {
                match rx.recv().await {
                    Ok(RunnerEvent::Connected { runner_id, .. }) => {
                        tracing::info!("Runner {} connected after wake", runner_id);
                        return runner_id;
                    }
                    Ok(_) => {
                        // Ignore other events (Disconnected, StatusChanged)
                        continue;
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Missed some events, continue waiting
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        // Channel closed, shouldn't happen
                        break String::new();
                    }
                }
            }
        })
        .await
        .map_err(|_| WakeError::Timeout(self.gateway_config.wake_timeout_secs))?;

        if runner_id.is_empty() {
            return Err(WakeError::Timeout(self.gateway_config.wake_timeout_secs));
        }

        let wait_duration = start.elapsed();
        tracing::info!(
            "Runner {} connected after {:.1}s",
            runner_id,
            wait_duration.as_secs_f64()
        );

        Ok(WakeResult {
            runner_id,
            wait_duration,
            runners_woken: 1,
        })
    }
}

#[cfg(test)]
mod wake_service_tests {
    use super::*;

    #[test]
    fn test_wake_error_display() {
        let err = WakeError::NoWakeableRunners;
        assert!(err.to_string().contains("No wakeable runners"));

        let err = WakeError::Timeout(90);
        assert!(err.to_string().contains("90s"));

        let err = WakeError::WakeFailed("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_wake_result_struct() {
        let result = WakeResult {
            runner_id: "runner-1".to_string(),
            wait_duration: Duration::from_secs(5),
            runners_woken: 2,
        };
        assert_eq!(result.runner_id, "runner-1");
        assert_eq!(result.runners_woken, 2);
    }
}
