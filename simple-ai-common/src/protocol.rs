//! WebSocket protocol types for gateway-runner communication.
//!
//! This module defines the message format for real-time communication
//! between inference runners and the central gateway.
//!
//! # Protocol Overview
//!
//! The protocol uses JSON-encoded messages over WebSocket. Each message has a `type` field
//! that determines its structure.
//!
//! ## Connection Flow
//!
//! 1. Runner connects to gateway WebSocket endpoint
//! 2. Runner sends `Register` message with auth token and initial status
//! 3. Gateway responds with `RegisterAck` on success or `Error` on failure
//! 4. Runner sends periodic `Heartbeat` messages with current status
//! 5. Gateway may send commands (`LoadModel`, `UnloadModel`, `RequestStatus`)
//! 6. Runner responds to commands with `CommandResponse`
//!
//! ## Ping Mechanisms
//!
//! There are two ping mechanisms:
//! - **WebSocket ping/pong**: Transport-level keepalive, handled automatically
//! - **Application-level `Ping`**: Gateway requests status update, runner responds with `StatusUpdate`
//!
//! # Security
//!
//! - Always use `wss://` in production to encrypt the connection
//! - Auth tokens should be securely generated and rotated periodically

use serde::{Deserialize, Serialize};

use crate::CapabilityInfo;

/// Messages sent from runner to gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunnerMessage {
    /// Initial registration when connecting.
    Register(RunnerRegistration),
    /// Periodic heartbeat with current status.
    Heartbeat(RunnerStatus),
    /// Status update (capabilities changed, model loaded/unloaded, etc.).
    StatusUpdate(RunnerStatus),
    /// Response to a gateway command.
    CommandResponse(CommandResponse),
}

/// Messages sent from gateway to runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GatewayMessage {
    /// Acknowledgment of successful registration.
    RegisterAck { runner_id: String },
    /// Command to load a model.
    LoadModel { model_id: String, request_id: String },
    /// Command to unload a model.
    UnloadModel { model_id: String, request_id: String },
    /// Request current status.
    RequestStatus { request_id: String },
    /// Ping for connection health.
    Ping { timestamp: i64 },
    /// Error message from gateway.
    Error { code: String, message: String },
}

/// Runner registration data sent on connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerRegistration {
    /// Unique runner identifier.
    pub runner_id: String,
    /// Human-readable runner name.
    pub runner_name: String,
    /// Machine type for routing (e.g., "gpu-server", "strix-halo").
    #[serde(default)]
    pub machine_type: Option<String>,
    /// HTTP port the runner's API is listening on (default: 8080).
    #[serde(default = "default_http_port")]
    pub http_port: u16,
    /// Protocol version for compatibility checking.
    pub protocol_version: u32,
    /// Authentication token.
    pub auth_token: String,
    /// Initial status.
    pub status: RunnerStatus,
    /// MAC address for Wake-on-LAN (format: AA:BB:CC:DD:EE:FF).
    #[serde(default)]
    pub mac_address: Option<String>,
}

fn default_http_port() -> u16 {
    8080
}

/// Current status of a runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerStatus {
    /// Current health state.
    pub health: RunnerHealth,
    /// Available capabilities and their status.
    pub capabilities: Vec<CapabilityInfo>,
    /// Available engine types (e.g., ["ollama", "llama_cpp"]).
    pub engines: Vec<EngineStatus>,
    /// System metrics (optional).
    #[serde(default)]
    pub metrics: Option<RunnerMetrics>,
    /// Model aliases: maps canonical names to local engine names.
    #[serde(default)]
    pub model_aliases: std::collections::HashMap<String, String>,
}

/// Health state of the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerHealth {
    /// All engines healthy and ready.
    Healthy,
    /// Some engines have issues but still operational.
    Degraded,
    /// Runner is starting up.
    Starting,
    /// Runner is shutting down.
    ShuttingDown,
    /// Critical failure.
    Unhealthy,
}

/// Information about an available model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g., "llama3.2:3b")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model size in bytes (if known)
    #[serde(default)]
    pub size_bytes: Option<u64>,
    /// Parameter count (if known)
    #[serde(default)]
    pub parameter_count: Option<u64>,
    /// Maximum context length
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Quantization type (e.g., "Q4_K_M")
    #[serde(default)]
    pub quantization: Option<String>,
    /// When the model was last modified
    #[serde(default)]
    pub modified_at: Option<String>,
}

/// Status of a single inference engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    /// Engine type (e.g., "ollama", "llama_cpp").
    pub engine_type: String,
    /// Whether this engine is healthy.
    pub is_healthy: bool,
    /// Engine version if available.
    #[serde(default)]
    pub version: Option<String>,
    /// Models currently loaded in this engine.
    #[serde(default)]
    pub loaded_models: Vec<String>,
    /// All models available on disk through this engine.
    #[serde(default)]
    pub available_models: Vec<ModelInfo>,
    /// Error message if unhealthy.
    #[serde(default)]
    pub error: Option<String>,
}

/// System metrics for the runner.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunnerMetrics {
    /// Requests processed since last report.
    #[serde(default)]
    pub requests_processed: u64,
    /// Average latency in milliseconds.
    #[serde(default)]
    pub avg_latency_ms: Option<f64>,
    /// GPU memory used (bytes) if applicable.
    #[serde(default)]
    pub gpu_memory_used: Option<u64>,
    /// GPU memory total (bytes) if applicable.
    #[serde(default)]
    pub gpu_memory_total: Option<u64>,
    /// CPU usage percentage.
    #[serde(default)]
    pub cpu_usage_percent: Option<f32>,
    /// System memory used (bytes).
    #[serde(default)]
    pub memory_used: Option<u64>,
}

/// Response to a gateway command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResponse {
    /// ID of the request being responded to.
    pub request_id: String,
    /// Whether the command succeeded.
    pub success: bool,
    /// Error message if failed.
    #[serde(default)]
    pub error: Option<String>,
    /// Updated status after command execution.
    #[serde(default)]
    pub status: Option<RunnerStatus>,
}

/// Protocol version constant.
pub const PROTOCOL_VERSION: u32 = 1;

impl RunnerRegistration {
    pub fn new(
        runner_id: String,
        runner_name: String,
        machine_type: Option<String>,
        http_port: u16,
        auth_token: String,
        status: RunnerStatus,
    ) -> Self {
        Self {
            runner_id,
            runner_name,
            machine_type,
            http_port,
            protocol_version: PROTOCOL_VERSION,
            auth_token,
            status,
            mac_address: None,
        }
    }
}

impl RunnerStatus {
    /// Create an empty status indicating the runner is starting.
    pub fn starting() -> Self {
        Self {
            health: RunnerHealth::Starting,
            capabilities: vec![],
            engines: vec![],
            metrics: None,
            model_aliases: std::collections::HashMap::new(),
        }
    }
}

impl RunnerHealth {
    pub fn is_operational(&self) -> bool {
        matches!(self, RunnerHealth::Healthy | RunnerHealth::Degraded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_message_serialization() {
        let msg = RunnerMessage::Heartbeat(RunnerStatus::starting());
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""type":"heartbeat""#));
        assert!(json.contains(r#""health":"starting""#));
    }

    #[test]
    fn test_gateway_message_serialization() {
        let msg = GatewayMessage::LoadModel {
            model_id: "llama3.2:3b".to_string(),
            request_id: "req-123".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""type":"load_model""#));
        assert!(json.contains(r#""model_id":"llama3.2:3b""#));
    }

    #[test]
    fn test_runner_registration_new() {
        let status = RunnerStatus::starting();
        let reg = RunnerRegistration::new(
            "runner-1".to_string(),
            "My Runner".to_string(),
            Some("gpu-server".to_string()),
            8080,
            "secret".to_string(),
            status,
        );
        assert_eq!(reg.protocol_version, PROTOCOL_VERSION);
        assert_eq!(reg.runner_id, "runner-1");
        assert_eq!(reg.http_port, 8080);
    }

    #[test]
    fn test_runner_health_is_operational() {
        assert!(RunnerHealth::Healthy.is_operational());
        assert!(RunnerHealth::Degraded.is_operational());
        assert!(!RunnerHealth::Starting.is_operational());
        assert!(!RunnerHealth::Unhealthy.is_operational());
    }

    #[test]
    fn test_engine_status_serialization() {
        let status = EngineStatus {
            engine_type: "ollama".to_string(),
            is_healthy: true,
            version: Some("0.5.0".to_string()),
            loaded_models: vec!["llama3.2:3b".to_string()],
            available_models: vec![],
            error: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains(r#""engine_type":"ollama""#));
        assert!(json.contains(r#""is_healthy":true"#));
    }

    #[test]
    fn test_command_response_success() {
        let resp = CommandResponse {
            request_id: "req-456".to_string(),
            success: true,
            error: None,
            status: Some(RunnerStatus::starting()),
        };
        assert!(resp.success);
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_command_response_failure() {
        let resp = CommandResponse {
            request_id: "req-789".to_string(),
            success: false,
            error: Some("Model not found".to_string()),
            status: None,
        };
        assert!(!resp.success);
        assert_eq!(resp.error.as_deref(), Some("Model not found"));
    }

    #[test]
    fn test_runner_metrics_default() {
        let metrics = RunnerMetrics::default();
        assert_eq!(metrics.requests_processed, 0);
        assert!(metrics.avg_latency_ms.is_none());
    }

    #[test]
    fn test_gateway_error_message() {
        let msg = GatewayMessage::Error {
            code: "AUTH_FAILED".to_string(),
            message: "Invalid token".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: GatewayMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            GatewayMessage::Error { code, message } => {
                assert_eq!(code, "AUTH_FAILED");
                assert_eq!(message, "Invalid token");
            }
            _ => panic!("Expected Error message"),
        }
    }

    #[test]
    fn test_runner_message_register_roundtrip() {
        let status = RunnerStatus::starting();
        let reg = RunnerRegistration::new(
            "runner-1".to_string(),
            "Test Runner".to_string(),
            Some("gpu".to_string()),
            8080,
            "token".to_string(),
            status,
        );
        let msg = RunnerMessage::Register(reg);
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: RunnerMessage = serde_json::from_str(&json).unwrap();

        match parsed {
            RunnerMessage::Register(r) => {
                assert_eq!(r.runner_id, "runner-1");
                assert_eq!(r.http_port, 8080);
                assert_eq!(r.protocol_version, PROTOCOL_VERSION);
            }
            _ => panic!("Expected Register message"),
        }
    }

    #[test]
    fn test_runner_status_with_capabilities() {
        use crate::{Capability, CapabilityStatus};

        let status = RunnerStatus {
            health: RunnerHealth::Healthy,
            capabilities: vec![CapabilityInfo {
                capability: Capability::FastChat,
                status: CapabilityStatus::Loaded,
                model_id: "llama3.2:3b".to_string(),
                active_requests: 5,
                avg_latency_ms: Some(150.5),
            }],
            engines: vec![EngineStatus {
                engine_type: "ollama".to_string(),
                is_healthy: true,
                version: Some("0.5.0".to_string()),
                loaded_models: vec!["llama3.2:3b".to_string()],
                available_models: vec![],
                error: None,
            }],
            metrics: Some(RunnerMetrics {
                requests_processed: 100,
                avg_latency_ms: Some(150.5),
                gpu_memory_used: Some(1024 * 1024 * 1024),
                gpu_memory_total: Some(8 * 1024 * 1024 * 1024),
                cpu_usage_percent: Some(45.5),
                memory_used: Some(4 * 1024 * 1024 * 1024),
            }),
            model_aliases: std::collections::HashMap::new(),
        };

        let json = serde_json::to_string(&status).unwrap();
        let parsed: RunnerStatus = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.health, RunnerHealth::Healthy);
        assert_eq!(parsed.capabilities.len(), 1);
        assert_eq!(parsed.engines.len(), 1);
        assert!(parsed.metrics.is_some());
    }

    #[test]
    fn test_shutting_down_health_not_operational() {
        assert!(!RunnerHealth::ShuttingDown.is_operational());
    }
}
