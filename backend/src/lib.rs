pub mod arp;
pub mod audit;
pub mod auth;
pub mod circuit_breaker;
pub mod config;
pub mod gateway;
pub mod llm;
pub mod logging;
pub mod models;
pub mod rate_limit;
pub mod routes;
pub mod test_util;
pub mod wol;

pub use audit::AuditLogger;
pub use audit::{DashboardStats, RequestSummary, RequestWithResponse, UserWithStats};
pub use auth::AuthUser;
pub use auth::JwksClient;
pub use circuit_breaker::CircuitBreaker;
pub use config::{Config, GatewayConfig, ModelsConfig, WolConfig};
pub use gateway::{
    BatchDispatcher, BatchQueue, BatchQueueConfig, InferenceRouter, RequestScheduler,
    RouterTelemetry, RunnerRegistry,
};
pub use llm::OllamaClient;
pub use models::chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolCall, ToolFunction,
};
pub use routes::language::{DetectLanguageRequest, DetectLanguageResponse};
pub use wol::WakeService;

use fasttext::FastText;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

/// Event emitted when a new request is completed.
#[derive(Debug, Clone, Serialize)]
pub struct RequestEvent {
    pub id: String,
    pub timestamp: String,
    pub user_id: String,
    pub user_email: Option<String>,
    pub request_path: String,
    pub model: Option<String>,
    pub client_ip: Option<String>,
    pub status: Option<i32>,
    pub latency_ms: Option<i64>,
    pub tokens_prompt: Option<i64>,
    pub tokens_completion: Option<i64>,
    pub runner_id: Option<String>,
    pub wol_sent: bool,
}

/// Shared application state.
pub struct AppState {
    pub config: Config,
    pub jwks_client: JwksClient,
    pub ollama_client: OllamaClient,
    pub audit_logger: Arc<AuditLogger>,
    pub lang_detector: Mutex<FastText>,
    /// Runner registry for connected inference runners.
    pub runner_registry: Arc<RunnerRegistry>,
    /// Inference router for distributing requests.
    pub inference_router: Arc<InferenceRouter>,
    /// Central scheduler for request preparation and dispatch.
    pub request_scheduler: Arc<RequestScheduler>,
    /// Wake-on-LAN configuration.
    pub wol_config: WolConfig,
    /// Wake service for on-demand runner waking.
    pub wake_service: Arc<WakeService>,
    /// Broadcast channel for request events (for admin dashboard).
    pub request_events: broadcast::Sender<RequestEvent>,
    /// Router telemetry for scheduler state and recent events.
    pub router_telemetry: Arc<RouterTelemetry>,
    /// Batch queue for request batching (if enabled).
    pub batch_queue: Option<Arc<BatchQueue>>,
    /// Batch dispatcher for cache invalidation (if enabled).
    pub batch_dispatcher: Option<Arc<BatchDispatcher>>,
    /// Circuit breaker for failing backends.
    pub circuit_breaker: Arc<CircuitBreaker>,
}
