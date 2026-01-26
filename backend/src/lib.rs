pub mod config;
pub mod routes;
pub mod auth;
pub mod llm;
pub mod audit;
pub mod models;
pub mod logging;
pub mod gateway;
pub mod test_util;
pub mod wol;
pub mod arp;

pub use config::{Config, GatewayConfig, ModelsConfig, WolConfig};
pub use auth::JwksClient;
pub use llm::OllamaClient;
pub use audit::AuditLogger;
pub use models::chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolCall, ToolFunction};
pub use routes::language::{DetectLanguageRequest, DetectLanguageResponse};
pub use auth::AuthUser;
pub use audit::{DashboardStats, UserWithStats, RequestSummary, RequestWithResponse};
pub use gateway::{BatchDispatcher, BatchQueue, BatchQueueConfig, InferenceRouter, RunnerRegistry};
pub use wol::WakeService;

use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use fasttext::FastText;
use serde::Serialize;

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
    /// Wake-on-LAN configuration.
    pub wol_config: WolConfig,
    /// Wake service for on-demand runner waking.
    pub wake_service: Arc<WakeService>,
    /// Broadcast channel for request events (for admin dashboard).
    pub request_events: broadcast::Sender<RequestEvent>,
    /// Batch queue for request batching (if enabled).
    pub batch_queue: Option<Arc<BatchQueue>>,
}
