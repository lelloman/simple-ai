//! Gateway module for managing inference runners.
//!
//! This module provides:
//! - WebSocket server for runner connections
//! - Runner registry for tracking connected runners
//! - Load balancer for routing inference requests
//! - Request proxying to runners
//! - Model classification for routing and permissions
//! - Request batching for improved throughput

pub mod batch_dispatcher;
pub mod batch_queue;
pub mod model_class;
mod registry;
pub mod router;
pub mod scheduler;
pub mod telemetry;
mod ws;

pub use batch_dispatcher::BatchDispatcher;
pub use batch_queue::{BatchQueue, BatchQueueConfig, BatchedResponse, ModelQueueStats};
pub use model_class::{can_request_model, classify_model, ModelClass, ModelRequest};
pub use registry::{ConnectedRunner, RunnerEvent, RunnerRegistry};
pub use router::{InferenceRouter, RoutePlan, RoutedResponse, RouterError};
pub use scheduler::{RequestScheduler, ScheduledResponse, SchedulerError};
pub use telemetry::{RouterEventRecord, RouterStateSnapshot, RouterTelemetry};
pub use ws::{ws_handler, WsState};
