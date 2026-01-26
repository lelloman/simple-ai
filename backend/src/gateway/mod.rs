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
mod ws;

pub use batch_dispatcher::BatchDispatcher;
pub use batch_queue::{BatchQueue, BatchQueueConfig, BatchedResponse, ModelQueueStats};
pub use model_class::{classify_model, can_request_model, ModelClass, ModelRequest};
pub use registry::{ConnectedRunner, RunnerEvent, RunnerRegistry};
pub use router::{InferenceRouter, RouterError, RoutedResponse};
pub use ws::{ws_handler, WsState};
