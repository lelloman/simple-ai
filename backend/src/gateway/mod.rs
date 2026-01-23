//! Gateway module for managing inference runners.
//!
//! This module provides:
//! - WebSocket server for runner connections
//! - Runner registry for tracking connected runners
//! - Load balancer for routing inference requests
//! - Request proxying to runners
//! - Model classification for routing and permissions

pub mod model_class;
mod registry;
pub mod router;
mod ws;

pub use model_class::{classify_model, can_request_model, ModelClass, ModelRequest};
pub use registry::{ConnectedRunner, RunnerRegistry};
pub use router::{InferenceRouter, RouterError};
pub use ws::{ws_handler, WsState};
