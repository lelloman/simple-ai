//! Shared application state.

use std::sync::Arc;

use crate::config::Config;
use crate::engine::EngineRegistry;

/// Shared application state passed to all handlers.
pub struct AppState {
    /// Configuration (used in Phase 2+ for capability mappings)
    #[allow(dead_code)]
    pub config: Config,
    pub engine_registry: Arc<EngineRegistry>,
}

impl AppState {
    pub fn new(config: Config, engine_registry: Arc<EngineRegistry>) -> Self {
        Self {
            config,
            engine_registry,
        }
    }
}
