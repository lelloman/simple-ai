//! Shared application state.

use std::sync::Arc;

use crate::config::Config;
use crate::engine::EngineRegistry;
use crate::ocr::OcrProvider;

/// Shared application state passed to all handlers.
pub struct AppState {
    /// Configuration (used in Phase 2+ for capability mappings)
    #[allow(dead_code)]
    pub config: Config,
    pub engine_registry: Arc<EngineRegistry>,
    pub ocr_provider: Option<Arc<dyn OcrProvider>>,
    pub decision_slots: Arc<tokio::sync::Semaphore>,
    pub decision_queue: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    pub fn new(
        config: Config,
        engine_registry: Arc<EngineRegistry>,
        ocr_provider: Option<Arc<dyn OcrProvider>>,
    ) -> Self {
        Self {
            config,
            engine_registry,
            ocr_provider,
            decision_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            decision_queue: Arc::new(tokio::sync::Semaphore::new(33)),
        }
    }
}
