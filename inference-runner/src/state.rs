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
        }
    }
}
