//! Engine registry for managing multiple inference engines.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::InferenceEngine;

/// Registry of all available inference engines.
///
/// The registry allows looking up engines by type or finding an engine
/// that can serve a particular model.
pub struct EngineRegistry {
    engines: RwLock<HashMap<String, Arc<dyn InferenceEngine>>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new engine.
    pub async fn register(&self, engine: Arc<dyn InferenceEngine>) {
        let mut engines = self.engines.write().await;
        engines.insert(engine.engine_type().to_string(), engine);
    }

    /// Get an engine by type (Phase 2 - for targeted engine operations).
    #[allow(dead_code)]
    pub async fn get(&self, engine_type: &str) -> Option<Arc<dyn InferenceEngine>> {
        let engines = self.engines.read().await;
        engines.get(engine_type).cloned()
    }

    /// Get all registered engines.
    pub async fn all(&self) -> Vec<Arc<dyn InferenceEngine>> {
        let engines = self.engines.read().await;
        engines.values().cloned().collect()
    }

    /// Find an engine that has a specific model available.
    ///
    /// Searches all registered engines and returns the first one
    /// that reports having the model.
    pub async fn find_engine_for_model(
        &self,
        model_id: &str,
    ) -> Option<Arc<dyn InferenceEngine>> {
        // Clone engines to avoid holding the lock across async calls
        let engines: Vec<Arc<dyn InferenceEngine>> = {
            let guard = self.engines.read().await;
            guard.values().cloned().collect()
        };

        for engine in engines {
            if let Ok(Some(_)) = engine.get_model(model_id).await {
                return Some(engine);
            }
        }
        None
    }

    /// Get the first available engine (convenience method for single-engine setups).
    #[allow(dead_code)]
    pub async fn first(&self) -> Option<Arc<dyn InferenceEngine>> {
        let engines = self.engines.read().await;
        engines.values().next().cloned()
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}
