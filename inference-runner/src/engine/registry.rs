//! Engine registry for managing multiple inference engines.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedMutexGuard, RwLock};

use super::InferenceEngine;
use crate::config::ModelRouteConfig;

/// Registry of all available inference engines.
///
/// The registry allows looking up engines by type or finding an engine
/// that can serve a particular model.
pub struct EngineRegistry {
    engines: RwLock<HashMap<String, Arc<dyn InferenceEngine>>>,
    routes: RwLock<HashMap<String, ModelRouteConfig>>,
    route_aliases: RwLock<HashMap<String, String>>,
    engine_resources: RwLock<HashMap<String, String>>,
    resource_locks: RwLock<HashMap<String, Arc<Mutex<()>>>>,
}

/// Loaded model plus an optional exclusive resource guard. Keep this value
/// alive for the full inference response (including streaming bodies).
pub struct ModelLease {
    pub engine: Arc<dyn InferenceEngine>,
    pub engine_model: String,
    _resource_guard: Option<OwnedMutexGuard<()>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: RwLock::new(HashMap::new()),
            routes: RwLock::new(HashMap::new()),
            route_aliases: RwLock::new(HashMap::new()),
            engine_resources: RwLock::new(HashMap::new()),
            resource_locks: RwLock::new(HashMap::new()),
        }
    }

    pub async fn configure_routes(
        &self,
        routes: HashMap<String, ModelRouteConfig>,
    ) -> Result<(), String> {
        let mut aliases = HashMap::new();
        for (canonical, route) in &routes {
            if canonical.trim().is_empty()
                || route.engine.trim().is_empty()
                || route.engine_model.trim().is_empty()
            {
                return Err(format!("invalid model route for '{canonical}'"));
            }
            for alias in &route.aliases {
                if routes.contains_key(alias)
                    || aliases.insert(alias.clone(), canonical.clone()).is_some()
                {
                    return Err(format!("duplicate model route alias '{alias}'"));
                }
            }
        }
        *self.routes.write().await = routes;
        *self.route_aliases.write().await = aliases;
        Ok(())
    }

    pub async fn resolve_engine_for_model(
        &self,
        model_id: &str,
    ) -> Option<(Arc<dyn InferenceEngine>, String)> {
        let canonical = self
            .route_aliases
            .read()
            .await
            .get(model_id)
            .cloned()
            .unwrap_or_else(|| model_id.to_string());
        if let Some(route) = self.routes.read().await.get(&canonical).cloned() {
            let engine = self.get(&route.engine).await?;
            return engine
                .get_model(&route.engine_model)
                .await
                .ok()
                .flatten()
                .map(|_| (engine, route.engine_model));
        }

        let engines: Vec<Arc<dyn InferenceEngine>> = {
            let guard = self.engines.read().await;
            guard.values().cloned().collect()
        };
        let mut found = None;
        for engine in engines {
            if let Ok(Some(_)) = engine.get_model(model_id).await {
                if found.is_some() {
                    tracing::error!(
                        "model '{}' is claimed by multiple engines; add model_routes",
                        model_id
                    );
                    return None;
                }
                found = Some((engine, model_id.to_string()));
            }
        }
        found
    }

    /// Public names advertised to the gateway mapped directly to the local
    /// engine model. Gateway aliases are single-hop, so aliases cannot point at
    /// the canonical name here.
    pub async fn gateway_model_aliases(&self) -> HashMap<String, String> {
        let routes = self.routes.read().await;
        let mut mappings = HashMap::new();
        for (canonical, route) in routes.iter() {
            mappings.insert(canonical.clone(), route.engine_model.clone());
            for alias in &route.aliases {
                mappings.insert(alias.clone(), route.engine_model.clone());
            }
        }
        mappings
    }

    /// Register a new engine.
    pub async fn register(&self, engine: Arc<dyn InferenceEngine>) {
        let mut engines = self.engines.write().await;
        engines.insert(engine.engine_type().to_string(), engine);
    }

    pub async fn set_engine_resources(&self, resources: HashMap<String, String>) {
        let locks = resources
            .values()
            .map(|group| (group.clone(), Arc::new(Mutex::new(()))))
            .collect();
        *self.engine_resources.write().await = resources;
        *self.resource_locks.write().await = locks;
    }

    pub async fn load_model(&self, model_id: &str) -> crate::error::Result<()> {
        self.acquire_model(model_id).await.map(|_| ())
    }

    pub async fn acquire_model(&self, model_id: &str) -> crate::error::Result<ModelLease> {
        let (target, engine_model) = self
            .resolve_engine_for_model(model_id)
            .await
            .ok_or_else(|| crate::error::Error::ModelNotFound(model_id.to_string()))?;
        let target_type = target.engine_type().to_string();
        let group = self
            .engine_resources
            .read()
            .await
            .get(&target_type)
            .cloned();
        let lock = if let Some(group) = &group {
            self.resource_locks.read().await.get(group).cloned()
        } else {
            None
        };
        let resource_guard = match lock {
            Some(lock) => Some(lock.lock_owned().await),
            None => None,
        };

        if let Some(group) = group {
            let resources = self.engine_resources.read().await.clone();
            for engine in self.all().await {
                if engine.engine_type() == target_type
                    || resources.get(engine.engine_type()) != Some(&group)
                {
                    continue;
                }
                if let Ok(health) = engine.health_check().await {
                    for loaded in health.models_loaded {
                        engine.unload_model(&loaded).await?;
                    }
                }
            }
        }
        target.load_model(&engine_model).await?;
        Ok(ModelLease {
            engine: target,
            engine_model,
            _resource_guard: resource_guard,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn route(engine_model: &str, aliases: &[&str]) -> ModelRouteConfig {
        ModelRouteConfig {
            engine: "vllm".to_string(),
            engine_model: engine_model.to_string(),
            aliases: aliases.iter().map(|alias| (*alias).to_string()).collect(),
        }
    }

    #[tokio::test]
    async fn gateway_routes_are_advertised_as_single_hop_local_aliases() {
        let registry = EngineRegistry::new();
        registry
            .configure_routes(HashMap::from([(
                "qwen3.8-27b".to_string(),
                route("qwen38-fast", &["qwen3.8-27b-uncensored"]),
            )]))
            .await
            .unwrap();

        let aliases = registry.gateway_model_aliases().await;
        assert_eq!(aliases["qwen3.8-27b"], "qwen38-fast");
        assert_eq!(aliases["qwen3.8-27b-uncensored"], "qwen38-fast");
    }

    #[tokio::test]
    async fn duplicate_route_aliases_are_rejected() {
        let registry = EngineRegistry::new();
        let error = registry
            .configure_routes(HashMap::from([
                ("first".to_string(), route("local-a", &["duplicate"])),
                ("second".to_string(), route("local-b", &["duplicate"])),
            ]))
            .await
            .unwrap_err();

        assert!(error.contains("duplicate model route alias"));
    }
}
