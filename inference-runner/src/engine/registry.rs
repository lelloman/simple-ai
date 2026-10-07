//! Engine registry for managing multiple inference engines.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock as StdRwLock;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::InferenceEngine;
use crate::config::ModelRouteConfig;

/// Registry of all available inference engines.
///
/// The registry allows looking up engines by type or finding an engine
/// that can serve a particular model.
pub struct EngineRegistry {
    pub drain: Arc<crate::drain::Drain>,
    engines: RwLock<HashMap<String, Arc<dyn InferenceEngine>>>,
    routes: RwLock<HashMap<String, ModelRouteConfig>>,
    route_aliases: RwLock<HashMap<String, String>>,
    engine_resources: RwLock<HashMap<String, String>>,
    resource_gates: RwLock<HashMap<String, Arc<ResourceGate>>>,
}

struct ResourceGate {
    lock: Arc<RwLock<()>>,
    owner: StdRwLock<Option<String>>,
}

enum ResourceAccess {
    Shared(OwnedRwLockReadGuard<()>),
    Exclusive(OwnedRwLockWriteGuard<()>),
}

impl ResourceGate {
    fn new() -> Self {
        Self {
            lock: Arc::new(RwLock::new(())),
            owner: StdRwLock::new(None),
        }
    }

    fn is_owner(&self, owner_key: &str) -> bool {
        self.owner
            .read()
            .expect("resource owner lock poisoned")
            .as_deref()
            == Some(owner_key)
    }

    fn set_owner(&self, owner_key: String) {
        *self.owner.write().expect("resource owner lock poisoned") = Some(owner_key);
    }

    fn clear_owner(&self) {
        *self.owner.write().expect("resource owner lock poisoned") = None;
    }

    async fn acquire(&self, owner_key: &str) -> ResourceAccess {
        loop {
            if self.is_owner(owner_key) {
                let guard = self.lock.clone().read_owned().await;
                if self.is_owner(owner_key) {
                    return ResourceAccess::Shared(guard);
                }
                drop(guard);
                continue;
            }

            let guard = self.lock.clone().write_owned().await;
            if self.is_owner(owner_key) {
                return ResourceAccess::Shared(OwnedRwLockWriteGuard::downgrade(guard));
            }
            return ResourceAccess::Exclusive(guard);
        }
    }
}

/// Loaded model plus an optional exclusive resource guard. Keep this value
/// alive for the full inference response (including streaming bodies).
pub struct ModelLease {
    pub engine: Arc<dyn InferenceEngine>,
    pub engine_model: String,
    _resource_guard: Option<OwnedRwLockReadGuard<()>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            drain: Arc::new(crate::drain::Drain::default()),
            engines: RwLock::new(HashMap::new()),
            routes: RwLock::new(HashMap::new()),
            route_aliases: RwLock::new(HashMap::new()),
            engine_resources: RwLock::new(HashMap::new()),
            resource_gates: RwLock::new(HashMap::new()),
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
        let gates = resources
            .values()
            .map(|group| (group.clone(), Arc::new(ResourceGate::new())))
            .collect();
        *self.engine_resources.write().await = resources;
        *self.resource_gates.write().await = gates;
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
        let gate = if let Some(group) = &group {
            self.resource_gates.read().await.get(group).cloned()
        } else {
            None
        };
        let owner_key = format!("{}\0{}", target_type, engine_model);
        let resources = self.engine_resources.read().await.clone();
        let engines = self.all().await;
        let owned_target = target.clone();
        let owned_model = engine_model.clone();
        // An abandoned HTTP/WebSocket task must not release the gate while an
        // external process is still starting or stopping.
        let work = self.drain.track_existing();
        let resource_guard = tokio::spawn(async move {
            let _work = work;
            let target = owned_target;
            let engine_model = owned_model;
            let guard = match gate {
                Some(gate) => match gate.acquire(&owner_key).await {
                    ResourceAccess::Shared(guard) => {
                        target.load_model(&engine_model).await?;
                        Some(guard)
                    }
                    ResourceAccess::Exclusive(write_guard) => {
                        // A failed stop/start must not leave a stale shared-owner
                        // shortcut that bypasses cleanup on the next request.
                        gate.clear_owner();
                        if let Some(group) = group {
                            for engine in engines {
                                if engine.engine_type() == target_type
                                    || resources.get(engine.engine_type()) != Some(&group)
                                {
                                    continue;
                                }
                                engine.quiesce().await?;
                            }
                        }
                        target.load_model(&engine_model).await?;
                        gate.set_owner(owner_key);
                        Some(OwnedRwLockWriteGuard::downgrade(write_guard))
                    }
                },
                None => {
                    target.load_model(&engine_model).await?;
                    None
                }
            };
            Ok::<_, crate::error::Error>(guard)
        })
        .await
        .map_err(|e| crate::error::Error::Internal(e.to_string()))??;
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
    use std::time::Duration;

    struct SlowEngine {
        kind: &'static str,
        quiesced: Arc<std::sync::atomic::AtomicBool>,
        started: Arc<tokio::sync::Notify>,
        finish: Arc<tokio::sync::Notify>,
        loaded: std::sync::atomic::AtomicBool,
    }
    #[async_trait::async_trait]
    impl InferenceEngine for SlowEngine {
        fn engine_type(&self) -> &'static str {
            self.kind
        }
        async fn quiesce(&self) -> crate::error::Result<()> {
            if self.kind == "failing" { return Err(crate::error::Error::EngineNotAvailable("cannot stop".into())); }
            self.quiesced.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
        async fn health_check(&self) -> crate::error::Result<super::super::EngineHealth> {
            Ok(super::super::EngineHealth {
                is_healthy: true,
                version: None,
                models_loaded: vec![],
            })
        }
        async fn list_models(&self) -> crate::error::Result<Vec<super::super::ModelInfo>> {
            Ok(vec![])
        }
        async fn get_model(
            &self,
            _: &str,
        ) -> crate::error::Result<Option<super::super::ModelInfo>> {
            if self.kind != "slow" { return Ok(None); }
            Ok(Some(super::super::ModelInfo {
                id: "model".into(),
                name: "model".into(),
                size_bytes: None,
                parameter_count: None,
                context_length: None,
                quantization: None,
                modified_at: None,
                reasoning: None,
            }))
        }
        async fn load_model(&self, _: &str) -> crate::error::Result<()> {
            if !self.loaded.load(std::sync::atomic::Ordering::SeqCst) {
                self.started.notify_one();
                self.finish.notified().await;
                self.loaded.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(())
        }
        async fn unload_model(&self, _: &str) -> crate::error::Result<()> {
            Ok(())
        }
        async fn chat_completion(
            &self,
            _: &str,
            _: &simple_ai_common::ChatCompletionRequest,
        ) -> crate::error::Result<simple_ai_common::ChatCompletionResponse> {
            unimplemented!()
        }
        async fn chat_completion_stream(
            &self,
            _: &str,
            _: &simple_ai_common::ChatCompletionRequest,
        ) -> crate::error::Result<super::super::ChatCompletionStream> {
            unimplemented!()
        }
    }
    #[tokio::test]
    async fn cancelled_load_retains_resource_until_external_startup_finishes() {
        let registry = Arc::new(EngineRegistry::new());
        let engine = Arc::new(SlowEngine {
            kind: "slow", quiesced: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            started: Arc::new(tokio::sync::Notify::new()),
            finish: Arc::new(tokio::sync::Notify::new()),
            loaded: std::sync::atomic::AtomicBool::new(false),
        });
        registry.register(engine.clone()).await;
        registry
            .set_engine_resources(HashMap::from([("slow".into(), "gpu".into())]))
            .await;
        let r = registry.clone();
        let caller = tokio::spawn(async move { r.acquire_model("model").await });
        engine.started.notified().await;
        caller.abort();
        let gate = registry.resource_gates.read().await["gpu"].clone();
        assert!(
            tokio::time::timeout(Duration::from_millis(30), gate.lock.clone().write_owned())
                .await
                .is_err()
        );
        engine.finish.notify_one();
        let _guard = tokio::time::timeout(Duration::from_secs(1), gate.lock.clone().write_owned())
            .await
            .unwrap();
        assert!(engine.loaded.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn same_owner_revalidates_after_backend_crash() {
        let registry = Arc::new(EngineRegistry::new());
        let engine = Arc::new(SlowEngine {
            kind: "slow", quiesced: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            started: Arc::new(tokio::sync::Notify::new()),
            finish: Arc::new(tokio::sync::Notify::new()),
            loaded: std::sync::atomic::AtomicBool::new(true),
        });
        registry.register(engine.clone()).await;
        registry
            .set_engine_resources(HashMap::from([("slow".into(), "gpu".into())]))
            .await;
        drop(registry.acquire_model("model").await.unwrap());
        engine
            .loaded
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let r = registry.clone();
        let caller = tokio::spawn(async move { r.acquire_model("model").await });
        tokio::time::timeout(Duration::from_secs(1), engine.started.notified())
            .await
            .expect("shared owner must revalidate backend");
        engine.finish.notify_one();
        let lease = caller.await.unwrap().unwrap();
        assert!(engine.loaded.load(std::sync::atomic::Ordering::SeqCst));
        drop(lease);
    }

    #[tokio::test]
    async fn switching_quiesces_engines_without_healthy_loaded_models() {
        let registry = EngineRegistry::new();
        let quiesced = Arc::new(std::sync::atomic::AtomicBool::new(false));
        for kind in ["slow", "warming"] {
            registry.register(Arc::new(SlowEngine {
                kind, quiesced:quiesced.clone(),
                started:Arc::new(tokio::sync::Notify::new()), finish:Arc::new(tokio::sync::Notify::new()),
                loaded:std::sync::atomic::AtomicBool::new(true),
            })).await;
        }
        registry.set_engine_resources(HashMap::from([("slow".into(),"gpu".into()),("warming".into(),"gpu".into())])).await;
        let _lease = registry.acquire_model("model").await.unwrap();
        assert!(quiesced.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn failed_switch_clears_owner_so_next_request_cannot_bypass_cleanup() {
        let registry = EngineRegistry::new();
        registry.set_engine_resources(HashMap::from([("slow".into(),"gpu".into()),("failing".into(),"gpu".into())])).await;
        for kind in ["slow", "failing"] {
            registry.register(Arc::new(SlowEngine {
                kind, quiesced:Arc::new(std::sync::atomic::AtomicBool::new(false)),
                started:Arc::new(tokio::sync::Notify::new()), finish:Arc::new(tokio::sync::Notify::new()),
                loaded:std::sync::atomic::AtomicBool::new(true),
            })).await;
            if kind == "slow" { drop(registry.acquire_model("model").await.unwrap()); }
        }
        assert!(registry.acquire_model("other").await.is_err());
        let gate = registry.resource_gates.read().await["gpu"].clone();
        assert!(!gate.is_owner("slow\0model"));
        assert!(registry.acquire_model("model").await.is_err(), "must retry cleanup instead of using stale owner");
    }

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

    #[tokio::test]
    async fn resource_gate_shares_same_model_and_blocks_switches() {
        let gate = Arc::new(ResourceGate::new());
        let first = match gate.acquire("llama_cpp\0gemma").await {
            ResourceAccess::Exclusive(guard) => {
                gate.set_owner("llama_cpp\0gemma".to_string());
                OwnedRwLockWriteGuard::downgrade(guard)
            }
            ResourceAccess::Shared(_) => panic!("first owner must acquire exclusively"),
        };

        let second =
            match tokio::time::timeout(Duration::from_millis(50), gate.acquire("llama_cpp\0gemma"))
                .await
                .expect("same-model lease should not block")
            {
                ResourceAccess::Shared(guard) => guard,
                ResourceAccess::Exclusive(_) => panic!("same owner must share the resource"),
            };

        let switching_gate = gate.clone();
        let mut switch = tokio::spawn(async move { switching_gate.acquire("vllm\0qwen").await });
        assert!(
            tokio::time::timeout(Duration::from_millis(25), &mut switch)
                .await
                .is_err(),
            "different model must wait for active shared leases"
        );

        drop(first);
        drop(second);
        assert!(matches!(
            tokio::time::timeout(Duration::from_millis(100), switch)
                .await
                .expect("switch should proceed after leases drain")
                .expect("switch task should succeed"),
            ResourceAccess::Exclusive(_)
        ));
    }
}
