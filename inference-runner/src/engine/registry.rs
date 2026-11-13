//! Engine registry for managing multiple inference engines.

use std::collections::{HashMap, VecDeque};
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
    batches: std::sync::Mutex<BatchQueue>,
    changed: tokio::sync::Notify,
}

#[derive(Default)]
struct BatchQueue {
    next_id: u64,
    waiting: VecDeque<(u64, String, bool)>,
}

// A reservation stays in the queue through startup AND inference. Dropping a
// waiting caller removes it, while detached external operations retain it.
struct BatchPermit {
    gate: Arc<ResourceGate>,
    id: u64,
}

impl Drop for BatchPermit {
    fn drop(&mut self) {
        self.gate
            .batches
            .lock()
            .unwrap()
            .waiting
            .retain(|entry| entry.0 != self.id);
        self.gate.changed.notify_waiters();
    }
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
            batches: Default::default(),
            changed: tokio::sync::Notify::new(),
        }
    }

    async fn batch(self: &Arc<Self>, owner_key: &str) -> BatchPermit {
        let id = {
            let mut queue = self.batches.lock().unwrap();
            let id = queue.next_id;
            queue.next_id += 1;
            queue.waiting.push_back((id, owner_key.to_owned(), false));
            id
        };
        let permit = BatchPermit {
            gate: self.clone(),
            id,
        };
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let mut queue = self.batches.lock().unwrap();
                let active_owner = queue
                    .waiting
                    .iter()
                    .find(|entry| entry.2)
                    .map(|entry| entry.1.clone());
                let can_join = active_owner
                    .as_ref()
                    .is_some_and(|owner| queue.waiting.iter().all(|entry| &entry.1 == owner));
                if active_owner.is_none() || can_join {
                    let owner =
                        active_owner.unwrap_or_else(|| queue.waiting.front().unwrap().1.clone());
                    // Keep continuous batching open until a competing model
                    // queues. Then freeze the cohort so that model gets a turn.
                    for entry in &mut queue.waiting {
                        if entry.1 == owner {
                            entry.2 = true;
                        }
                    }
                    self.changed.notify_waiters();
                }
                if queue.waiting.iter().any(|entry| entry.0 == id && entry.2) {
                    return permit;
                }
            }
            changed.await;
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
        self.changed.notify_waiters();
    }

    fn clear_owner(&self) {
        *self.owner.write().expect("resource owner lock poisoned") = None;
    }

    async fn acquire(&self, owner_key: &str) -> ResourceAccess {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.is_owner(owner_key) {
                let guard = self.lock.clone().read_owned().await;
                if self.is_owner(owner_key) {
                    return ResourceAccess::Shared(guard);
                }
                drop(guard);
                continue;
            }

            let guard = tokio::select! {
                biased;
                _ = &mut changed => continue,
                guard = self.lock.clone().write_owned() => guard,
            };
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
    _batch: Option<BatchPermit>,
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

    pub async fn acquire_model(&self, model_id: &str) -> crate::error::Result<ModelLease> {
        self.acquire_model_for_request(model_id, &uuid::Uuid::new_v4().to_string())
            .await
    }

    pub async fn acquire_model_for_request(
        &self,
        model_id: &str,
        request_id: &str,
    ) -> crate::error::Result<ModelLease> {
        self.acquire_model_with_policy(model_id, request_id, true)
            .await
    }

    pub async fn acquire_embedding(
        &self,
        model_id: &str,
        allow_engine_swap: bool,
        request_id: &str,
    ) -> crate::error::Result<ModelLease> {
        self.acquire_model_with_policy(model_id, request_id, allow_engine_swap)
            .await
    }

    async fn acquire_model_with_policy(
        &self,
        model_id: &str,
        request_id: &str,
        allow_engine_swap: bool,
    ) -> crate::error::Result<ModelLease> {
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
        let request_id = request_id.to_owned();
        // Queue in the caller's future, not in a detached task. Cancellation
        // before ownership is granted must never start an engine.
        let (batch, access) = if allow_engine_swap {
            let batch = match &gate {
                Some(gate) => Some(gate.batch(&owner_key).await),
                None => None,
            };
            let access = match &gate {
                Some(gate) => Some(gate.acquire(&owner_key).await),
                None => None,
            };
            (batch, access)
        } else {
            // Do not queue behind a switch or wait for another engine's lease.
            // Revalidate under the same gate used by normal acquisition: gateway
            // status may have become stale after routing.
            let unavailable = || {
                crate::error::Error::EngineNotAvailable(format!(
                "cannot serve {model_id} without an engine swap; retry or set allow_engine_swap=true for bulk work"))
            };
            tracing::debug!(%request_id, %model_id, "Checking non-evicting embedding admission");
            let gate = gate.as_ref().ok_or_else(unavailable)?;
            if gate
                .batches
                .lock()
                .unwrap()
                .waiting
                .iter()
                .any(|entry| entry.1 != owner_key)
            {
                return Err(unavailable());
            }
            let access = if gate.is_owner(&owner_key) {
                let guard = gate
                    .lock
                    .clone()
                    .try_read_owned()
                    .map_err(|_| unavailable())?;
                if !gate.is_owner(&owner_key) {
                    return Err(unavailable());
                }
                ResourceAccess::Shared(guard)
            } else {
                ResourceAccess::Exclusive(
                    gate.lock
                        .clone()
                        .try_write_owned()
                        .map_err(|_| unavailable())?,
                )
            };
            // With the gate held, a competing request cannot load between this
            // check and inference. Even switching models in one engine is denied.
            for engine in &engines {
                if engine.engine_type() == target_type
                    || group
                        .as_ref()
                        .is_some_and(|g| resources.get(engine.engine_type()) == Some(g))
                {
                    let loaded = engine.health_check().await?.models_loaded;
                    if loaded.iter().any(|model| {
                        engine.engine_type() != target_type
                            || !model.eq_ignore_ascii_case(&engine_model)
                    }) {
                        tracing::warn!(%request_id, %model_id, engine = engine.engine_type(), "Embedding rejected: engine swap prohibited");
                        return Err(unavailable());
                    }
                }
            }
            (None, Some(access))
        };
        // An abandoned HTTP/WebSocket task must not release the gate while an
        // external process is still starting or stopping.
        let work = self.drain.track_existing();
        let (resource_guard, batch) = tokio::spawn(async move {
            let batch = batch;
            let _work = work;
            let target = owned_target;
            let engine_model = owned_model;
            let guard = match access {
                Some(access) => match access {
                    ResourceAccess::Shared(guard) => {
                        tracing::info!(%request_id, model = %engine_model, engine = %target_type, "Request triggering model load/revalidation");
                        target.load_model(&engine_model).await.inspect_err(|error| {
                            tracing::error!(%request_id, model = %engine_model, engine = %target_type, %error, "Model load failed");
                        })?;
                        Some(guard)
                    }
                    ResourceAccess::Exclusive(write_guard) => {
                        // A failed stop/start must not leave a stale shared-owner
                        // shortcut that bypasses cleanup on the next request.
                        let gate = gate.as_ref().unwrap();
                        let previous_owner = gate.owner.read().unwrap().clone();
                        gate.clear_owner();
                        if let Some(group) = group.filter(|_| allow_engine_swap) {
                            for engine in engines {
                                if engine.engine_type() == target_type
                                    || resources.get(engine.engine_type()) != Some(&group)
                                {
                                    continue;
                                }
                                tracing::info!(%request_id, model = %engine_model, engine = %target_type,
                                    evicted_engine = engine.engine_type(),
                                    evicted_model = previous_owner.as_deref()
                                        .and_then(|owner| owner.split_once('\0'))
                                        .filter(|(kind, _)| *kind == engine.engine_type())
                                        .map(|(_, model)| model),
                                    "Request triggering engine unload");
                                engine.quiesce().await.inspect_err(|error| {
                                    tracing::error!(%request_id, model = %engine_model, engine = engine.engine_type(), %error, "Engine unload failed");
                                })?;
                            }
                        }
                        tracing::info!(%request_id, model = %engine_model, engine = %target_type, "Request triggering model load/revalidation");
                        target.load_model(&engine_model).await.inspect_err(|error| {
                            tracing::error!(%request_id, model = %engine_model, engine = %target_type, %error, "Model load failed");
                        })?;
                        tracing::info!(%request_id, model = %engine_model, engine = %target_type,
                            previous_owner = ?previous_owner, "Model resource transition complete");
                        gate.set_owner(owner_key);
                        Some(OwnedRwLockWriteGuard::downgrade(write_guard))
                    }
                },
                None => {
                    tracing::info!(%request_id, model = %engine_model, engine = %target_type, "Request triggering model load/revalidation");
                    target.load_model(&engine_model).await.inspect_err(|error| {
                            tracing::error!(%request_id, model = %engine_model, engine = %target_type, %error, "Model load failed");
                        })?;
                    None
                }
            };
            Ok::<_, crate::error::Error>((guard, batch))
        })
        .await
        .map_err(|e| crate::error::Error::Internal(e.to_string()))??;
        Ok(ModelLease {
            engine: target,
            engine_model,
            _resource_guard: resource_guard,
            _batch: batch,
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

    struct FakeEngine {
        kind: &'static str,
        loaded: std::sync::atomic::AtomicBool,
        events: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl InferenceEngine for FakeEngine {
        fn engine_type(&self) -> &'static str {
            self.kind
        }
        async fn health_check(&self) -> crate::error::Result<super::super::EngineHealth> {
            Ok(super::super::EngineHealth {
                is_healthy: true,
                version: None,
                models_loaded: if self.loaded.load(std::sync::atomic::Ordering::SeqCst) {
                    vec![self.kind.into()]
                } else {
                    vec![]
                },
            })
        }
        async fn list_models(&self) -> crate::error::Result<Vec<super::super::ModelInfo>> {
            Ok(vec![])
        }
        async fn get_model(
            &self,
            id: &str,
        ) -> crate::error::Result<Option<super::super::ModelInfo>> {
            Ok((id == self.kind).then(|| super::super::ModelInfo {
                id: id.into(),
                name: id.into(),
                revision: None,
                size_bytes: None,
                parameter_count: None,
                context_length: None,
                quantization: None,
                modified_at: None,
                reasoning: None,
            }))
        }
        async fn load_model(&self, _: &str) -> crate::error::Result<()> {
            if !self.loaded.swap(true, std::sync::atomic::Ordering::SeqCst) {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("load {}", self.kind));
                tokio::task::yield_now().await;
            }
            Ok(())
        }
        async fn unload_model(&self, _: &str) -> crate::error::Result<()> {
            self.quiesce().await
        }
        async fn quiesce(&self) -> crate::error::Result<()> {
            if self.loaded.swap(false, std::sync::atomic::Ordering::SeqCst) {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("unload {}", self.kind));
            }
            Ok(())
        }
        async fn chat_completion(
            &self,
            _: &str,
            _: &simple_ai_common::ChatCompletionRequest,
        ) -> crate::error::Result<simple_ai_common::ChatCompletionResponse> {
            assert!(self.loaded.load(std::sync::atomic::Ordering::SeqCst));
            self.events
                .lock()
                .unwrap()
                .push(format!("serve {}", self.kind));
            Ok(serde_json::from_value(serde_json::json!({
                "id":"fake", "object":"chat.completion", "created":0, "model": self.kind, "choices":[]
            })).unwrap())
        }
        async fn chat_completion_stream(
            &self,
            _: &str,
            _: &simple_ai_common::ChatCompletionRequest,
        ) -> crate::error::Result<super::super::ChatCompletionStream> {
            unimplemented!()
        }
    }

    async fn fake_registry() -> (Arc<EngineRegistry>, Arc<std::sync::Mutex<Vec<String>>>) {
        let registry = Arc::new(EngineRegistry::new());
        let events = Arc::new(std::sync::Mutex::new(vec![]));
        for kind in ["tag", "code"] {
            registry
                .register(Arc::new(FakeEngine {
                    kind,
                    loaded: std::sync::atomic::AtomicBool::new(false),
                    events: events.clone(),
                }))
                .await;
        }
        registry
            .set_engine_resources(HashMap::from([
                ("tag".into(), "gpu".into()),
                ("code".into(), "gpu".into()),
            ]))
            .await;
        (registry, events)
    }

    async fn wait_for_queue(registry: &EngineRegistry, count: usize) {
        let gate = registry.resource_gates.read().await["gpu"].clone();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if gate.batches.lock().unwrap().waiting.len() == count {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("queue did not reach expected size");
    }

    #[tokio::test]
    async fn same_model_arrivals_join_active_inference() {
        let (registry, _) = fake_registry().await;
        let held = registry.acquire_model("tag").await.unwrap();
        let started = std::time::Instant::now();
        let mut callers = Vec::new();
        for _ in 0..64 {
            let registry = registry.clone();
            callers.push(tokio::spawn(async move {
                let _lease = registry.acquire_model("tag").await.unwrap();
                tokio::time::sleep(Duration::from_millis(200)).await;
            }));
        }
        let result = tokio::time::timeout(Duration::from_secs(2), async {
            for caller in &mut callers {
                caller.await.unwrap();
            }
        })
        .await;
        eprintln!(
            "64 delayed requests with an active same-model lease: {:?}",
            started.elapsed()
        );
        for caller in callers {
            caller.abort();
        }
        drop(held);
        result.expect("same-model requests must overlap the active lease");
    }

    #[tokio::test]
    async fn default_embedding_rejects_swap_and_opt_in_permits_it() {
        let (registry, events) = fake_registry().await;
        let tag = registry.acquire_model("tag").await.unwrap();
        // Busy conflicting engine: reject immediately, without queueing.
        let result = registry.acquire_embedding("code", false, "test").await;
        assert!(matches!(
            result,
            Err(crate::error::Error::EngineNotAvailable(_))
        ));
        drop(tag);
        // Even after inference finishes, its resident weights must stay intact.
        assert!(registry
            .acquire_embedding("code", false, "test")
            .await
            .is_err());
        assert_eq!(*events.lock().unwrap(), ["load tag"]);
        let code = registry
            .acquire_embedding("code", true, "test")
            .await
            .unwrap();
        drop(code);
        let warm = registry
            .acquire_embedding("code", false, "test")
            .await
            .unwrap();
        drop(warm);
        assert_eq!(
            *events.lock().unwrap(),
            ["load tag", "unload tag", "load code"]
        );
    }

    #[tokio::test]
    async fn default_embedding_loads_empty_group_without_quiescing_peers() {
        let (registry, events) = fake_registry().await;
        let code = registry
            .acquire_embedding("code", false, "test")
            .await
            .unwrap();
        drop(code);
        assert_eq!(*events.lock().unwrap(), ["load code"]);
        // A different engine wins after the gateway observed a warm embedder.
        drop(registry.acquire_model("tag").await.unwrap());
        assert!(registry
            .acquire_embedding("code", false, "test")
            .await
            .is_err());
        assert_eq!(
            *events.lock().unwrap(),
            ["load code", "unload code", "load tag"]
        );
        // The rejection must not poison ownership or leave a queue entry.
        drop(
            registry
                .acquire_embedding("code", true, "test")
                .await
                .unwrap(),
        );
        assert_eq!(
            registry.resource_gates.read().await["gpu"]
                .batches
                .lock()
                .unwrap()
                .waiting
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn alternating_engines_drain_finite_batches_before_switching() {
        let (registry, events) = fake_registry().await;
        let first = registry.acquire_model("tag").await.unwrap();
        let mut callers = vec![];
        for (index, model) in ["code", "tag", "code", "tag"].into_iter().enumerate() {
            let registry_copy = registry.clone();
            callers.push(tokio::spawn(async move {
                let lease = registry_copy.acquire_model(model).await.unwrap();
                lease
                    .engine
                    .chat_completion(
                        &lease.engine_model,
                        &serde_json::from_value(serde_json::json!({"messages":[]})).unwrap(),
                    )
                    .await
                    .unwrap();
            }));
            wait_for_queue(&registry, index + 2).await;
        }
        drop(first);
        tokio::time::timeout(Duration::from_secs(2), async {
            for caller in callers {
                caller.await.unwrap();
            }
        })
        .await
        .expect("alternating requests must make progress");
        assert_eq!(
            *events.lock().unwrap(),
            [
                "load tag",
                "unload tag",
                "load code",
                "serve code",
                "serve code",
                "unload code",
                "load tag",
                "serve tag",
                "serve tag",
            ]
        );
    }

    #[tokio::test]
    async fn cancelled_waiters_never_load_or_unload_an_engine() {
        let (registry, events) = fake_registry().await;
        let lease = registry.acquire_model("tag").await.unwrap();
        for model in ["code", "code"] {
            let r = registry.clone();
            let caller = tokio::spawn(async move { r.acquire_model(model).await });
            wait_for_queue(&registry, 2).await;
            caller.abort();
            assert!(matches!(caller.await, Err(error) if error.is_cancelled()));
            wait_for_queue(&registry, 1).await;
        }
        drop(lease);
        drop(registry.acquire_model("tag").await.unwrap());
        assert_eq!(*events.lock().unwrap(), ["load tag"]);
    }

    #[tokio::test]
    async fn cancelled_competitor_reopens_same_model_admission() {
        let (registry, _) = fake_registry().await;
        let held = registry.acquire_model("tag").await.unwrap();
        let r = registry.clone();
        let competitor = tokio::spawn(async move { r.acquire_model("code").await });
        wait_for_queue(&registry, 2).await;
        let r = registry.clone();
        let mut same = tokio::spawn(async move { r.acquire_model("tag").await });
        wait_for_queue(&registry, 3).await;
        assert!(tokio::time::timeout(Duration::from_millis(20), &mut same)
            .await
            .is_err());
        competitor.abort();
        let _ = competitor.await;
        let joined = tokio::time::timeout(Duration::from_secs(1), same)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        drop(joined);
        drop(held);
    }

    #[tokio::test]
    async fn disconnected_http_client_and_expired_deadline_remove_waiters() {
        use simple_server::lifecycle::{Lifecycle, ShutdownOptions};
        use tokio::io::AsyncWriteExt;
        let (registry, events) = fake_registry().await;
        let held = registry.acquire_model("tag").await.unwrap();
        let config = serde_json::from_value(serde_json::json!({
            "runner":{"id":"test", "name":"test"}, "api":{}
        }))
        .unwrap();
        let state = Arc::new(crate::state::AppState::new(config, registry.clone(), None));
        let app = crate::api::chat::router().with_state(state);
        let listener = simple_server::http::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let lifecycle = Lifecycle::new(ShutdownOptions {
            grace_period: Duration::from_secs(1),
        });
        let shutdown = lifecycle.shutdown();
        let server = tokio::spawn(simple_server::web::serve(listener, app, shutdown));
        let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
        let body = r#"{"model":"code","messages":[]}"#;
        socket.write_all(format!("POST /chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).as_bytes()).await.unwrap();
        wait_for_queue(&registry, 2).await;
        drop(socket);
        wait_for_queue(&registry, 1).await;
        let response = reqwest::Client::new()
            .post(format!("http://{addr}/chat/completions"))
            .header(
                "x-simple-ai-deadline",
                (chrono::Utc::now().timestamp_millis() + 50).to_string(),
            )
            .json(&serde_json::json!({"model":"code", "messages":[]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 504);
        wait_for_queue(&registry, 1).await;
        drop(held);
        drop(registry.acquire_model("tag").await.unwrap());
        assert_eq!(*events.lock().unwrap(), ["load tag"]);
        server.abort();
        let _ = server.await;
    }

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
            if self.kind == "failing" {
                return Err(crate::error::Error::EngineNotAvailable(
                    "cannot stop".into(),
                ));
            }
            self.quiesced
                .store(true, std::sync::atomic::Ordering::SeqCst);
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
            if self.kind != "slow" {
                return Ok(None);
            }
            Ok(Some(super::super::ModelInfo {
                revision: None,
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
            Ok(serde_json::from_value(serde_json::json!({
                "id": "mock", "object": "chat.completion", "created": 0, "model": "model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "served"},
                    "finish_reason": "stop"}]
            }))
            .unwrap())
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
            kind: "slow",
            quiesced: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
            kind: "slow",
            quiesced: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
            registry
                .register(Arc::new(SlowEngine {
                    kind,
                    quiesced: quiesced.clone(),
                    started: Arc::new(tokio::sync::Notify::new()),
                    finish: Arc::new(tokio::sync::Notify::new()),
                    loaded: std::sync::atomic::AtomicBool::new(true),
                }))
                .await;
        }
        registry
            .set_engine_resources(HashMap::from([
                ("slow".into(), "gpu".into()),
                ("warming".into(), "gpu".into()),
            ]))
            .await;
        let _lease = registry.acquire_model("model").await.unwrap();
        assert!(quiesced.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn failed_switch_clears_owner_so_next_request_cannot_bypass_cleanup() {
        let registry = EngineRegistry::new();
        registry
            .set_engine_resources(HashMap::from([
                ("slow".into(), "gpu".into()),
                ("failing".into(), "gpu".into()),
            ]))
            .await;
        for kind in ["slow", "failing"] {
            registry
                .register(Arc::new(SlowEngine {
                    kind,
                    quiesced: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                    started: Arc::new(tokio::sync::Notify::new()),
                    finish: Arc::new(tokio::sync::Notify::new()),
                    loaded: std::sync::atomic::AtomicBool::new(true),
                }))
                .await;
            if kind == "slow" {
                drop(registry.acquire_model("model").await.unwrap());
            }
        }
        assert!(registry.acquire_model("other").await.is_err());
        let gate = registry.resource_gates.read().await["gpu"].clone();
        assert!(!gate.is_owner("slow\0model"));
        assert!(
            registry.acquire_model("model").await.is_err(),
            "must retry cleanup instead of using stale owner"
        );
    }

    #[tokio::test]
    async fn dead_provider_releases_gpu_for_following_model_request() {
        use crate::config::ClassificationEngineConfig;
        let registry = EngineRegistry::new();
        let config = ClassificationEngineConfig {
            command: vec![
                "sh".into(),
                "-c".into(),
                "echo provider-broken >&2; exit 42".into(),
            ],
            models: serde_json::from_value(serde_json::json!([{"id": "broken"}])).unwrap(),
            startup_timeout_secs: 600,
            ..Default::default()
        };
        let classification = Arc::new(super::super::ClassificationEngine::new(config));
        registry.register(classification.clone()).await;
        registry
            .register(Arc::new(SlowEngine {
                kind: "slow",
                quiesced: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                started: Arc::new(tokio::sync::Notify::new()),
                finish: Arc::new(tokio::sync::Notify::new()),
                loaded: std::sync::atomic::AtomicBool::new(true),
            }))
            .await;
        registry
            .set_engine_resources(HashMap::from([
                ("classification".into(), "gpu".into()),
                ("slow".into(), "gpu".into()),
            ]))
            .await;
        registry
            .configure_routes(HashMap::from([
                (
                    "broken".into(),
                    ModelRouteConfig {
                        engine: "classification".into(),
                        engine_model: "broken".into(),
                        aliases: vec![],
                    },
                ),
                (
                    "model".into(),
                    ModelRouteConfig {
                        engine: "slow".into(),
                        engine_model: "model".into(),
                        aliases: vec![],
                    },
                ),
            ]))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            let error = match registry.acquire_model("broken").await {
                Err(error) => error,
                Ok(_) => panic!("broken provider must fail"),
            };
            assert!(error.to_string().contains("exit status: 42"), "{error}");
            assert!(error.to_string().contains("provider-broken"), "{error}");
            assert!(classification
                .health_check()
                .await
                .unwrap()
                .models_loaded
                .is_empty());
            // A following request acquires the same GPU and its serving model immediately.
            let lease = registry.acquire_model("model").await.unwrap();
            let request = serde_json::from_value(serde_json::json!({
                "model": "model", "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap();
            let response = lease
                .engine
                .chat_completion(&lease.engine_model, &request)
                .await
                .unwrap();
            assert_eq!(response.id, "mock");
        })
        .await
        .expect("failed provider must release GPU without the 600s timeout");
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
