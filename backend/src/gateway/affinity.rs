use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;
use rand::{rngs::OsRng, RngCore};
use serde::Serialize;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const DOMAIN: &[u8] = b"simpleai-prompt-cache-affinity-v1";

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct AffinityKey([u8; 32]);

impl Hash for AffinityKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Debug for AffinityKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AffinityKey([redacted])")
    }
}

impl AffinityKey {
    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }
}

#[derive(Clone)]
pub struct AffinityContext {
    key: AffinityKey,
}

impl fmt::Debug for AffinityContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AffinityContext([redacted])")
    }
}

impl AffinityContext {
    pub fn derive(
        secret: &[u8; 32],
        user_id: &str,
        requested_selector: &str,
        cache_key: &ValidatedPromptCacheKey,
    ) -> Self {
        let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts 32-byte keys");
        mac.update(DOMAIN);
        update_component(&mut mac, user_id.as_bytes());
        update_component(&mut mac, requested_selector.as_bytes());
        update_component(&mut mac, cache_key.as_str().as_bytes());
        let bytes: [u8; 32] = mac.finalize().into_bytes().into();
        Self {
            key: AffinityKey(bytes),
        }
    }

    pub fn key(&self) -> AffinityKey {
        self.key
    }

    pub fn runner_cache_key(&self) -> String {
        self.key.to_hex()
    }
}

fn update_component(mac: &mut HmacSha256, bytes: &[u8]) {
    let len = u32::try_from(bytes.len()).expect("affinity component length fits u32");
    mac.update(&len.to_be_bytes());
    mac.update(bytes);
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PromptCacheKeyError {
    #[error("prompt_cache_key must not be empty")]
    Empty,
    #[error("prompt_cache_key must not exceed 64 UTF-8 bytes")]
    TooLong,
}

#[derive(Debug)]
pub struct ValidatedPromptCacheKey(String);

impl ValidatedPromptCacheKey {
    pub fn parse(raw: String) -> Result<Self, PromptCacheKeyError> {
        let normalized = raw.trim();
        if normalized.is_empty() {
            return Err(PromptCacheKeyError::Empty);
        }
        if normalized.len() > 64 {
            return Err(PromptCacheKeyError::TooLong);
        }
        Ok(Self(normalized.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub fn random_affinity_secret() -> [u8; 32] {
    let mut secret = [0_u8; 32];
    OsRng.fill_bytes(&mut secret);
    secret
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinityBinding {
    pub runner_id: String,
    pub runner_generation: u64,
    pub resolved_model: String,
    pub revision: u64,
    last_used: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinityRemovalReason {
    Expired,
    Capacity,
    RunnerDisconnected,
    RunnerUnhealthy,
    ModelUnloaded,
    CircuitOpen,
    DispatchFailed,
    BindingInvalid,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct DecisionLabels {
    outcome: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RemovalLabels {
    reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct AffinityMetricsSnapshot {
    pub bindings: usize,
    pub decisions: HashMap<String, u64>,
    pub evictions: HashMap<String, u64>,
}

pub struct AffinityMetrics {
    registry: Mutex<Registry>,
    decisions: Family<DecisionLabels, Counter>,
    evictions: Family<RemovalLabels, Counter>,
    bindings: Gauge,
}

impl AffinityMetrics {
    pub fn new() -> Arc<Self> {
        let decisions = Family::<DecisionLabels, Counter>::default();
        let evictions = Family::<RemovalLabels, Counter>::default();
        let bindings = Gauge::default();
        let mut registry = Registry::default();
        registry.register(
            "simpleai_cache_affinity_decisions",
            "Final cache-affinity routing decisions",
            decisions.clone(),
        );
        registry.register(
            "simpleai_cache_affinity_evictions",
            "Cache-affinity binding removals",
            evictions.clone(),
        );
        registry.register(
            "simpleai_cache_affinity_bindings",
            "Current cache-affinity bindings",
            bindings.clone(),
        );
        Arc::new(Self {
            registry: Mutex::new(registry),
            decisions,
            evictions,
            bindings,
        })
    }

    pub fn record_decision(&self, outcome: &'static str) {
        self.decisions
            .get_or_create(&DecisionLabels { outcome })
            .inc();
    }

    pub fn record_removals(&self, reason: AffinityRemovalReason, count: usize) {
        if count > 0 {
            self.evictions
                .get_or_create(&RemovalLabels {
                    reason: reason.as_str(),
                })
                .inc_by(count as u64);
        }
    }

    fn set_bindings(&self, count: usize) {
        self.bindings.set(count as i64);
    }

    pub fn encode(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        encode(&mut output, &self.registry.lock().unwrap())?;
        Ok(output)
    }

    pub fn snapshot(&self, bindings: usize) -> AffinityMetricsSnapshot {
        const DECISIONS: [&str; 5] = [
            "new",
            "reuse",
            "spillover_overloaded",
            "rebind_invalid",
            "disabled",
        ];
        const REMOVALS: [AffinityRemovalReason; 8] = [
            AffinityRemovalReason::Expired,
            AffinityRemovalReason::Capacity,
            AffinityRemovalReason::RunnerDisconnected,
            AffinityRemovalReason::RunnerUnhealthy,
            AffinityRemovalReason::ModelUnloaded,
            AffinityRemovalReason::CircuitOpen,
            AffinityRemovalReason::DispatchFailed,
            AffinityRemovalReason::BindingInvalid,
        ];
        let decisions = DECISIONS
            .into_iter()
            .map(|outcome| {
                (
                    outcome.to_string(),
                    self.decisions
                        .get_or_create(&DecisionLabels { outcome })
                        .get(),
                )
            })
            .collect();
        let evictions = REMOVALS
            .into_iter()
            .map(|reason| {
                (
                    reason.as_str().to_string(),
                    self.evictions
                        .get_or_create(&RemovalLabels {
                            reason: reason.as_str(),
                        })
                        .get(),
                )
            })
            .collect();
        AffinityMetricsSnapshot {
            bindings,
            decisions,
            evictions,
        }
    }
}

impl AffinityRemovalReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Expired => "expired",
            Self::Capacity => "capacity",
            Self::RunnerDisconnected => "runner_disconnected",
            Self::RunnerUnhealthy => "runner_unhealthy",
            Self::ModelUnloaded => "model_unloaded",
            Self::CircuitOpen => "circuit_open",
            Self::DispatchFailed => "dispatch_failed",
            Self::BindingInvalid => "binding_invalid",
        }
    }
}

struct StoreState {
    entries: HashMap<AffinityKey, AffinityBinding>,
    next_revision: u64,
}

pub struct AffinityStore {
    ttl: Duration,
    max_entries: usize,
    state: Mutex<StoreState>,
    metrics: Arc<AffinityMetrics>,
}

impl AffinityStore {
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self::with_metrics(ttl, max_entries, AffinityMetrics::new())
    }

    pub fn with_metrics(ttl: Duration, max_entries: usize, metrics: Arc<AffinityMetrics>) -> Self {
        assert!(!ttl.is_zero(), "affinity TTL must be positive");
        assert!(max_entries > 0, "affinity capacity must be positive");
        Self {
            ttl,
            max_entries,
            state: Mutex::new(StoreState {
                entries: HashMap::new(),
                next_revision: 1,
            }),
            metrics,
        }
    }

    pub fn lookup(&self, key: AffinityKey, now: Instant) -> Option<AffinityBinding> {
        let mut state = self.state.lock().unwrap();
        if state
            .entries
            .get(&key)
            .is_some_and(|binding| now.saturating_duration_since(binding.last_used) >= self.ttl)
        {
            state.entries.remove(&key);
            self.metrics
                .record_removals(AffinityRemovalReason::Expired, 1);
            self.metrics.set_bindings(state.entries.len());
            return None;
        }
        state.entries.get(&key).cloned()
    }

    pub fn bind_if_absent(
        &self,
        key: AffinityKey,
        runner_id: String,
        runner_generation: u64,
        resolved_model: String,
        now: Instant,
    ) -> (AffinityBinding, bool) {
        let mut state = self.state.lock().unwrap();
        if let Some(existing) = state.entries.get(&key) {
            return (existing.clone(), false);
        }
        if state.entries.len() >= self.max_entries {
            if let Some(lru_key) = state
                .entries
                .iter()
                .min_by_key(|(_, binding)| binding.last_used)
                .map(|(key, _)| *key)
            {
                state.entries.remove(&lru_key);
                self.metrics
                    .record_removals(AffinityRemovalReason::Capacity, 1);
            }
        }
        let revision = state.next_revision;
        state.next_revision = state.next_revision.wrapping_add(1).max(1);
        let binding = AffinityBinding {
            runner_id,
            runner_generation,
            resolved_model,
            revision,
            last_used: now,
        };
        state.entries.insert(key, binding.clone());
        self.metrics.set_bindings(state.entries.len());
        (binding, true)
    }

    pub fn touch_if(&self, key: AffinityKey, revision: u64, now: Instant) -> bool {
        let mut state = self.state.lock().unwrap();
        let Some(binding) = state.entries.get_mut(&key) else {
            return false;
        };
        if binding.revision != revision {
            return false;
        }
        binding.last_used = now;
        true
    }

    pub fn replace_if(
        &self,
        key: AffinityKey,
        expected_revision: u64,
        runner_id: String,
        runner_generation: u64,
        resolved_model: String,
        now: Instant,
    ) -> Option<AffinityBinding> {
        let mut state = self.state.lock().unwrap();
        if state.entries.get(&key)?.revision != expected_revision {
            return None;
        }
        let revision = state.next_revision;
        state.next_revision = state.next_revision.wrapping_add(1).max(1);
        let replacement = AffinityBinding {
            runner_id,
            runner_generation,
            resolved_model,
            revision,
            last_used: now,
        };
        state.entries.insert(key, replacement.clone());
        Some(replacement)
    }

    pub fn invalidate_if(&self, key: AffinityKey, revision: u64) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.entries.get(&key).map(|b| b.revision) != Some(revision) {
            return false;
        }
        state.entries.remove(&key);
        self.metrics.set_bindings(state.entries.len());
        true
    }

    pub fn invalidate_runner(&self, runner_id: &str, generation: u64) -> usize {
        let mut state = self.state.lock().unwrap();
        let before = state.entries.len();
        state.entries.retain(|_, binding| {
            binding.runner_id != runner_id || binding.runner_generation != generation
        });
        let removed = before - state.entries.len();
        self.metrics.set_bindings(state.entries.len());
        removed
    }

    pub fn invalidate_runner_id(&self, runner_id: &str) -> usize {
        let mut state = self.state.lock().unwrap();
        let before = state.entries.len();
        state
            .entries
            .retain(|_, binding| binding.runner_id != runner_id);
        let removed = before - state.entries.len();
        self.metrics.set_bindings(state.entries.len());
        removed
    }

    pub fn invalidate_unloaded_models(
        &self,
        runner_id: &str,
        generation: u64,
        loaded_models: &[String],
    ) -> usize {
        let mut state = self.state.lock().unwrap();
        let before = state.entries.len();
        state.entries.retain(|_, binding| {
            binding.runner_id != runner_id
                || binding.runner_generation != generation
                || loaded_models.contains(&binding.resolved_model)
        });
        let removed = before - state.entries.len();
        self.metrics.set_bindings(state.entries.len());
        removed
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn metrics(&self) -> &Arc<AffinityMetrics> {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_normalizes_keys() {
        assert_eq!(
            ValidatedPromptCacheKey::parse(" key ".into())
                .unwrap()
                .as_str(),
            "key"
        );
        assert_eq!(
            ValidatedPromptCacheKey::parse("  ".into()).unwrap_err(),
            PromptCacheKeyError::Empty
        );
        assert_eq!(
            ValidatedPromptCacheKey::parse("x".repeat(65)).unwrap_err(),
            PromptCacheKeyError::TooLong
        );
    }

    #[test]
    fn derivation_is_scoped_and_redacted() {
        let key = ValidatedPromptCacheKey::parse("conversation".into()).unwrap();
        let a = AffinityContext::derive(&[7; 32], "user-a", "class:fast", &key);
        let b = AffinityContext::derive(&[7; 32], "user-b", "class:fast", &key);
        assert_ne!(a.key(), b.key());
        assert_eq!(a.runner_cache_key().len(), 64);
        assert!(!format!("{a:?}").contains("conversation"));
    }

    #[test]
    fn binding_ttl_touch_and_conditional_invalidation() {
        let store = AffinityStore::new(Duration::from_secs(10), 4);
        let now = Instant::now();
        let key = AffinityKey([1; 32]);
        let (binding, inserted) = store.bind_if_absent(key, "r1".into(), 1, "m".into(), now);
        assert!(inserted);
        assert!(store.touch_if(key, binding.revision, now + Duration::from_secs(5)));
        assert!(store.lookup(key, now + Duration::from_secs(14)).is_some());
        assert!(!store.invalidate_if(key, binding.revision + 1));
        assert!(store.invalidate_if(key, binding.revision));
    }

    #[test]
    fn evicts_lru_at_capacity() {
        let store = AffinityStore::new(Duration::from_secs(60), 2);
        let now = Instant::now();
        store.bind_if_absent(AffinityKey([1; 32]), "r".into(), 1, "m".into(), now);
        store.bind_if_absent(
            AffinityKey([2; 32]),
            "r".into(),
            1,
            "m".into(),
            now + Duration::from_secs(1),
        );
        store.bind_if_absent(
            AffinityKey([3; 32]),
            "r".into(),
            1,
            "m".into(),
            now + Duration::from_secs(2),
        );
        assert!(store
            .lookup(AffinityKey([1; 32]), now + Duration::from_secs(3))
            .is_none());
        assert_eq!(store.len(), 2);
    }
}
