//! Runner registry for tracking connected inference runners.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::sync::{broadcast, mpsc, RwLock};

use simple_ai_common::{GatewayMessage, RunnerStatus};

/// Event emitted when runner state changes.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunnerEvent {
    /// A runner has connected to the gateway.
    Connected {
        runner_id: String,
        name: String,
        machine_type: Option<String>,
        health: String,
        loaded_models: Vec<String>,
    },
    /// A runner has disconnected from the gateway.
    Disconnected { runner_id: String },
    /// A runner's status has changed (health or loaded models).
    StatusChanged {
        runner_id: String,
        health: String,
        loaded_models: Vec<String>,
    },
}

/// Information about a connected runner.
#[derive(Debug, Clone)]
pub struct ConnectedRunner {
    /// Unique runner identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Machine type for routing decisions.
    pub machine_type: Option<String>,
    /// Current status.
    pub status: RunnerStatus,
    /// When the runner connected.
    pub connected_at: DateTime<Utc>,
    /// Last heartbeat received.
    pub last_heartbeat: DateTime<Utc>,
    /// HTTP base URL for proxying requests (e.g., "http://192.168.1.102:8080").
    pub http_base_url: Option<String>,
    /// Channel to send messages to this runner.
    pub tx: mpsc::Sender<GatewayMessage>,
    /// MAC address for Wake-on-LAN (format: AA:BB:CC:DD:EE:FF).
    pub mac_address: Option<String>,
}

impl ConnectedRunner {
    /// Check if the runner is healthy and operational.
    pub fn is_operational(&self) -> bool {
        self.status.health.is_operational()
    }

    /// Get list of loaded models across all engines.
    pub fn loaded_models(&self) -> Vec<String> {
        self.status
            .engines
            .iter()
            .flat_map(|e| e.loaded_models.clone())
            .collect()
    }

    /// Check if a specific model is loaded.
    pub fn has_model(&self, model_id: &str) -> bool {
        self.status
            .engines
            .iter()
            .any(|e| e.loaded_models.iter().any(|m| m == model_id))
    }
}

/// Registry of connected inference runners.
pub struct RunnerRegistry {
    runners: RwLock<HashMap<String, ConnectedRunner>>,
    /// Broadcast channel for runner events (connect, disconnect, status changes).
    event_tx: broadcast::Sender<RunnerEvent>,
}

impl Default for RunnerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RunnerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnerRegistry")
            .field("runners", &"<RwLock<...>>")
            .field("event_tx", &"<broadcast::Sender>")
            .finish()
    }
}

impl RunnerRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(64);
        Self {
            runners: RwLock::new(HashMap::new()),
            event_tx: tx,
        }
    }

    /// Subscribe to runner events (connect, disconnect, status changes).
    /// Returns a receiver that will receive RunnerEvent when runner state changes.
    pub fn subscribe_events(&self) -> broadcast::Receiver<RunnerEvent> {
        self.event_tx.subscribe()
    }

    /// Register a new runner.
    pub async fn register(
        &self,
        id: String,
        name: String,
        machine_type: Option<String>,
        status: RunnerStatus,
        http_base_url: Option<String>,
        tx: mpsc::Sender<GatewayMessage>,
        mac_address: Option<String>,
    ) {
        let now = Utc::now();
        let health = format!("{:?}", status.health);
        let loaded_models: Vec<String> = status
            .engines
            .iter()
            .flat_map(|e| e.loaded_models.clone())
            .collect();

        let runner = ConnectedRunner {
            id: id.clone(),
            name: name.clone(),
            machine_type: machine_type.clone(),
            status,
            connected_at: now,
            last_heartbeat: now,
            http_base_url,
            tx,
            mac_address,
        };
        self.runners.write().await.insert(id.clone(), runner);

        // Notify subscribers that a runner connected
        // Ignore send errors (no active subscribers)
        let _ = self.event_tx.send(RunnerEvent::Connected {
            runner_id: id,
            name,
            machine_type,
            health,
            loaded_models,
        });
    }

    /// Remove a runner from the registry.
    pub async fn unregister(&self, id: &str) -> Option<ConnectedRunner> {
        let removed = self.runners.write().await.remove(id);
        if removed.is_some() {
            // Notify subscribers that a runner disconnected
            let _ = self.event_tx.send(RunnerEvent::Disconnected {
                runner_id: id.to_string(),
            });
        }
        removed
    }

    /// Update a runner's status.
    pub async fn update_status(&self, id: &str, status: RunnerStatus) {
        let mut runners = self.runners.write().await;
        if let Some(runner) = runners.get_mut(id) {
            // Check if health or loaded_models changed
            let old_health = format!("{:?}", runner.status.health);
            let new_health = format!("{:?}", status.health);
            let old_models: Vec<String> = runner.loaded_models();
            let new_models: Vec<String> = status
                .engines
                .iter()
                .flat_map(|e| e.loaded_models.clone())
                .collect();

            let changed = old_health != new_health || old_models != new_models;

            runner.status = status;
            runner.last_heartbeat = Utc::now();

            if changed {
                // Notify subscribers of the status change
                let _ = self.event_tx.send(RunnerEvent::StatusChanged {
                    runner_id: id.to_string(),
                    health: new_health,
                    loaded_models: new_models,
                });
            }
        }
    }

    /// Get a runner by ID.
    pub async fn get(&self, id: &str) -> Option<ConnectedRunner> {
        self.runners.read().await.get(id).cloned()
    }

    /// Get all connected runners.
    pub async fn all(&self) -> Vec<ConnectedRunner> {
        self.runners.read().await.values().cloned().collect()
    }

    /// Get all operational runners (healthy or degraded).
    pub async fn operational(&self) -> Vec<ConnectedRunner> {
        self.runners
            .read()
            .await
            .values()
            .filter(|r| r.is_operational())
            .cloned()
            .collect()
    }

    /// Get runners that have a specific model loaded.
    pub async fn with_model(&self, model_id: &str) -> Vec<ConnectedRunner> {
        self.runners
            .read()
            .await
            .values()
            .filter(|r| r.is_operational() && r.has_model(model_id))
            .cloned()
            .collect()
    }

    /// Get all unique models available across all runners.
    pub async fn all_models(&self) -> Vec<ModelInfo> {
        let runners = self.runners.read().await;
        let mut models: HashMap<String, ModelInfo> = HashMap::new();

        for runner in runners.values() {
            if !runner.is_operational() {
                continue;
            }

            // Collect all loaded model IDs for this runner
            let loaded_ids: std::collections::HashSet<String> = runner
                .loaded_models()
                .into_iter()
                .collect();

            // Get all available models from each engine
            for engine in &runner.status.engines {
                for available_model in &engine.available_models {
                    let model_id = &available_model.id;
                    let is_loaded = loaded_ids.contains(model_id);

                    let entry = models
                        .entry(model_id.clone())
                        .or_insert_with(|| ModelInfo {
                            id: model_id.clone(),
                            name: available_model.name.clone(),
                            size_bytes: available_model.size_bytes,
                            parameter_count: available_model.parameter_count,
                            context_length: available_model.context_length,
                            quantization: available_model.quantization.clone(),
                            modified_at: available_model.modified_at.clone(),
                            loaded: false,
                            runners: vec![],
                            available_on: vec![],
                        });

                    // Track which runners have this model available
                    if !entry.available_on.contains(&runner.id) {
                        entry.available_on.push(runner.id.clone());
                    }

                    if is_loaded && !entry.runners.contains(&runner.id) {
                        entry.runners.push(runner.id.clone());
                    }

                    // Update loaded status if any runner has it loaded
                    if is_loaded {
                        entry.loaded = true;
                    }
                }
            }
        }

        models.into_values().collect()
    }

    /// Count connected runners.
    pub async fn count(&self) -> usize {
        self.runners.read().await.len()
    }

    /// Remove stale runners (no heartbeat within timeout).
    pub async fn remove_stale(&self, timeout_secs: i64) -> Vec<String> {
        let now = Utc::now();
        let mut removed = vec![];
        let mut runners = self.runners.write().await;

        runners.retain(|id, runner| {
            let age = (now - runner.last_heartbeat).num_seconds();
            if age > timeout_secs {
                tracing::warn!(
                    "Removing stale runner {} (no heartbeat for {}s)",
                    id,
                    age
                );
                removed.push(id.clone());
                false
            } else {
                true
            }
        });

        // Drop the lock before sending events to avoid holding it during broadcast
        drop(runners);

        // Notify subscribers of disconnections
        for id in &removed {
            let _ = self.event_tx.send(RunnerEvent::Disconnected {
                runner_id: id.clone(),
            });
        }

        removed
    }
}

/// Information about a model available in the fleet.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Model size in bytes (if known).
    pub size_bytes: Option<u64>,
    /// Parameter count (if known).
    pub parameter_count: Option<u64>,
    /// Maximum context length.
    pub context_length: Option<u32>,
    /// Quantization type (e.g., "Q4_K_M").
    pub quantization: Option<String>,
    /// When the model was last modified.
    pub modified_at: Option<String>,
    /// Whether this model is currently loaded on any runner.
    pub loaded: bool,
    /// Runner IDs that have this model loaded (in GPU memory).
    pub runners: Vec<String>,
    /// Runner IDs that have this model available (on disk).
    pub available_on: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use simple_ai_common::{EngineStatus, ModelInfo as ProtocolModelInfo, RunnerHealth};

    fn create_test_status(models: Vec<String>) -> RunnerStatus {
        RunnerStatus {
            health: RunnerHealth::Healthy,
            capabilities: vec![],
            engines: vec![EngineStatus {
                engine_type: "test".to_string(),
                is_healthy: true,
                version: None,
                loaded_models: models.clone(),
                available_models: models
                    .into_iter()
                    .map(|id| ProtocolModelInfo {
                        id: id.clone(),
                        name: id.clone(),
                        size_bytes: None,
                        parameter_count: None,
                        context_length: None,
                        quantization: None,
                        modified_at: None,
                    })
                    .collect(),
                error: None,
            }],
            metrics: None,
        }
    }

    #[tokio::test]
    async fn test_register_and_get() {
        let registry = RunnerRegistry::new();
        let (tx, _rx) = mpsc::channel(32);
        let status = create_test_status(vec!["model-a".to_string()]);

        registry
            .register(
                "runner-1".to_string(),
                "Test Runner".to_string(),
                Some("gpu".to_string()),
                status,
                Some("http://localhost:8080".to_string()),
                tx,
                None,
            )
            .await;

        let runner = registry.get("runner-1").await.unwrap();
        assert_eq!(runner.id, "runner-1");
        assert_eq!(runner.name, "Test Runner");
        assert_eq!(runner.machine_type, Some("gpu".to_string()));
    }

    #[tokio::test]
    async fn test_unregister() {
        let registry = RunnerRegistry::new();
        let (tx, _rx) = mpsc::channel(32);
        let status = create_test_status(vec![]);

        registry
            .register(
                "runner-1".to_string(),
                "Test".to_string(),
                None,
                status,
                None,
                tx,
                None,
            )
            .await;

        assert!(registry.get("runner-1").await.is_some());

        let removed = registry.unregister("runner-1").await;
        assert!(removed.is_some());
        assert!(registry.get("runner-1").await.is_none());
    }

    #[tokio::test]
    async fn test_with_model() {
        let registry = RunnerRegistry::new();

        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);

        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                None,
                create_test_status(vec!["llama3".to_string()]),
                None,
                tx1,
                None,
            )
            .await;

        registry
            .register(
                "runner-2".to_string(),
                "Runner 2".to_string(),
                None,
                create_test_status(vec!["gpt4".to_string()]),
                None,
                tx2,
                None,
            )
            .await;

        let with_llama = registry.with_model("llama3").await;
        assert_eq!(with_llama.len(), 1);
        assert_eq!(with_llama[0].id, "runner-1");

        let with_gpt = registry.with_model("gpt4").await;
        assert_eq!(with_gpt.len(), 1);
        assert_eq!(with_gpt[0].id, "runner-2");

        let with_none = registry.with_model("nonexistent").await;
        assert!(with_none.is_empty());
    }

    #[tokio::test]
    async fn test_all_models() {
        let registry = RunnerRegistry::new();

        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);

        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                None,
                create_test_status(vec!["model-a".to_string(), "model-b".to_string()]),
                None,
                tx1,
                None,
            )
            .await;

        registry
            .register(
                "runner-2".to_string(),
                "Runner 2".to_string(),
                None,
                create_test_status(vec!["model-a".to_string(), "model-c".to_string()]),
                None,
                tx2,
                None,
            )
            .await;

        let models = registry.all_models().await;
        assert_eq!(models.len(), 3);

        let model_a = models.iter().find(|m| m.id == "model-a").unwrap();
        assert_eq!(model_a.runners.len(), 2);
        assert!(model_a.loaded); // model-a is loaded on both runners
        assert_eq!(model_a.name, "model-a");
    }

    #[tokio::test]
    async fn test_operational_filters_unhealthy() {
        let registry = RunnerRegistry::new();

        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);

        let healthy_status = create_test_status(vec!["model".to_string()]);
        let mut unhealthy_status = create_test_status(vec!["model".to_string()]);
        unhealthy_status.health = RunnerHealth::Unhealthy;

        registry
            .register(
                "healthy".to_string(),
                "Healthy".to_string(),
                None,
                healthy_status,
                None,
                tx1,
                None,
            )
            .await;

        registry
            .register(
                "unhealthy".to_string(),
                "Unhealthy".to_string(),
                None,
                unhealthy_status,
                None,
                tx2,
                None,
            )
            .await;

        let operational = registry.operational().await;
        assert_eq!(operational.len(), 1);
        assert_eq!(operational[0].id, "healthy");
    }

    #[tokio::test]
    async fn test_update_status() {
        let registry = RunnerRegistry::new();
        let (tx, _) = mpsc::channel(32);

        registry
            .register(
                "runner-1".to_string(),
                "Test".to_string(),
                None,
                create_test_status(vec![]),
                None,
                tx,
                None,
            )
            .await;

        let new_status = create_test_status(vec!["new-model".to_string()]);
        registry.update_status("runner-1", new_status).await;

        let runner = registry.get("runner-1").await.unwrap();
        assert!(runner.has_model("new-model"));
    }
}
