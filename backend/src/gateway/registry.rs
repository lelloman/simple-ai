//! Runner registry for tracking connected inference runners.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, RwLock};

use simple_ai_common::{GatewayMessage, RunnerStatus};

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
#[derive(Debug, Default)]
pub struct RunnerRegistry {
    runners: RwLock<HashMap<String, ConnectedRunner>>,
}

impl RunnerRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            runners: RwLock::new(HashMap::new()),
        }
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
    ) {
        let now = Utc::now();
        let runner = ConnectedRunner {
            id: id.clone(),
            name,
            machine_type,
            status,
            connected_at: now,
            last_heartbeat: now,
            http_base_url,
            tx,
        };
        self.runners.write().await.insert(id, runner);
    }

    /// Remove a runner from the registry.
    pub async fn unregister(&self, id: &str) -> Option<ConnectedRunner> {
        self.runners.write().await.remove(id)
    }

    /// Update a runner's status.
    pub async fn update_status(&self, id: &str, status: RunnerStatus) {
        if let Some(runner) = self.runners.write().await.get_mut(id) {
            runner.status = status;
            runner.last_heartbeat = Utc::now();
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
            for model_id in runner.loaded_models() {
                models
                    .entry(model_id.clone())
                    .or_insert_with(|| ModelInfo {
                        id: model_id,
                        runners: vec![],
                    })
                    .runners
                    .push(runner.id.clone());
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

        removed
    }
}

/// Information about a model available in the fleet.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    /// Runner IDs that have this model loaded.
    pub runners: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use simple_ai_common::{EngineStatus, RunnerHealth};

    fn create_test_status(models: Vec<String>) -> RunnerStatus {
        RunnerStatus {
            health: RunnerHealth::Healthy,
            capabilities: vec![],
            engines: vec![EngineStatus {
                engine_type: "test".to_string(),
                is_healthy: true,
                version: None,
                loaded_models: models,
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
            )
            .await;

        let models = registry.all_models().await;
        assert_eq!(models.len(), 3);

        let model_a = models.iter().find(|m| m.id == "model-a").unwrap();
        assert_eq!(model_a.runners.len(), 2);
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
            )
            .await;

        let new_status = create_test_status(vec!["new-model".to_string()]);
        registry.update_status("runner-1", new_status).await;

        let runner = registry.get("runner-1").await.unwrap();
        assert!(runner.has_model("new-model"));
    }
}
