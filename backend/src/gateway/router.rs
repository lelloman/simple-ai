//! Inference router for distributing requests to runners.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

use super::{ConnectedRunner, RunnerRegistry};

/// Errors from the inference router.
#[derive(Debug, Error)]
pub enum RouterError {
    #[error("No runners available")]
    NoRunners,
    #[error("No runners have model '{0}' loaded")]
    ModelNotLoaded(String),
    #[error("Failed to connect to runner: {0}")]
    ConnectionFailed(String),
    #[error("Runner returned error: {0}")]
    RunnerError(String),
    #[error("Request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),
}

/// Strategy for selecting a runner.
#[derive(Debug, Clone, Copy, Default)]
pub enum LoadBalanceStrategy {
    /// Round-robin across available runners.
    #[default]
    RoundRobin,
    /// Random selection.
    Random,
    /// Prefer runners with specific machine type.
    PreferMachineType,
}

/// Router for distributing inference requests to runners.
pub struct InferenceRouter {
    registry: Arc<RunnerRegistry>,
    http_client: Client,
    strategy: LoadBalanceStrategy,
    round_robin_counter: AtomicUsize,
}

impl InferenceRouter {
    /// Create a new router.
    pub fn new(registry: Arc<RunnerRegistry>) -> Self {
        Self {
            registry,
            http_client: Client::builder()
                .timeout(std::time::Duration::from_secs(300)) // 5 min for long generations
                .build()
                .expect("Failed to create HTTP client"),
            strategy: LoadBalanceStrategy::RoundRobin,
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Create a router with a specific strategy.
    pub fn with_strategy(registry: Arc<RunnerRegistry>, strategy: LoadBalanceStrategy) -> Self {
        Self {
            registry,
            http_client: Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
            strategy,
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Route a chat completion request to an appropriate runner.
    pub async fn chat_completion<Req, Resp>(
        &self,
        model: &str,
        request: &Req,
    ) -> Result<Resp, RouterError>
    where
        Req: Serialize,
        Resp: DeserializeOwned,
    {
        let runner = self.select_runner(Some(model)).await?;
        self.proxy_request(&runner, "/v1/chat/completions", request)
            .await
    }

    /// Route a chat completion request and return the raw response for streaming.
    pub async fn chat_completion_raw<Req>(
        &self,
        model: &str,
        request: &Req,
    ) -> Result<reqwest::Response, RouterError>
    where
        Req: Serialize,
    {
        let runner = self.select_runner(Some(model)).await?;
        self.proxy_request_raw(&runner, "/v1/chat/completions", request)
            .await
    }

    /// Get models from all runners.
    pub async fn list_models(&self) -> Result<Vec<ModelEntry>, RouterError> {
        let models = self.registry.all_models().await;
        Ok(models
            .into_iter()
            .map(|m| ModelEntry {
                id: m.id,
                object: "model".to_string(),
                owned_by: "local".to_string(),
            })
            .collect())
    }

    /// Get models with full details from all runners.
    pub async fn list_models_with_details(&self) -> Result<Vec<crate::gateway::registry::ModelInfo>, RouterError> {
        Ok(self.registry.all_models().await)
    }

    /// Select a runner for the request.
    async fn select_runner(&self, model: Option<&str>) -> Result<ConnectedRunner, RouterError> {
        let candidates = if let Some(model_id) = model {
            let with_model = self.registry.with_model(model_id).await;
            if with_model.is_empty() {
                // Check if any runner is operational
                let operational = self.registry.operational().await;
                if operational.is_empty() {
                    return Err(RouterError::NoRunners);
                }
                // Model not loaded on any runner - return first operational
                // The runner will load the model on demand
                return Ok(self.select_from_candidates(&operational));
            }
            with_model
        } else {
            let operational = self.registry.operational().await;
            if operational.is_empty() {
                return Err(RouterError::NoRunners);
            }
            operational
        };

        Ok(self.select_from_candidates(&candidates))
    }

    /// Select one runner from candidates based on strategy.
    fn select_from_candidates(&self, candidates: &[ConnectedRunner]) -> ConnectedRunner {
        match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
                candidates[idx % candidates.len()].clone()
            }
            LoadBalanceStrategy::Random => {
                use std::collections::hash_map::RandomState;
                use std::hash::{BuildHasher, Hasher};
                let hasher = RandomState::new().build_hasher();
                let idx = hasher.finish() as usize % candidates.len();
                candidates[idx].clone()
            }
            LoadBalanceStrategy::PreferMachineType => {
                // Prefer GPU runners
                candidates
                    .iter()
                    .find(|r| r.machine_type.as_deref() == Some("gpu-server"))
                    .or_else(|| candidates.first())
                    .cloned()
                    .unwrap()
            }
        }
    }

    /// Proxy a request to a runner and deserialize the response.
    async fn proxy_request<Req, Resp>(
        &self,
        runner: &ConnectedRunner,
        path: &str,
        request: &Req,
    ) -> Result<Resp, RouterError>
    where
        Req: Serialize,
        Resp: DeserializeOwned,
    {
        let base_url = runner
            .http_base_url
            .as_ref()
            .ok_or_else(|| RouterError::ConnectionFailed("Runner has no HTTP URL".to_string()))?;

        let url = format!("{}{}", base_url, path);
        tracing::debug!("Proxying request to {} (runner: {})", url, runner.id);

        let response = self
            .http_client
            .post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(RouterError::RunnerError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        response.json().await.map_err(RouterError::RequestFailed)
    }

    /// Proxy a request and return the raw response (for streaming).
    async fn proxy_request_raw<Req>(
        &self,
        runner: &ConnectedRunner,
        path: &str,
        request: &Req,
    ) -> Result<reqwest::Response, RouterError>
    where
        Req: Serialize,
    {
        let base_url = runner
            .http_base_url
            .as_ref()
            .ok_or_else(|| RouterError::ConnectionFailed("Runner has no HTTP URL".to_string()))?;

        let url = format!("{}{}", base_url, path);
        tracing::debug!(
            "Proxying streaming request to {} (runner: {})",
            url,
            runner.id
        );

        let response = self
            .http_client
            .post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(RouterError::RunnerError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        Ok(response)
    }
}

/// Model entry for /v1/models response.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}


#[cfg(test)]
mod tests {
    use super::*;
    use simple_ai_common::{EngineStatus, ModelInfo, RunnerHealth, RunnerStatus};
    use tokio::sync::mpsc;

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
                    .map(|id| ModelInfo {
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
    async fn test_select_runner_no_runners() {
        let registry = Arc::new(RunnerRegistry::new());
        let router = InferenceRouter::new(registry);

        let result = router.select_runner(Some("model")).await;
        assert!(matches!(result, Err(RouterError::NoRunners)));
    }

    #[tokio::test]
    async fn test_select_runner_with_model() {
        let registry = Arc::new(RunnerRegistry::new());
        let (tx, _) = mpsc::channel(32);

        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                None,
                create_test_status(vec!["llama3".to_string()]),
                Some("http://localhost:8080".to_string()),
                tx,
            )
            .await;

        let router = InferenceRouter::new(registry);
        let result = router.select_runner(Some("llama3")).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().id, "runner-1");
    }

    #[tokio::test]
    async fn test_select_runner_round_robin() {
        let registry = Arc::new(RunnerRegistry::new());

        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);

        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                None,
                create_test_status(vec!["model".to_string()]),
                Some("http://host1:8080".to_string()),
                tx1,
            )
            .await;

        registry
            .register(
                "runner-2".to_string(),
                "Runner 2".to_string(),
                None,
                create_test_status(vec!["model".to_string()]),
                Some("http://host2:8080".to_string()),
                tx2,
            )
            .await;

        let router = InferenceRouter::new(registry);

        // Round-robin should alternate
        let r1 = router.select_runner(Some("model")).await.unwrap();
        let r2 = router.select_runner(Some("model")).await.unwrap();

        // They should be different (round robin)
        assert_ne!(r1.id, r2.id);
    }

    #[tokio::test]
    async fn test_list_models() {
        let registry = Arc::new(RunnerRegistry::new());

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

        let router = InferenceRouter::new(registry);
        let models = router.list_models().await.unwrap();

        assert_eq!(models.len(), 3);
        assert!(models.iter().any(|m| m.id == "model-a"));
        assert!(models.iter().any(|m| m.id == "model-b"));
        assert!(models.iter().any(|m| m.id == "model-c"));
    }
}
