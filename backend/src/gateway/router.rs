//! Inference router for distributing requests to runners.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

use simple_ai_common::{Capability, ChatCompletionRequest, OcrOptions, OcrResponse};

use super::batch_queue::BatchQueue;
use super::model_class::{classify_model, ModelClass, ModelRequest};
use super::{ConnectedRunner, RunnerRegistry};
use crate::audit::{AuditLogger, RunnerMetricRow};
use crate::circuit_breaker::CircuitBreaker;
use crate::config::{ModelsConfig, RoutingConfig};

/// Errors from the inference router.
#[derive(Debug, Error)]
pub enum RouterError {
    #[error("No runners available")]
    NoRunners,
    #[error("Unknown model: {0}")]
    UnknownModel(String),
    #[error("No runners have model '{0}' loaded")]
    ModelNotLoaded(String),
    #[error("No runners have models of class '{0}'")]
    NoModelsOfClass(String),
    #[error("No runners have ready capability '{0}'")]
    NoRunnersWithCapability(String),
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
    /// Smart routing with preference, queue, and latency scoring.
    SmartRouting,
}

/// Result of runner selection, includes the resolved model name.
#[derive(Debug)]
struct SelectedRunner {
    runner: ConnectedRunner,
    /// The actual model to use (resolved from class if needed).
    resolved_model: String,
    /// Whether the resolved model is already loaded on the selected runner.
    is_loaded: bool,
}

/// Planning information for a routed request.
#[derive(Debug, Clone)]
pub struct RoutePlan {
    pub runner: ConnectedRunner,
    pub resolved_model: String,
    pub is_loaded: bool,
}

/// Result of a routed request, includes metadata about routing.
#[derive(Debug)]
pub struct RoutedResponse<T> {
    /// The response from the runner.
    pub response: T,
    /// ID of the runner that handled the request.
    pub runner_id: String,
    /// The resolved model name (may differ from requested if class was used).
    pub resolved_model: String,
}

/// Router for distributing inference requests to runners.
pub struct InferenceRouter {
    registry: Arc<RunnerRegistry>,
    http_client: Client,
    strategy: LoadBalanceStrategy,
    round_robin_counter: AtomicUsize,
    models_config: ModelsConfig,
    routing_config: RoutingConfig,
    audit_logger: Arc<AuditLogger>,
    circuit_breaker: Option<Arc<CircuitBreaker>>,
}

impl InferenceRouter {
    /// Create a new router.
    pub fn new(
        registry: Arc<RunnerRegistry>,
        models_config: ModelsConfig,
        routing_config: RoutingConfig,
        audit_logger: Arc<AuditLogger>,
    ) -> Self {
        Self {
            registry,
            http_client: Client::builder()
                .timeout(std::time::Duration::from_secs(300)) // 5 min for long generations
                .build()
                .expect("Failed to create HTTP client"),
            strategy: LoadBalanceStrategy::SmartRouting,
            round_robin_counter: AtomicUsize::new(0),
            models_config,
            routing_config,
            audit_logger,
            circuit_breaker: None,
        }
    }

    /// Create a router with a specific strategy.
    pub fn with_strategy(
        registry: Arc<RunnerRegistry>,
        strategy: LoadBalanceStrategy,
        models_config: ModelsConfig,
        routing_config: RoutingConfig,
        audit_logger: Arc<AuditLogger>,
    ) -> Self {
        Self {
            registry,
            http_client: Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
            strategy,
            round_robin_counter: AtomicUsize::new(0),
            models_config,
            routing_config,
            audit_logger,
            circuit_breaker: None,
        }
    }

    /// Set the circuit breaker for this router.
    pub fn with_circuit_breaker(mut self, cb: Arc<CircuitBreaker>) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }

    /// Get a reference to the registry.
    pub fn registry(&self) -> &Arc<RunnerRegistry> {
        &self.registry
    }

    /// Route a chat completion request to an appropriate runner.
    ///
    /// The model parameter can be:
    /// - A specific model ID (e.g., "llama3:8b")
    /// - A class request (e.g., "class:fast" or "class:big")
    ///
    /// Returns a RoutedResponse containing the response and routing metadata.
    /// Automatically tracks active requests for smart routing.
    pub async fn chat_completion<Req, Resp>(
        &self,
        model: &str,
        request: &Req,
    ) -> Result<RoutedResponse<Resp>, RouterError>
    where
        Req: Serialize + Clone,
        Resp: DeserializeOwned,
    {
        let selection = self.select_runner_for_model(model).await?;
        let runner_id = selection.runner.id.clone();
        let resolved_model = selection.resolved_model.clone();

        // Resolve canonical → local for this runner (handles both class requests and aliases)
        let local_model = selection.runner.resolve_model_alias(&resolved_model);

        // Always modify request with runner's local model name
        let mut request_value = serde_json::to_value(request)
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;
        if let Some(obj) = request_value.as_object_mut() {
            obj.insert("model".to_string(), serde_json::Value::String(local_model));
        }

        // Track active request for smart routing
        self.registry.increment_requests(&runner_id).await;

        let response = self
            .proxy_request_value(&selection.runner, "/v1/chat/completions", request_value)
            .await;

        // Always decrement, even on error
        self.registry.decrement_requests(&runner_id).await;

        // Record circuit breaker outcome
        if let Some(cb) = &self.circuit_breaker {
            if response.is_ok() {
                cb.record_success(&runner_id);
            } else {
                cb.record_failure(&runner_id);
            }
        }

        Ok(RoutedResponse {
            response: response?,
            runner_id,
            resolved_model,
        })
    }

    /// Route an embedding request to an appropriate runner.
    ///
    /// Identical to `chat_completion()` except it proxies to `/v1/embeddings`.
    pub async fn embed<Req, Resp>(
        &self,
        model: &str,
        request: &Req,
    ) -> Result<RoutedResponse<Resp>, RouterError>
    where
        Req: Serialize + Clone,
        Resp: DeserializeOwned,
    {
        let selection = self.select_runner_for_model(model).await?;
        let runner_id = selection.runner.id.clone();
        let resolved_model = selection.resolved_model.clone();

        let local_model = selection.runner.resolve_model_alias(&resolved_model);

        let mut request_value = serde_json::to_value(request)
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;
        if let Some(obj) = request_value.as_object_mut() {
            obj.insert("model".to_string(), serde_json::Value::String(local_model));
        }

        self.registry.increment_requests(&runner_id).await;

        let response = self
            .proxy_request_value(&selection.runner, "/v1/embeddings", request_value)
            .await;

        self.registry.decrement_requests(&runner_id).await;

        if let Some(cb) = &self.circuit_breaker {
            if response.is_ok() {
                cb.record_success(&runner_id);
            } else {
                cb.record_failure(&runner_id);
            }
        }

        Ok(RoutedResponse {
            response: response?,
            runner_id,
            resolved_model,
        })
    }

    /// Route an OCR multipart request to a runner with ready OCR capability.
    pub async fn ocr_multipart(
        &self,
        file_name: String,
        file_bytes: Vec<u8>,
        options_json: String,
    ) -> Result<RoutedResponse<OcrResponse>, RouterError> {
        let options: OcrOptions = serde_json::from_str(&options_json)
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;
        let runner = self.select_runner_for_ocr(options.mode).await?;
        let runner_id = runner.id.clone();

        let form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(file_bytes).file_name(file_name),
            )
            .text("options", options_json);

        self.registry.increment_requests(&runner_id).await;
        let response = self.proxy_multipart(&runner, "/v1/ocr", form).await;
        self.registry.decrement_requests(&runner_id).await;

        if let Some(cb) = &self.circuit_breaker {
            if response.is_ok() {
                cb.record_success(&runner_id);
            } else {
                cb.record_failure(&runner_id);
            }
        }

        Ok(RoutedResponse {
            response: response?,
            runner_id,
            resolved_model: "ocr".to_string(),
        })
    }

    /// Route a chat completion request through the batch queue.
    ///
    /// This method enqueues the request and waits for it to be processed
    /// as part of a batch. The batch dispatcher will handle the actual
    /// routing and response delivery.
    pub async fn chat_completion_batched(
        &self,
        model: &str,
        request: &ChatCompletionRequest,
        batch_queue: &BatchQueue,
    ) -> Result<RoutedResponse<simple_ai_common::ChatCompletionResponse>, RouterError> {
        // Resolve model class if needed
        let resolved_model = if model.starts_with("class:") {
            let class = match model {
                "class:fast" => ModelClass::Fast,
                "class:big" => ModelClass::Big,
                _ => return Err(RouterError::UnknownModel(model.to_string())),
            };

            // Find a model for this class
            self.resolve_class_to_model(class).await?
        } else {
            model.to_string()
        };

        // Enqueue the request and wait for the response
        let rx = batch_queue
            .enqueue(resolved_model.clone(), request.clone())
            .await;

        // Wait for the batched response
        let batched = rx.await.map_err(|_| {
            RouterError::ConnectionFailed("Batch queue response channel closed".to_string())
        })??;

        Ok(RoutedResponse {
            response: batched.response,
            runner_id: batched.runner_id,
            resolved_model: batched.resolved_model,
        })
    }

    /// Route a chat completion request and return the raw response for streaming.
    ///
    /// The model parameter can be:
    /// - A specific model ID (e.g., "llama3:8b")
    /// - A class request (e.g., "class:fast" or "class:big")
    ///
    /// Note: Streaming requests don't automatically decrement the active request count
    /// since we can't track when the stream completes. This means queue depth may be
    /// slightly underestimated for runners serving streaming requests.
    pub async fn chat_completion_raw<Req>(
        &self,
        model: &str,
        request: &Req,
    ) -> Result<RoutedResponse<reqwest::Response>, RouterError>
    where
        Req: Serialize + Clone,
    {
        let selection = self.select_runner_for_model(model).await?;
        let runner_id = selection.runner.id.clone();

        // Resolve canonical → local for this runner (handles both class requests and aliases)
        let local_model = selection
            .runner
            .resolve_model_alias(&selection.resolved_model);

        // Always modify request with runner's local model name
        let mut request_value = serde_json::to_value(request)
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;
        if let Some(obj) = request_value.as_object_mut() {
            obj.insert("model".to_string(), serde_json::Value::String(local_model));
        }

        // Track active request (Note: we can't easily decrement for streaming)
        self.registry.increment_requests(&runner_id).await;

        let response = self
            .proxy_request_raw_value(&selection.runner, "/v1/chat/completions", request_value)
            .await;

        // Decrement on connection error (if response succeeded, stream is ongoing)
        if response.is_err() {
            self.registry.decrement_requests(&runner_id).await;
            if let Some(cb) = &self.circuit_breaker {
                cb.record_failure(&runner_id);
            }
        } else if let Some(cb) = &self.circuit_breaker {
            cb.record_success(&runner_id);
        }

        response.map(|response| RoutedResponse {
            response,
            runner_id,
            resolved_model: selection.resolved_model,
        })
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
    pub async fn list_models_with_details(
        &self,
    ) -> Result<Vec<crate::gateway::registry::ModelInfo>, RouterError> {
        Ok(self.registry.all_models().await)
    }

    /// Plan a routed request without executing it.
    pub async fn plan_request(&self, model: &str) -> Result<RoutePlan, RouterError> {
        let selection = self.select_runner_for_model(model).await?;
        Ok(RoutePlan {
            runner: selection.runner,
            resolved_model: selection.resolved_model,
            is_loaded: selection.is_loaded,
        })
    }

    /// Select a runner for the given model string.
    ///
    /// Handles both specific model requests and class requests.
    async fn select_runner_for_model(&self, model: &str) -> Result<SelectedRunner, RouterError> {
        let model_request = ModelRequest::parse(model);

        match model_request {
            ModelRequest::Specific(model_id) => {
                let runner = self.select_runner_for_specific(&model_id).await?;
                let is_loaded = runner.has_model_or_alias(&model_id);
                Ok(SelectedRunner {
                    runner,
                    resolved_model: model_id,
                    is_loaded,
                })
            }
            ModelRequest::Class(class) => self.select_runner_for_class(class).await,
        }
    }

    /// Filter out runners whose circuit breaker is open.
    fn filter_available(&self, runners: Vec<ConnectedRunner>) -> Vec<ConnectedRunner> {
        if let Some(cb) = &self.circuit_breaker {
            runners
                .into_iter()
                .filter(|r| cb.is_available(&r.id))
                .collect()
        } else {
            runners
        }
    }

    async fn select_runner_for_ocr(
        &self,
        mode: simple_ai_common::OcrMode,
    ) -> Result<ConnectedRunner, RouterError> {
        let candidates: Vec<ConnectedRunner> = self
            .filter_available(self.registry.operational().await)
            .into_iter()
            .filter(|r| r.has_ocr_mode(mode))
            .collect();

        if candidates.is_empty() {
            return Err(RouterError::NoRunnersWithCapability(format!(
                "{}:{:?}",
                Capability::Ocr,
                mode
            )));
        }

        Ok(self.select_from_candidates(&candidates))
    }

    /// Select a runner for a specific model.
    async fn select_runner_for_specific(
        &self,
        model_id: &str,
    ) -> Result<ConnectedRunner, RouterError> {
        let with_model = self.filter_available(self.registry.with_model(model_id).await);
        if with_model.is_empty() {
            let compatible =
                self.filter_available(self.registry.with_available_model(model_id).await);
            if compatible.is_empty() {
                return Err(RouterError::NoRunners);
            }
            return Ok(self.select_from_candidates(&compatible));
        }
        Ok(self.select_from_candidates(&with_model))
    }

    /// Resolve a model class to a specific model ID.
    ///
    /// Finds any available model of the requested class.
    async fn resolve_class_to_model(&self, class: ModelClass) -> Result<String, RouterError> {
        let operational = self.filter_available(self.registry.operational().await);
        if operational.is_empty() {
            return Err(RouterError::NoRunners);
        }

        // Find any model of the requested class
        for runner in &operational {
            for model in runner
                .status
                .engines
                .iter()
                .flat_map(|e| e.available_models.iter())
            {
                if classify_model(&model.id, &self.models_config) == Some(class) {
                    return Ok(model.id.clone());
                }
            }
        }

        Err(RouterError::NoModelsOfClass(class.to_string()))
    }

    /// Select a runner for a model class.
    ///
    /// Finds runners that have models of the requested class and picks one.
    async fn select_runner_for_class(
        &self,
        class: ModelClass,
    ) -> Result<SelectedRunner, RouterError> {
        let operational = self.filter_available(self.registry.operational().await);
        if operational.is_empty() {
            return Err(RouterError::NoRunners);
        }

        // Prefer runners that already have a matching model loaded before considering cold loads.
        let mut loaded_candidates: Vec<(ConnectedRunner, String)> = Vec::new();
        let mut available_candidates: Vec<(ConnectedRunner, String)> = Vec::new();

        for runner in &operational {
            let models: Vec<_> = runner
                .status
                .engines
                .iter()
                .flat_map(|e| e.available_models.iter())
                .collect();
            for model in models {
                if classify_model(&model.id, &self.models_config) == Some(class) {
                    let target = if runner.has_model_or_alias(&model.id) {
                        &mut loaded_candidates
                    } else {
                        &mut available_candidates
                    };
                    target.push((runner.clone(), model.id.clone()));
                    break; // One model per runner is enough
                }
            }
        }

        let is_loaded = !loaded_candidates.is_empty();
        let candidates_with_models = if is_loaded {
            loaded_candidates
        } else {
            available_candidates
        };

        if candidates_with_models.is_empty() {
            return Err(RouterError::NoModelsOfClass(class.to_string()));
        }

        // Select using load balancing
        let idx = match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                self.round_robin_counter.fetch_add(1, Ordering::Relaxed)
                    % candidates_with_models.len()
            }
            LoadBalanceStrategy::Random => {
                use std::collections::hash_map::RandomState;
                use std::hash::{BuildHasher, Hasher};
                let hasher = RandomState::new().build_hasher();
                hasher.finish() as usize % candidates_with_models.len()
            }
            LoadBalanceStrategy::PreferMachineType => {
                // Prefer GPU runners for big models
                candidates_with_models
                    .iter()
                    .position(|(r, _)| r.machine_type.as_deref() == Some("gpu-server"))
                    .unwrap_or(0)
            }
            LoadBalanceStrategy::SmartRouting => {
                self.select_smart_routing(&candidates_with_models, class)
                    .await
            }
        };

        let (runner, resolved_model) = candidates_with_models.into_iter().nth(idx).unwrap();
        tracing::info!(
            "Selected runner {} with model {} for class:{} (strategy: {:?})",
            runner.id,
            resolved_model,
            class,
            self.strategy
        );

        Ok(SelectedRunner {
            runner,
            resolved_model,
            is_loaded,
        })
    }

    /// Select runner using smart routing with preference, queue, and latency scoring.
    ///
    /// Score = (pref_weight * preference_score) + (queue_weight * queue_score) + (latency_weight * latency_score)
    /// Lower score is better.
    async fn select_smart_routing(
        &self,
        candidates: &[(ConnectedRunner, String)],
        class: ModelClass,
    ) -> usize {
        let class_name = class.as_str();
        // Ensure pref_weight is non-negative even if queue_weight + latency_weight > 1.0
        let pref_weight =
            (1.0 - self.routing_config.queue_weight - self.routing_config.latency_weight).max(0.0);

        // Get class preferences (ordered list of preferred machine types)
        let preferences = self.routing_config.class_preferences.get(class_name);
        let all_operational = self.registry.operational().await;

        // Collect all runner metrics for scoring
        let all_metrics = self.audit_logger.get_all_metrics().unwrap_or_default();

        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, (runner, _model))| {
                let score = self.score_runner(
                    runner,
                    class_name,
                    preferences,
                    &all_metrics,
                    &all_operational,
                    pref_weight,
                );
                (idx, score)
            })
            .collect();

        // Sort by score (ascending - lower is better)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_idx, best_score) = scored[0];
        let best_runner = &candidates[best_idx].0;
        tracing::info!(
            "SmartRouting selected runner {} with score {:.2} (active_requests: {}, machine_type: {:?})",
            best_runner.id,
            best_score,
            best_runner.active_requests.load(Ordering::SeqCst),
            best_runner.machine_type
        );

        best_idx
    }

    /// Score a runner for smart routing.
    ///
    /// Lower score is better.
    fn score_runner(
        &self,
        runner: &ConnectedRunner,
        class_name: &str,
        preferences: Option<&Vec<String>>,
        all_metrics: &[RunnerMetricRow],
        all_operational: &[ConnectedRunner],
        pref_weight: f64,
    ) -> f64 {
        // 1. Machine preference score (0.0 = first choice, 1.0 = second choice, etc.)
        let pref_score = if let Some(prefs) = preferences {
            if let Some(machine_type) = &runner.machine_type {
                prefs
                    .iter()
                    .position(|p| p == machine_type)
                    .map(|pos| pos as f64)
                    .unwrap_or(prefs.len() as f64) // Not in list = worst preference
            } else {
                prefs.len() as f64 // No machine type = worst preference
            }
        } else {
            0.0 // No preferences configured = neutral
        };

        // 2. Queue depth score (number of active requests)
        let queue_score = runner.active_requests.load(Ordering::SeqCst) as f64;

        // 3. Latency score (average latency in seconds, or 1.0 if no data)
        let latency_score = all_metrics
            .iter()
            .find(|m| m.runner_id == runner.id && m.model_class == class_name)
            .and_then(|m| m.avg_ms())
            .map(|avg_ms| avg_ms / 1000.0) // Convert to seconds
            .unwrap_or(1.0); // Default to 1.0 if no data

        // 4. Scarcity penalty: preserve runners that uniquely cover other classes/models.
        let scarcity_penalty = self.scarcity_penalty(runner, class_name, all_operational);

        // Combined score
        let score = (pref_weight * pref_score)
            + (self.routing_config.queue_weight * queue_score)
            + (self.routing_config.latency_weight * latency_score)
            + scarcity_penalty;

        tracing::debug!(
            "Runner {} score: {:.2} (pref={:.2}*{:.2}, queue={:.2}*{:.2}, latency={:.2}*{:.2}, scarcity={:.2})",
            runner.id,
            score,
            pref_weight,
            pref_score,
            self.routing_config.queue_weight,
            queue_score,
            self.routing_config.latency_weight,
            latency_score,
            scarcity_penalty
        );

        score
    }

    fn scarcity_penalty(
        &self,
        runner: &ConnectedRunner,
        requested_class: &str,
        all_operational: &[ConnectedRunner],
    ) -> f64 {
        let mut penalty = 0.0;

        for model_id in runner.available_models() {
            let Some(model_class) = classify_model(&model_id, &self.models_config) else {
                continue;
            };
            if model_class.as_str() == requested_class {
                continue;
            }

            let supported_elsewhere = all_operational.iter().any(|other| {
                other.id != runner.id
                    && other.available_models().iter().any(|other_model| {
                        classify_model(other_model, &self.models_config) == Some(model_class)
                    })
            });

            if !supported_elsewhere {
                penalty += 2.0;
            }
        }

        penalty
    }

    /// Select a runner for the request (legacy method for backward compatibility).
    #[allow(dead_code)]
    async fn select_runner(&self, model: Option<&str>) -> Result<ConnectedRunner, RouterError> {
        let candidates = if let Some(model_id) = model {
            let with_model = self.registry.with_model(model_id).await;
            if with_model.is_empty() {
                let compatible = self.registry.with_available_model(model_id).await;
                if compatible.is_empty() {
                    return Err(RouterError::NoRunners);
                }
                return Ok(self.select_from_candidates(&compatible));
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
            LoadBalanceStrategy::SmartRouting => {
                // For non-class requests, use queue-based selection
                // Find runner with fewest active requests
                candidates
                    .iter()
                    .min_by_key(|r| r.active_requests.load(Ordering::SeqCst))
                    .cloned()
                    .unwrap()
            }
        }
    }

    /// Proxy a request with a JSON Value body and deserialize the response.
    async fn proxy_request_value<Resp>(
        &self,
        runner: &ConnectedRunner,
        path: &str,
        request: serde_json::Value,
    ) -> Result<Resp, RouterError>
    where
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
            .json(&request)
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

    /// Proxy a request with a JSON Value body and return the raw response (for streaming).
    async fn proxy_request_raw_value(
        &self,
        runner: &ConnectedRunner,
        path: &str,
        request: serde_json::Value,
    ) -> Result<reqwest::Response, RouterError> {
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
            .json(&request)
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

    async fn proxy_multipart<Resp>(
        &self,
        runner: &ConnectedRunner,
        path: &str,
        form: reqwest::multipart::Form,
    ) -> Result<Resp, RouterError>
    where
        Resp: DeserializeOwned,
    {
        let base_url = runner
            .http_base_url
            .as_ref()
            .ok_or_else(|| RouterError::ConnectionFailed("Runner has no HTTP URL".to_string()))?;

        let url = format!("{}{}", base_url, path);
        tracing::debug!(
            "Proxying multipart request to {} (runner: {})",
            url,
            runner.id
        );

        let response = self
            .http_client
            .post(&url)
            .multipart(form)
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
    use uuid::Uuid;

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
                batch_size: 1,
            }],
            metrics: None,
            model_aliases: std::collections::HashMap::new(),
        }
    }

    fn create_test_router(registry: Arc<RunnerRegistry>) -> InferenceRouter {
        let test_db_path = format!(
            "test_router_{}.db",
            Uuid::new_v4().to_string().replace('-', "")
        );
        let audit_logger = Arc::new(AuditLogger::new(&test_db_path).unwrap());
        InferenceRouter::with_strategy(
            registry,
            LoadBalanceStrategy::RoundRobin,
            crate::config::ModelsConfig::default(),
            crate::config::RoutingConfig::default(),
            audit_logger,
        )
    }

    #[tokio::test]
    async fn test_select_runner_no_runners() {
        let registry = Arc::new(RunnerRegistry::new());
        let router = create_test_router(registry);

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
                None,
            )
            .await;

        let router = create_test_router(registry);
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
                None,
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
                None,
            )
            .await;

        let router = create_test_router(registry);

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

        let router = create_test_router(registry);
        let models = router.list_models().await.unwrap();

        assert_eq!(models.len(), 3);
        assert!(models.iter().any(|m| m.id == "model-a"));
        assert!(models.iter().any(|m| m.id == "model-b"));
        assert!(models.iter().any(|m| m.id == "model-c"));
    }

    #[tokio::test]
    async fn test_smart_routing_prefers_machine_type() {
        let registry = Arc::new(RunnerRegistry::new());

        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);

        // Register two runners with different machine types
        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                Some("halo".to_string()),
                create_test_status(vec!["llama3:8b".to_string()]),
                Some("http://host1:8080".to_string()),
                tx1,
                None,
            )
            .await;

        registry
            .register(
                "runner-2".to_string(),
                "Runner 2".to_string(),
                Some("gpu-server".to_string()),
                create_test_status(vec!["llama3:8b".to_string()]),
                Some("http://host2:8080".to_string()),
                tx2,
                None,
            )
            .await;

        // Create routing config that prefers gpu-server for fast models
        let mut class_preferences = std::collections::HashMap::new();
        class_preferences.insert(
            "fast".to_string(),
            vec!["gpu-server".to_string(), "halo".to_string()],
        );

        let routing_config = crate::config::RoutingConfig {
            class_preferences,
            queue_weight: 0.1,
            latency_weight: 0.1,
            speculative_wake_enabled: false,
            speculative_wake_targets: std::collections::HashMap::new(),
            ..Default::default()
        };

        let models_config = crate::config::ModelsConfig {
            fast: vec!["llama3:8b".to_string()],
            ..Default::default()
        };

        let test_db_path = format!(
            "test_smart_routing_{}.db",
            Uuid::new_v4().to_string().replace('-', "")
        );
        let audit_logger = Arc::new(AuditLogger::new(&test_db_path).unwrap());
        let router = InferenceRouter::new(registry, models_config, routing_config, audit_logger);

        // Smart routing should prefer gpu-server
        let result = router
            .select_runner_for_class(ModelClass::Fast)
            .await
            .unwrap();
        assert_eq!(result.runner.machine_type, Some("gpu-server".to_string()));
    }

    #[tokio::test]
    async fn test_smart_routing_considers_queue_depth() {
        let registry = Arc::new(RunnerRegistry::new());

        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);

        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                Some("gpu-server".to_string()),
                create_test_status(vec!["llama3:8b".to_string()]),
                Some("http://host1:8080".to_string()),
                tx1,
                None,
            )
            .await;

        registry
            .register(
                "runner-2".to_string(),
                "Runner 2".to_string(),
                Some("gpu-server".to_string()),
                create_test_status(vec!["llama3:8b".to_string()]),
                Some("http://host2:8080".to_string()),
                tx2,
                None,
            )
            .await;

        // Add 5 active requests to runner-1
        for _ in 0..5 {
            registry.increment_requests("runner-1").await;
        }

        // Create routing config with high queue weight
        let routing_config = crate::config::RoutingConfig {
            class_preferences: std::collections::HashMap::new(),
            queue_weight: 0.8,
            latency_weight: 0.1,
            speculative_wake_enabled: false,
            speculative_wake_targets: std::collections::HashMap::new(),
            ..Default::default()
        };

        let models_config = crate::config::ModelsConfig {
            fast: vec!["llama3:8b".to_string()],
            ..Default::default()
        };

        let test_db_path = format!(
            "test_smart_queue_{}.db",
            Uuid::new_v4().to_string().replace('-', "")
        );
        let audit_logger = Arc::new(AuditLogger::new(&test_db_path).unwrap());
        let router = InferenceRouter::new(registry, models_config, routing_config, audit_logger);

        // Smart routing should prefer runner-2 (fewer active requests)
        let result = router
            .select_runner_for_class(ModelClass::Fast)
            .await
            .unwrap();
        assert_eq!(result.runner.id, "runner-2");
    }

    #[tokio::test]
    async fn test_smart_routing_handles_excessive_weights() {
        // Test that pref_weight is clamped to 0 when queue_weight + latency_weight > 1.0
        let registry = Arc::new(RunnerRegistry::new());
        let (tx1, _) = mpsc::channel(32);

        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                Some("gpu-server".to_string()),
                create_test_status(vec!["llama3:8b".to_string()]),
                Some("http://host1:8080".to_string()),
                tx1,
                None,
            )
            .await;

        // Create config with weights that exceed 1.0
        let routing_config = crate::config::RoutingConfig {
            class_preferences: std::collections::HashMap::new(),
            queue_weight: 0.8,
            latency_weight: 0.5, // 0.8 + 0.5 = 1.3 > 1.0
            speculative_wake_enabled: false,
            speculative_wake_targets: std::collections::HashMap::new(),
            ..Default::default()
        };

        let models_config = crate::config::ModelsConfig {
            fast: vec!["llama3:8b".to_string()],
            ..Default::default()
        };

        let test_db_path = format!(
            "test_excessive_weights_{}.db",
            Uuid::new_v4().to_string().replace('-', "")
        );
        let audit_logger = Arc::new(AuditLogger::new(&test_db_path).unwrap());
        let router = InferenceRouter::new(registry, models_config, routing_config, audit_logger);

        // Should not panic and should still work
        let result = router.select_runner_for_class(ModelClass::Fast).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_smart_routing_preserves_complementary_coverage() {
        let registry = Arc::new(RunnerRegistry::new());
        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);

        registry
            .register(
                "runner-a".to_string(),
                "Runner A".to_string(),
                Some("multi".to_string()),
                create_test_status(vec!["model-x".to_string(), "model-y".to_string()]),
                Some("http://host1:8080".to_string()),
                tx1,
                None,
            )
            .await;

        registry
            .register(
                "runner-b".to_string(),
                "Runner B".to_string(),
                Some("single".to_string()),
                create_test_status(vec!["model-x".to_string()]),
                Some("http://host2:8080".to_string()),
                tx2,
                None,
            )
            .await;

        let models_config = crate::config::ModelsConfig {
            fast: vec!["model-x".to_string()],
            big: vec!["model-y".to_string()],
            ..Default::default()
        };

        let test_db_path = format!(
            "test_complementary_coverage_{}.db",
            Uuid::new_v4().to_string().replace('-', "")
        );
        let audit_logger = Arc::new(AuditLogger::new(&test_db_path).unwrap());
        let router = InferenceRouter::new(
            registry,
            models_config,
            crate::config::RoutingConfig::default(),
            audit_logger,
        );

        let result = router
            .select_runner_for_class(ModelClass::Fast)
            .await
            .unwrap();
        assert_eq!(result.runner.id, "runner-b");
    }
}
