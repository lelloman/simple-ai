//! Batch dispatcher for processing queued requests.
//!
//! The dispatcher runs an async loop that monitors the batch queue and
//! dispatches requests to runners when batches are ready.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use simple_ai_common::ChatCompletionResponse;

use super::batch_queue::{BatchQueue, BatchedResponse, RequestBatch};
use super::{ConnectedRunner, RouterError, RunnerRegistry};

/// Batch dispatcher that processes queued requests.
pub struct BatchDispatcher {
    queue: Arc<BatchQueue>,
    registry: Arc<RunnerRegistry>,
    http_client: reqwest::Client,
    /// Cache of max batch sizes by model (updated periodically).
    batch_size_cache: RwLock<std::collections::HashMap<String, u32>>,
}

impl BatchDispatcher {
    /// Create a new batch dispatcher.
    pub fn new(queue: Arc<BatchQueue>, registry: Arc<RunnerRegistry>) -> Self {
        Self {
            queue,
            registry,
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
            batch_size_cache: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Run the dispatcher loop.
    ///
    /// This method runs indefinitely, processing batches as they become ready.
    pub async fn run(&self) {
        let notify = self.queue.notifier();
        let check_interval = Duration::from_millis(10);

        loop {
            // Wait for notification or timeout (for periodic timeout-based dispatch)
            tokio::select! {
                _ = notify.notified() => {
                    // New request added, check all models
                }
                _ = tokio::time::sleep(check_interval) => {
                    // Periodic check for timeout-based dispatch
                }
            }

            // Try to dispatch for all pending models
            if let Err(e) = self.try_dispatch_all().await {
                tracing::warn!("Batch dispatch error: {}", e);
            }
        }
    }

    /// Try to dispatch batches for all pending models.
    async fn try_dispatch_all(&self) -> Result<(), RouterError> {
        let models = self.queue.pending_models().await;

        for model in models {
            if let Err(e) = self.try_dispatch(&model).await {
                tracing::warn!("Failed to dispatch batch for model {}: {}", model, e);
            }
        }

        Ok(())
    }

    /// Try to dispatch a batch for a specific model.
    async fn try_dispatch(&self, model: &str) -> Result<(), RouterError> {
        let batch_size = self.get_runner_batch_size(model).await;

        if !self.queue.should_dispatch(model, batch_size).await {
            return Ok(());
        }

        // Take the batch
        let batch = match self.queue.take_batch(model, batch_size).await {
            Some(b) => b,
            None => return Ok(()),
        };

        tracing::info!(
            "Dispatching batch of {} requests for model {} (max_batch_size={})",
            batch.requests.len(),
            model,
            batch_size
        );

        // Dispatch the batch
        self.dispatch_batch(batch).await
    }

    /// Get the maximum batch size for runners that have the given model.
    async fn get_runner_batch_size(&self, model: &str) -> u32 {
        // Check cache first
        {
            let cache = self.batch_size_cache.read().await;
            if let Some(&size) = cache.get(model) {
                return size;
            }
        }

        // Query registry for runners with this model
        let runners = self.registry.with_model(model).await;
        let max_batch_size = runners
            .iter()
            .flat_map(|r| r.status.engines.iter())
            .map(|e| e.batch_size)
            .max()
            .unwrap_or(1);

        // Update cache
        {
            let mut cache = self.batch_size_cache.write().await;
            cache.insert(model.to_string(), max_batch_size);
        }

        max_batch_size
    }

    /// Invalidate the batch size cache (call when runners connect/disconnect).
    pub async fn invalidate_cache(&self) {
        let mut cache = self.batch_size_cache.write().await;
        cache.clear();
    }

    /// Dispatch a batch of requests to a runner.
    async fn dispatch_batch(&self, batch: RequestBatch) -> Result<(), RouterError> {
        let model = &batch.model;

        // Select a runner for this model
        let runner = self.select_runner(model).await?;
        let runner_id = runner.id.clone();

        // Resolve model alias
        let local_model = runner.resolve_model_alias(model);

        // Track active requests
        self.registry.increment_requests(&runner_id).await;

        // Process each request in the batch
        // Note: For true batching, we'd send all requests together.
        // For now, we process them sequentially but benefit from the queue
        // organizing requests and reducing contention.
        for queued in batch.requests {
            let result = self
                .send_request(&runner, &local_model, &queued.request)
                .await;

            let response = match result {
                Ok(resp) => Ok(BatchedResponse {
                    response: resp,
                    runner_id: runner_id.clone(),
                    resolved_model: model.clone(),
                }),
                Err(e) => Err(e),
            };

            // Send response back to caller (ignore if receiver dropped)
            let _ = queued.response_tx.send(response);
        }

        self.registry.decrement_requests(&runner_id).await;

        Ok(())
    }

    /// Select a runner for the given model.
    async fn select_runner(&self, model: &str) -> Result<ConnectedRunner, RouterError> {
        let runners = self.registry.with_model(model).await;

        if runners.is_empty() {
            // Check if any operational runners exist
            let operational = self.registry.operational().await;
            if operational.is_empty() {
                return Err(RouterError::NoRunners);
            }
            // Return first operational runner (it will load the model on demand)
            return Ok(operational.into_iter().next().unwrap());
        }

        // Select runner with fewest active requests
        runners
            .into_iter()
            .min_by_key(|r| r.active_requests.load(std::sync::atomic::Ordering::SeqCst))
            .ok_or(RouterError::NoRunners)
    }

    /// Send a single request to a runner.
    async fn send_request(
        &self,
        runner: &ConnectedRunner,
        local_model: &str,
        request: &simple_ai_common::ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, RouterError> {
        let base_url = runner
            .http_base_url
            .as_ref()
            .ok_or_else(|| RouterError::ConnectionFailed("Runner has no HTTP URL".to_string()))?;

        let url = format!("{}/v1/chat/completions", base_url);

        // Modify request with local model name
        let mut request_value = serde_json::to_value(request)
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;
        if let Some(obj) = request_value.as_object_mut() {
            obj.insert(
                "model".to_string(),
                serde_json::Value::String(local_model.to_string()),
            );
        }

        let response = self
            .http_client
            .post(&url)
            .json(&request_value)
            .send()
            .await
            .map_err(|e| RouterError::ConnectionFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(RouterError::RunnerError(format!("HTTP {}: {}", status, body)));
        }

        response
            .json()
            .await
            .map_err(RouterError::RequestFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::batch_queue::BatchQueueConfig;

    #[tokio::test]
    async fn test_dispatcher_creation() {
        let queue = Arc::new(BatchQueue::new(BatchQueueConfig::default()));
        let registry = Arc::new(RunnerRegistry::new());
        let _dispatcher = BatchDispatcher::new(queue, registry);
    }
}
