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

    /// Dispatch a batch of queued requests.
    async fn dispatch_batch(&self, batch: RequestBatch) -> Result<(), RouterError> {
        let model = batch.model;
        let mut first_error = None;

        for queued in batch.requests {
            let runner = match self.select_runner(&model).await {
                Ok(runner) => runner,
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err.to_string());
                    }
                    let _ = queued.response_tx.send(Err(err));
                    continue;
                }
            };

            let runner_id = runner.id.clone();
            let local_model = runner.resolve_model_alias(&model);
            let resolved_model = model.clone();

            // Reserve capacity before selecting the next queued request so a
            // drained batch is distributed across runners instead of seeing the
            // same active-request counts for every request in the batch.
            self.registry.increment_requests(&runner_id).await;

            tracing::info!(
                "Dispatching queued request for model {} to runner {} (loaded={})",
                resolved_model,
                runner_id,
                runner.has_model_or_alias(&resolved_model)
            );

            let registry = self.registry.clone();
            let http_client = self.http_client.clone();
            tokio::spawn(async move {
                let result = Self::send_request_with_client(
                    http_client,
                    &runner,
                    &local_model,
                    &queued.request,
                )
                .await;

                registry.decrement_requests(&runner_id).await;

                let response = match result {
                    Ok(resp) => Ok(BatchedResponse {
                        response: resp,
                        runner_id,
                        resolved_model,
                    }),
                    Err(e) => Err(e),
                };

                // Send response back to caller (ignore if receiver dropped)
                let _ = queued.response_tx.send(response);
            });
        }

        if let Some(error) = first_error {
            return Err(RouterError::ConnectionFailed(error));
        }

        Ok(())
    }

    /// Select a runner for the given model.
    async fn select_runner(&self, model: &str) -> Result<ConnectedRunner, RouterError> {
        let mut runners = self.registry.with_model(model).await;
        let compatible = self.registry.with_available_model(model).await;

        for runner in compatible {
            if !runners.iter().any(|loaded| loaded.id == runner.id) {
                runners.push(runner);
            }
        }

        if runners.is_empty() {
            return Err(RouterError::NoRunners);
        }

        // Select runner with fewest active requests, preferring already-loaded
        // runners only when queue depth is otherwise tied.
        runners
            .into_iter()
            .min_by_key(|r| {
                (
                    r.active_requests.load(std::sync::atomic::Ordering::SeqCst),
                    !r.has_model_or_alias(model),
                )
            })
            .ok_or(RouterError::NoRunners)
    }

    async fn send_request_with_client(
        http_client: reqwest::Client,
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

        let response = http_client
            .post(&url)
            .json(&request_value)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::batch_queue::BatchQueueConfig;
    use axum::{extract::State, routing::post, Json, Router};
    use simple_ai_common::{ChatMessage, EngineStatus, ModelInfo, RunnerHealth, RunnerStatus};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_dispatcher_creation() {
        let queue = Arc::new(BatchQueue::new(BatchQueueConfig::default()));
        let registry = Arc::new(RunnerRegistry::new());
        let _dispatcher = BatchDispatcher::new(queue, registry);
    }

    fn create_test_request() -> simple_ai_common::ChatCompletionRequest {
        simple_ai_common::ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            model: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            thinking_budget_tokens: None,
            tools: None,
            stream: None,
        }
    }

    fn create_test_status(model: &str, batch_size: u32) -> RunnerStatus {
        create_test_status_with_loaded(model, batch_size, true)
    }

    fn create_test_status_with_loaded(model: &str, batch_size: u32, loaded: bool) -> RunnerStatus {
        RunnerStatus {
            health: RunnerHealth::Healthy,
            capabilities: vec![],
            engines: vec![EngineStatus {
                engine_type: "test".to_string(),
                is_healthy: true,
                version: None,
                loaded_models: if loaded {
                    vec![model.to_string()]
                } else {
                    vec![]
                },
                available_models: vec![ModelInfo {
                    id: model.to_string(),
                    name: model.to_string(),
                    size_bytes: None,
                    parameter_count: None,
                    context_length: None,
                    quantization: None,
                    modified_at: None,
                    reasoning: None,
                }],
                error: None,
                batch_size,
            }],
            metrics: None,
            model_aliases: std::collections::HashMap::new(),
        }
    }

    async fn chat_handler(
        State(counter): State<Arc<AtomicUsize>>,
        Json(_request): Json<serde_json::Value>,
    ) -> Json<ChatCompletionResponse> {
        counter.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Json(ChatCompletionResponse::new(
            "model-a".to_string(),
            ChatMessage {
                role: "assistant".to_string(),
                content: Some("ok".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Some("stop".to_string()),
        ))
    }

    async fn start_test_runner(counter: Arc<AtomicUsize>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_handler))
            .with_state(counter);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        format!("http://{}", addr)
    }

    #[tokio::test]
    async fn test_dispatch_batch_distributes_across_runners() {
        let queue = Arc::new(BatchQueue::new(BatchQueueConfig::default()));
        let registry = Arc::new(RunnerRegistry::new());
        let runner_1_count = Arc::new(AtomicUsize::new(0));
        let runner_2_count = Arc::new(AtomicUsize::new(0));
        let runner_1_url = start_test_runner(runner_1_count.clone()).await;
        let runner_2_url = start_test_runner(runner_2_count.clone()).await;

        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);
        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                None,
                create_test_status("model-a", 4),
                Some(runner_1_url),
                tx1,
                None,
            )
            .await;
        registry
            .register(
                "runner-2".to_string(),
                "Runner 2".to_string(),
                None,
                create_test_status("model-a", 4),
                Some(runner_2_url),
                tx2,
                None,
            )
            .await;

        let dispatcher = BatchDispatcher::new(queue.clone(), registry.clone());
        let receivers: Vec<_> = futures_util::future::join_all(
            (0..4).map(|_| queue.enqueue("model-a".to_string(), create_test_request())),
        )
        .await;

        let batch = queue.take_batch("model-a", 4).await.unwrap();
        dispatcher.dispatch_batch(batch).await.unwrap();

        let mut handled_by = Vec::new();
        for receiver in receivers {
            handled_by.push(receiver.await.unwrap().unwrap().runner_id);
        }

        assert_eq!(runner_1_count.load(Ordering::SeqCst), 2);
        assert_eq!(runner_2_count.load(Ordering::SeqCst), 2);
        assert_eq!(registry.get_active_requests("runner-1").await, 0);
        assert_eq!(registry.get_active_requests("runner-2").await, 0);
        assert_eq!(handled_by.iter().filter(|id| *id == "runner-1").count(), 2);
        assert_eq!(handled_by.iter().filter(|id| *id == "runner-2").count(), 2);
    }

    #[tokio::test]
    async fn test_single_request_batches_can_run_concurrently_on_compatible_runners() {
        let queue = Arc::new(BatchQueue::new(BatchQueueConfig::default()));
        let registry = Arc::new(RunnerRegistry::new());
        let runner_1_count = Arc::new(AtomicUsize::new(0));
        let runner_2_count = Arc::new(AtomicUsize::new(0));
        let runner_1_url = start_test_runner(runner_1_count.clone()).await;
        let runner_2_url = start_test_runner(runner_2_count.clone()).await;

        let (tx1, _) = mpsc::channel(32);
        let (tx2, _) = mpsc::channel(32);
        registry
            .register(
                "runner-1".to_string(),
                "Runner 1".to_string(),
                None,
                create_test_status_with_loaded("model-a", 1, true),
                Some(runner_1_url),
                tx1,
                None,
            )
            .await;
        registry
            .register(
                "runner-2".to_string(),
                "Runner 2".to_string(),
                None,
                create_test_status_with_loaded("model-a", 1, false),
                Some(runner_2_url),
                tx2,
                None,
            )
            .await;

        let dispatcher = BatchDispatcher::new(queue.clone(), registry.clone());
        let rx1 = queue
            .enqueue("model-a".to_string(), create_test_request())
            .await;
        let rx2 = queue
            .enqueue("model-a".to_string(), create_test_request())
            .await;

        let batch = queue.take_batch("model-a", 1).await.unwrap();
        dispatcher.dispatch_batch(batch).await.unwrap();
        let batch = queue.take_batch("model-a", 1).await.unwrap();
        dispatcher.dispatch_batch(batch).await.unwrap();

        let runner_1 = rx1.await.unwrap().unwrap().runner_id;
        let runner_2 = rx2.await.unwrap().unwrap().runner_id;

        assert_eq!(runner_1, "runner-1");
        assert_eq!(runner_2, "runner-2");
        assert_eq!(runner_1_count.load(Ordering::SeqCst), 1);
        assert_eq!(runner_2_count.load(Ordering::SeqCst), 1);
        assert_eq!(registry.get_active_requests("runner-1").await, 0);
        assert_eq!(registry.get_active_requests("runner-2").await, 0);
    }
}
