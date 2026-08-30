//! Batch dispatcher for processing queued requests.
//!
//! The dispatcher runs an async loop that monitors the batch queue and
//! dispatches requests to runners when batches are ready.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use super::batch_queue::{BatchQueue, BatchedResponse, RequestBatch};
use super::{AffinityDecision, InferenceRouter, RouterError, RouterTelemetry, RunnerRegistry};

/// Batch dispatcher that processes queued requests.
pub struct BatchDispatcher {
    queue: Arc<BatchQueue>,
    registry: Arc<RunnerRegistry>,
    router: Arc<InferenceRouter>,
    telemetry: Arc<RouterTelemetry>,
    /// Cache of max batch sizes by model (updated periodically).
    batch_size_cache: RwLock<std::collections::HashMap<String, u32>>,
}

impl BatchDispatcher {
    /// Create a new batch dispatcher.
    pub fn new(
        queue: Arc<BatchQueue>,
        registry: Arc<RunnerRegistry>,
        router: Arc<InferenceRouter>,
        telemetry: Arc<RouterTelemetry>,
    ) -> Self {
        Self {
            queue,
            registry,
            router,
            telemetry,
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
            let plan = match self
                .router
                .plan_queued_request(
                    &queued.requested_selector,
                    &model,
                    queued.class_hint,
                    queued.affinity.clone(),
                )
                .await
            {
                Ok(plan) => plan,
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err.to_string());
                    }
                    let _ = queued.response_tx.send(Err(err));
                    continue;
                }
            };

            let reserved = match self.router.reserve_plan(plan).await {
                Ok(reserved) => reserved,
                Err(RouterError::StalePlan) => {
                    let retry = self
                        .router
                        .plan_queued_request(
                            &queued.requested_selector,
                            &model,
                            queued.class_hint,
                            queued.affinity.clone(),
                        )
                        .await;
                    match retry {
                        Ok(plan) => match self.router.reserve_plan(plan).await {
                            Ok(reserved) => reserved,
                            Err(error) => {
                                let _ = queued.response_tx.send(Err(error));
                                continue;
                            }
                        },
                        Err(error) => {
                            let _ = queued.response_tx.send(Err(error));
                            continue;
                        }
                    }
                }
                Err(error) => {
                    let _ = queued.response_tx.send(Err(error));
                    continue;
                }
            };
            let runner_id = reserved.plan.runner.id.clone();
            let resolved_model = reserved.plan.resolved_model.clone();

            if !matches!(
                reserved.plan.affinity_decision,
                AffinityDecision::Unkeyed | AffinityDecision::Disabled
            ) {
                self.telemetry
                    .emit(
                        reserved.plan.affinity_decision.as_str(),
                        format!(
                            "Cache affinity decision: {}",
                            reserved.plan.affinity_decision.as_str()
                        ),
                        Some(queued.request_id.clone()),
                        Some(runner_id.clone()),
                        Some(resolved_model.clone()),
                    )
                    .await;
            }

            tracing::info!(
                "Dispatching queued request for model {} to runner {} (loaded={})",
                resolved_model,
                runner_id,
                reserved.plan.runner.has_model_or_alias(&resolved_model)
            );

            let router = self.router.clone();
            tokio::spawn(async move {
                let result = router
                    .execute_chat_plan::<_, simple_ai_common::ChatCompletionResponse>(
                        reserved,
                        &queued.request,
                    )
                    .await;

                let response = match result {
                    Ok(routed) => Ok(BatchedResponse {
                        response: routed.response,
                        runner_id: routed.runner_id,
                        resolved_model: routed.resolved_model,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::batch_queue::BatchQueueConfig;
    use axum::{extract::State, routing::post, Json, Router};
    use simple_ai_common::{
        ChatCompletionResponse, ChatMessage, EngineStatus, ModelInfo, RunnerHealth, RunnerStatus,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::mpsc;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_dispatcher_creation() {
        let queue = Arc::new(BatchQueue::new(BatchQueueConfig::default()));
        let registry = Arc::new(RunnerRegistry::new());
        let router = create_test_router(registry.clone());
        let _dispatcher =
            BatchDispatcher::new(queue, registry, router, Arc::new(RouterTelemetry::new()));
    }

    fn create_test_router(registry: Arc<RunnerRegistry>) -> Arc<InferenceRouter> {
        let test_db_path = format!(
            "test_batch_dispatcher_{}.db",
            Uuid::new_v4().to_string().replace('-', "")
        );
        let audit_logger = Arc::new(crate::audit::AuditLogger::new(&test_db_path).unwrap());
        Arc::new(InferenceRouter::new(
            registry,
            crate::config::ModelsConfig::default(),
            crate::config::RoutingConfig::default(),
            audit_logger,
        ))
    }

    fn create_test_request() -> simple_ai_common::ChatCompletionRequest {
        simple_ai_common::ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("Hello".into()),
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
            prompt_cache_key: None,
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
                prompt_cache: None,
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
                content: Some("ok".into()),
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

        let dispatcher = BatchDispatcher::new(
            queue.clone(),
            registry.clone(),
            create_test_router(registry.clone()),
            Arc::new(RouterTelemetry::new()),
        );
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
                create_test_status_with_loaded("model-a", 1, true),
                Some(runner_2_url),
                tx2,
                None,
            )
            .await;

        let dispatcher = BatchDispatcher::new(
            queue.clone(),
            registry.clone(),
            create_test_router(registry.clone()),
            Arc::new(RouterTelemetry::new()),
        );
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

        assert_ne!(runner_1, runner_2);
        assert!(["runner-1", "runner-2"].contains(&runner_1.as_str()));
        assert!(["runner-1", "runner-2"].contains(&runner_2.as_str()));
        assert_eq!(runner_1_count.load(Ordering::SeqCst), 1);
        assert_eq!(runner_2_count.load(Ordering::SeqCst), 1);
        assert_eq!(registry.get_active_requests("runner-1").await, 0);
        assert_eq!(registry.get_active_requests("runner-2").await, 0);
    }
}
