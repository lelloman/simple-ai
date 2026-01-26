//! Request batching queue for the gateway.
//!
//! This module provides a queue that collects incoming requests and batches them
//! based on runner-reported batch sizes for improved throughput.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{oneshot, RwLock, Notify};

use simple_ai_common::ChatCompletionRequest;

use crate::gateway::RouterError;

/// Configuration for the batch queue.
#[derive(Debug, Clone)]
pub struct BatchQueueConfig {
    /// Maximum time to wait for batch to fill.
    pub batch_timeout: Duration,
    /// Minimum batch size before sending (if timeout not reached).
    pub min_batch_size: u32,
}

impl BatchQueueConfig {
    pub fn new(batch_timeout_ms: u64, min_batch_size: u32) -> Self {
        Self {
            batch_timeout: Duration::from_millis(batch_timeout_ms),
            min_batch_size,
        }
    }
}

impl Default for BatchQueueConfig {
    fn default() -> Self {
        Self {
            batch_timeout: Duration::from_millis(50),
            min_batch_size: 1,
        }
    }
}

/// A queued request waiting to be dispatched.
pub struct QueuedRequest {
    /// The original chat completion request.
    pub request: ChatCompletionRequest,
    /// Channel to send the response back to the caller.
    pub response_tx: oneshot::Sender<Result<BatchedResponse, RouterError>>,
    /// When this request was enqueued.
    pub enqueued_at: Instant,
}

/// Response from a batched request.
#[derive(Debug, Clone)]
pub struct BatchedResponse {
    /// The chat completion response.
    pub response: simple_ai_common::ChatCompletionResponse,
    /// ID of the runner that handled the request.
    pub runner_id: String,
    /// The resolved model name.
    pub resolved_model: String,
}

/// A batch of requests ready to be dispatched.
pub struct RequestBatch {
    /// The model these requests are for.
    pub model: String,
    /// The requests in this batch.
    pub requests: Vec<QueuedRequest>,
}

/// Per-model queue holding pending requests.
struct ModelQueue {
    requests: Vec<QueuedRequest>,
    first_request_at: Option<Instant>,
}

impl ModelQueue {
    fn new() -> Self {
        Self {
            requests: Vec::new(),
            first_request_at: None,
        }
    }

    fn push(&mut self, request: QueuedRequest) {
        if self.requests.is_empty() {
            self.first_request_at = Some(Instant::now());
        }
        self.requests.push(request);
    }

    fn len(&self) -> usize {
        self.requests.len()
    }

    fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn age(&self) -> Option<Duration> {
        self.first_request_at.map(|t| t.elapsed())
    }

    fn take_batch(&mut self, max_size: usize) -> Vec<QueuedRequest> {
        let take_count = max_size.min(self.requests.len());
        let batch: Vec<_> = self.requests.drain(..take_count).collect();

        // Reset first_request_at if queue is now empty
        if self.requests.is_empty() {
            self.first_request_at = None;
        } else {
            // Update to the oldest remaining request
            self.first_request_at = self.requests.first().map(|r| r.enqueued_at);
        }

        batch
    }
}

/// Main batch queue manager.
///
/// Collects requests by model and provides batching based on runner capabilities.
pub struct BatchQueue {
    config: BatchQueueConfig,
    queues: RwLock<HashMap<String, ModelQueue>>,
    /// Notifier for when new requests are added.
    notify: Arc<Notify>,
}

impl BatchQueue {
    /// Create a new batch queue with the given configuration.
    pub fn new(config: BatchQueueConfig) -> Self {
        Self {
            config,
            queues: RwLock::new(HashMap::new()),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Get a reference to the notifier for waiting on new requests.
    pub fn notifier(&self) -> Arc<Notify> {
        self.notify.clone()
    }

    /// Enqueue a request for the given model.
    ///
    /// Returns a receiver that will receive the response when the request is processed.
    pub async fn enqueue(
        &self,
        model: String,
        request: ChatCompletionRequest,
    ) -> oneshot::Receiver<Result<BatchedResponse, RouterError>> {
        let (tx, rx) = oneshot::channel();

        let queued = QueuedRequest {
            request,
            response_tx: tx,
            enqueued_at: Instant::now(),
        };

        {
            let mut queues = self.queues.write().await;
            queues
                .entry(model)
                .or_insert_with(ModelQueue::new)
                .push(queued);
        }

        // Notify dispatcher that a new request is available
        self.notify.notify_one();

        rx
    }

    /// Check if a batch should be dispatched for the given model.
    ///
    /// Returns true if:
    /// - The queue size >= runner_batch_size, OR
    /// - The queue has been waiting >= batch_timeout
    pub async fn should_dispatch(&self, model: &str, runner_batch_size: u32) -> bool {
        let queues = self.queues.read().await;

        if let Some(queue) = queues.get(model) {
            if queue.is_empty() {
                return false;
            }

            // Dispatch if we have enough requests
            if queue.len() >= runner_batch_size as usize {
                return true;
            }

            // Dispatch if we've waited long enough (and have at least min_batch_size)
            if queue.len() >= self.config.min_batch_size as usize {
                if let Some(age) = queue.age() {
                    if age >= self.config.batch_timeout {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Take a batch of requests for the given model.
    ///
    /// Returns None if the queue is empty.
    pub async fn take_batch(&self, model: &str, max_size: u32) -> Option<RequestBatch> {
        let mut queues = self.queues.write().await;

        if let Some(queue) = queues.get_mut(model) {
            if queue.is_empty() {
                return None;
            }

            let requests = queue.take_batch(max_size as usize);
            if requests.is_empty() {
                return None;
            }

            Some(RequestBatch {
                model: model.to_string(),
                requests,
            })
        } else {
            None
        }
    }

    /// Get list of models with pending requests.
    pub async fn pending_models(&self) -> Vec<String> {
        let queues = self.queues.read().await;
        queues
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(m, _)| m.clone())
            .collect()
    }

    /// Get the number of pending requests for a model.
    pub async fn pending_count(&self, model: &str) -> usize {
        let queues = self.queues.read().await;
        queues.get(model).map(|q| q.len()).unwrap_or(0)
    }

    /// Get the age of the oldest request for a model.
    pub async fn oldest_request_age(&self, model: &str) -> Option<Duration> {
        let queues = self.queues.read().await;
        queues.get(model).and_then(|q| q.age())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simple_ai_common::ChatMessage;

    fn create_test_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            model: None,
            temperature: None,
            max_tokens: None,
            tools: None,
            stream: None,
        }
    }

    #[tokio::test]
    async fn test_enqueue_and_pending() {
        let queue = BatchQueue::new(BatchQueueConfig::default());

        assert!(queue.pending_models().await.is_empty());
        assert_eq!(queue.pending_count("model-a").await, 0);

        let _rx = queue.enqueue("model-a".to_string(), create_test_request()).await;

        assert_eq!(queue.pending_models().await, vec!["model-a".to_string()]);
        assert_eq!(queue.pending_count("model-a").await, 1);
    }

    #[tokio::test]
    async fn test_should_dispatch_by_size() {
        let queue = BatchQueue::new(BatchQueueConfig::default());

        // Enqueue 4 requests
        for _ in 0..4 {
            let _rx = queue.enqueue("model-a".to_string(), create_test_request()).await;
        }

        // Should dispatch when runner_batch_size is 4
        assert!(queue.should_dispatch("model-a", 4).await);

        // Should not dispatch when runner_batch_size is 8
        assert!(!queue.should_dispatch("model-a", 8).await);
    }

    #[tokio::test]
    async fn test_should_dispatch_by_timeout() {
        let config = BatchQueueConfig {
            batch_timeout: Duration::from_millis(10),
            min_batch_size: 1,
        };
        let queue = BatchQueue::new(config);

        let _rx = queue.enqueue("model-a".to_string(), create_test_request()).await;

        // Should not dispatch immediately (not enough for batch)
        assert!(!queue.should_dispatch("model-a", 4).await);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(15)).await;

        // Should dispatch after timeout
        assert!(queue.should_dispatch("model-a", 4).await);
    }

    #[tokio::test]
    async fn test_take_batch() {
        let queue = BatchQueue::new(BatchQueueConfig::default());

        // Enqueue 5 requests
        for _ in 0..5 {
            let _rx = queue.enqueue("model-a".to_string(), create_test_request()).await;
        }

        // Take batch of 3
        let batch = queue.take_batch("model-a", 3).await.unwrap();
        assert_eq!(batch.requests.len(), 3);
        assert_eq!(batch.model, "model-a");

        // 2 remaining
        assert_eq!(queue.pending_count("model-a").await, 2);

        // Take remaining
        let batch = queue.take_batch("model-a", 10).await.unwrap();
        assert_eq!(batch.requests.len(), 2);

        // Queue empty
        assert_eq!(queue.pending_count("model-a").await, 0);
        assert!(queue.take_batch("model-a", 10).await.is_none());
    }

    #[tokio::test]
    async fn test_min_batch_size() {
        let config = BatchQueueConfig {
            batch_timeout: Duration::from_millis(10),
            min_batch_size: 3,
        };
        let queue = BatchQueue::new(config);

        // Enqueue 2 requests (below min_batch_size)
        for _ in 0..2 {
            let _rx = queue.enqueue("model-a".to_string(), create_test_request()).await;
        }

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(15)).await;

        // Should NOT dispatch because we're below min_batch_size
        assert!(!queue.should_dispatch("model-a", 8).await);

        // Add one more to reach min_batch_size
        let _rx = queue.enqueue("model-a".to_string(), create_test_request()).await;

        // Wait for timeout on the new request
        tokio::time::sleep(Duration::from_millis(15)).await;

        // Now should dispatch
        assert!(queue.should_dispatch("model-a", 8).await);
    }
}
