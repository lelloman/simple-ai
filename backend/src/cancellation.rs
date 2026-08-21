use futures_util::future::{AbortHandle, AbortRegistration};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// Tracks HTTP inference work that can be cancelled by an administrator.
#[derive(Default)]
pub struct RequestCancellationRegistry {
    requests: Mutex<HashMap<String, AbortHandle>>,
}

impl RequestCancellationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a request and return the registration used to make its future abortable.
    pub fn register(self: &Arc<Self>, request_id: &str) -> RequestCancellationRegistration {
        let (handle, registration) = AbortHandle::new_pair();
        self.requests
            .lock()
            .expect("request cancellation registry poisoned")
            .insert(request_id.to_string(), handle);
        RequestCancellationRegistration {
            request_id: request_id.to_string(),
            registry: Arc::downgrade(self),
            registration: Some(registration),
        }
    }

    pub fn cancel(&self, request_id: &str) -> bool {
        let handle = self
            .requests
            .lock()
            .expect("request cancellation registry poisoned")
            .get(request_id)
            .cloned();
        if let Some(handle) = handle {
            handle.abort();
            true
        } else {
            false
        }
    }

    pub fn active_request_ids(&self) -> Vec<String> {
        self.requests
            .lock()
            .expect("request cancellation registry poisoned")
            .keys()
            .cloned()
            .collect()
    }
}

/// Keeps a cancellation entry registered for exactly as long as its work exists.
pub struct RequestCancellationRegistration {
    request_id: String,
    registry: Weak<RequestCancellationRegistry>,
    registration: Option<AbortRegistration>,
}

impl RequestCancellationRegistration {
    pub fn take_abort_registration(&mut self) -> AbortRegistration {
        self.registration
            .take()
            .expect("abort registration already taken")
    }
}

impl Drop for RequestCancellationRegistration {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            registry
                .requests
                .lock()
                .expect("request cancellation registry poisoned")
                .remove(&self.request_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::future::Abortable;

    #[tokio::test]
    async fn registered_request_can_be_cancelled() {
        let registry = Arc::new(RequestCancellationRegistry::new());
        let mut request = registry.register("request-1");
        let registration = request.take_abort_registration();
        assert_eq!(registry.active_request_ids(), vec!["request-1"]);
        assert!(registry.cancel("request-1"));

        let result = Abortable::new(std::future::pending::<()>(), registration).await;
        assert!(result.is_err());

        drop(request);
        assert!(registry.active_request_ids().is_empty());
        assert!(!registry.cancel("request-1"));
    }
}
