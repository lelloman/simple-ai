use chrono::{DateTime, Utc};
use serde::Serialize;

/// Audit log entry for tracking all API requests.
#[derive(Debug, Clone, Serialize)]
pub struct AuditLogEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    pub user_email: Option<String>,
    pub request_path: String,
    pub request_body: String,
    pub response_status: u16,
    pub response_body: String,
    pub latency_ms: u64,
    pub model_used: Option<String>,
    pub tokens_prompt: Option<u32>,
    pub tokens_completion: Option<u32>,
}

impl AuditLogEntry {
    pub fn new(user_id: String, request_path: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            user_id,
            user_email: None,
            request_path,
            request_body: String::new(),
            response_status: 0,
            response_body: String::new(),
            latency_ms: 0,
            model_used: None,
            tokens_prompt: None,
            tokens_completion: None,
        }
    }
}
