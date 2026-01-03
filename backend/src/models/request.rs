use chrono::{DateTime, Utc};
use serde::Serialize;

/// A logged API request.
#[derive(Debug, Clone, Serialize)]
pub struct Request {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    pub request_path: String,
    pub request_body: String,
    pub model: Option<String>,
}

impl Request {
    pub fn new(user_id: String, request_path: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            user_id,
            request_path,
            request_body: String::new(),
            model: None,
        }
    }
}

/// A logged API response, linked to a request.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub id: String,
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub status: u16,
    pub response_body: String,
    pub latency_ms: u64,
    pub tokens_prompt: Option<u32>,
    pub tokens_completion: Option<u32>,
}

impl Response {
    pub fn new(request_id: String, status: u16) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            request_id,
            timestamp: Utc::now(),
            status,
            response_body: String::new(),
            latency_ms: 0,
            tokens_prompt: None,
            tokens_completion: None,
        }
    }
}
