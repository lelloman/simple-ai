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

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_request_new() {
        let req = Request::new("user123".to_string(), "/v1/chat/completions".to_string());
        assert!(!req.id.is_empty());
        assert_eq!(req.user_id, "user123");
        assert_eq!(req.request_path, "/v1/chat/completions");
        assert!(req.request_body.is_empty());
        assert!(req.model.is_none());
        assert!(!req.timestamp.to_string().is_empty());
    }

    #[test]
    fn test_request_id_is_valid_uuid() {
        let req = Request::new("user123".to_string(), "/v1/chat/completions".to_string());
        let uuid_result = Uuid::parse_str(&req.id);
        assert!(uuid_result.is_ok());
    }

    #[test]
    fn test_request_with_model() {
        let mut req = Request::new("user123".to_string(), "/v1/chat/completions".to_string());
        req.model = Some("llama2".to_string());
        assert_eq!(req.model, Some("llama2".to_string()));
    }

    #[test]
    fn test_response_new() {
        let resp = Response::new("req123".to_string(), 200);
        assert!(!resp.id.is_empty());
        assert_eq!(resp.request_id, "req123");
        assert_eq!(resp.status, 200);
        assert!(resp.response_body.is_empty());
        assert_eq!(resp.latency_ms, 0);
        assert!(resp.tokens_prompt.is_none());
        assert!(resp.tokens_completion.is_none());
    }

    #[test]
    fn test_response_id_is_valid_uuid() {
        let resp = Response::new("req123".to_string(), 200);
        let uuid_result = Uuid::parse_str(&resp.id);
        assert!(uuid_result.is_ok());
    }

    #[test]
    fn test_response_different_status_codes() {
        let resp_200 = Response::new("req1".to_string(), 200);
        let resp_500 = Response::new("req2".to_string(), 500);
        let resp_403 = Response::new("req3".to_string(), 403);
        assert_eq!(resp_200.status, 200);
        assert_eq!(resp_500.status, 500);
        assert_eq!(resp_403.status, 403);
    }

    #[test]
    fn test_response_with_tokens() {
        let mut resp = Response::new("req123".to_string(), 200);
        resp.tokens_prompt = Some(10);
        resp.tokens_completion = Some(20);
        assert_eq!(resp.tokens_prompt, Some(10));
        assert_eq!(resp.tokens_completion, Some(20));
    }

    #[test]
    fn test_response_with_body() {
        let mut resp = Response::new("req123".to_string(), 200);
        resp.response_body = r#"{"choices":[{"message":{"content":"Hello"}}]}"#.to_string();
        assert!(resp.response_body.contains("Hello"));
    }

    #[test]
    fn test_response_with_latency() {
        let mut resp = Response::new("req123".to_string(), 200);
        resp.latency_ms = 150;
        assert_eq!(resp.latency_ms, 150);
    }

    #[test]
    fn test_request_serialize() {
        let mut req = Request::new("user123".to_string(), "/v1/chat/completions".to_string());
        req.request_body = r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_string();
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("user123"));
        assert!(json.contains("/v1/chat/completions"));
    }

    #[test]
    fn test_response_serialize() {
        let mut resp = Response::new("req123".to_string(), 200);
        resp.response_body = r#"{"id":"chatcmpl-123"}"#.to_string();
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("req123"));
        assert!(json.contains("200"));
    }
}
