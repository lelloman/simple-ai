//! Error types for the inference runner.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// Error types for inference operations.
///
/// Note: Some variants are for Phase 2+ features (model lifecycle management).
#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum Error {
    #[error("Engine not available: {0}")]
    EngineNotAvailable(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Upstream returned HTTP {status}: {body}")]
    UpstreamResponse { status: u16, body: String },

    #[error("Load failed: {0}")]
    LoadFailed(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Engine communication error: {0}")]
    Communication(String),

    #[error("Not supported: {0}")]
    NotSupported(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            Error::EngineNotAvailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "engine_unavailable"),
            Error::ModelNotFound(_) => (StatusCode::NOT_FOUND, "model_not_found"),
            Error::ModelNotLoaded(_) => (StatusCode::BAD_REQUEST, "model_not_loaded"),
            Error::InferenceFailed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "inference_failed"),
            Error::UpstreamResponse { status, .. } => (
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
                "upstream_error",
            ),
            Error::LoadFailed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "load_failed"),
            Error::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
            Error::Communication(_) => (StatusCode::BAD_GATEWAY, "communication_error"),
            Error::NotSupported(_) => (StatusCode::NOT_IMPLEMENTED, "not_supported"),
            Error::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        };

        let body = Json(json!({
            "error": {
                "type": error_type,
                "message": self.to_string()
            }
        }));

        (status, body).into_response()
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[tokio::test]
    async fn upstream_bad_request_preserves_status_and_message() {
        let response = Error::UpstreamResponse {
            status: 400,
            body: "context length exceeded".to_string(),
        }
        .into_response();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains("context length exceeded"));
    }
}
