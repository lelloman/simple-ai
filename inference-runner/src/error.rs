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

    #[error("Load failed: {0}")]
    LoadFailed(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Engine communication error: {0}")]
    Communication(String),

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
            Error::LoadFailed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "load_failed"),
            Error::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
            Error::Communication(_) => (StatusCode::BAD_GATEWAY, "communication_error"),
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
