//! Health check endpoint.

use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

/// GET /health - Health check endpoint.
///
/// TODO(Phase 2): This should check actual engine health via `engine.health_check()`
/// and return appropriate status based on whether engines are responsive.
pub async fn health() -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}
