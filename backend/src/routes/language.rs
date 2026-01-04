use std::sync::Arc;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::AppState;

#[derive(Deserialize)]
pub struct DetectLanguageRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct DetectLanguageResponse {
    pub code: Option<String>,
    pub confidence: Option<f64>,
}

async fn detect_language(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<DetectLanguageRequest>,
) -> Result<Json<DetectLanguageResponse>, (StatusCode, String)> {
    // Authenticate user
    let auth_user = state.jwks_client.authenticate(&headers).await
        .map_err(|e| (StatusCode::UNAUTHORIZED, e.to_string()))?;

    // Find or create user in database
    let user = state.audit_logger.find_or_create_user(
        &auth_user.sub,
        auth_user.email.as_deref(),
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Check if user is enabled
    if !user.is_enabled {
        return Err((StatusCode::FORBIDDEN, "User is disabled".to_string()));
    }

    // Detect language using FastText
    let detector = state.lang_detector.lock().await;
    let predictions = detector.predict(&request.text, 1, 0.0)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let response = match predictions.first() {
        Some(pred) => DetectLanguageResponse {
            code: Some(pred.label.replace("__label__", "")),
            confidence: Some(pred.prob as f64),
        },
        None => DetectLanguageResponse {
            code: None,
            confidence: None,
        },
    };

    Ok(Json(response))
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/detect-language", post(detect_language))
        .with_state(state)
}
