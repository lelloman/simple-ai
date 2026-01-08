use std::sync::Arc;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::AppState;

#[derive(Deserialize, Serialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detect_language_request_default() {
        let req = DetectLanguageRequest {
            text: String::new(),
        };
        assert!(req.text.is_empty());
    }

    #[tokio::test]
    async fn test_detect_language_request_with_text() {
        let req = DetectLanguageRequest {
            text: "This is a test sentence".to_string(),
        };
        assert_eq!(req.text, "This is a test sentence");
    }

    #[tokio::test]
    async fn test_detect_language_response_default() {
        let resp = DetectLanguageResponse {
            code: None,
            confidence: None,
        };
        assert!(resp.code.is_none());
        assert!(resp.confidence.is_none());
    }

    #[tokio::test]
    async fn test_detect_language_response_with_values() {
        let resp = DetectLanguageResponse {
            code: Some("en".to_string()),
            confidence: Some(0.95),
        };
        assert_eq!(resp.code, Some("en".to_string()));
        assert_eq!(resp.confidence, Some(0.95));
    }

    #[tokio::test]
    async fn test_detect_language_serialization() {
        let req = DetectLanguageRequest {
            text: "Bonjour le monde".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("Bonjour"));
    }

    #[tokio::test]
    async fn test_detect_language_response_serialization() {
        let resp = DetectLanguageResponse {
            code: Some("fr".to_string()),
            confidence: Some(0.99),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("fr"));
        assert!(json.contains("0.99"));
    }

    #[tokio::test]
    async fn test_language_label_replacement() {
        let label = "__label__en";
        let code = label.replace("__label__", "");
        assert_eq!(code, "en");
    }

    #[tokio::test]
    async fn test_language_label_replacement_french() {
        let label = "__label__fr";
        let code = label.replace("__label__", "");
        assert_eq!(code, "fr");
    }

    #[tokio::test]
    async fn test_language_label_replacement_multilingual() {
        let label = "__label__zh";
        let code = label.replace("__label__", "");
        assert_eq!(code, "zh");
    }

    #[tokio::test]
    async fn test_detect_language_empty_text() {
        let req = DetectLanguageRequest {
            text: "".to_string(),
        };
        assert!(req.text.is_empty());
        assert_eq!(req.text.len(), 0);
    }

    #[tokio::test]
    async fn test_detect_language_long_text() {
        let long_text = "a ".repeat(1000);
        let req = DetectLanguageRequest {
            text: long_text.clone(),
        };
        assert_eq!(req.text.len(), 2000);
    }

    #[tokio::test]
    async fn test_detect_language_special_characters() {
        let req = DetectLanguageRequest {
            text: "Hello! ¿Cómo estás? 你好".to_string(),
        };
        assert!(req.text.contains("!"));
        assert!(req.text.contains("¿"));
        assert!(req.text.contains("你"));
    }
}
