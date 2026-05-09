//! Audio embedding endpoint.

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::routing::post;
use axum::{Json, Router};
use simple_ai_common::{AudioEmbeddingOptions, AudioEmbeddingResponse};

use crate::error::{Error, Result};
use crate::state::AppState;

const DEFAULT_FILE_NAME: &str = "upload.bin";

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/audio/embeddings", post(create_audio_embedding))
        .layer(DefaultBodyLimit::max(200 * 1024 * 1024))
}

async fn create_audio_embedding(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<AudioEmbeddingResponse>> {
    let mut file_name = DEFAULT_FILE_NAME.to_string();
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut options: Option<AudioEmbeddingOptions> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| Error::InvalidRequest(e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            if let Some(upload_name) = field.file_name() {
                file_name = upload_name.to_string();
            }
            let bytes = field
                .bytes()
                .await
                .map_err(|e| Error::InvalidRequest(e.to_string()))?;
            file_bytes = Some(bytes.to_vec());
        } else if name == "options" {
            let text = field
                .text()
                .await
                .map_err(|e| Error::InvalidRequest(e.to_string()))?;
            options = Some(serde_json::from_str(&text).map_err(|e| {
                Error::InvalidRequest(format!("invalid audio embedding options: {}", e))
            })?);
        }
    }

    let file_bytes =
        file_bytes.ok_or_else(|| Error::InvalidRequest("missing multipart file".to_string()))?;
    let options =
        options.ok_or_else(|| Error::InvalidRequest("missing multipart options".to_string()))?;

    let resolved_model = state
        .config
        .aliases
        .mappings
        .get(options.model.as_str())
        .cloned()
        .unwrap_or_else(|| options.model.clone());
    let mut resolved_options = options.clone();
    resolved_options.model = resolved_model.clone();

    let engine = state
        .engine_registry
        .find_engine_for_model(&resolved_model)
        .await
        .ok_or_else(|| Error::ModelNotFound(options.model.clone()))?;
    let response = engine
        .audio_embedding(&resolved_model, file_name, file_bytes, &resolved_options)
        .await?;
    Ok(Json(response))
}
