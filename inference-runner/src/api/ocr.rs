//! OCR endpoint.

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::routing::post;
use axum::{Json, Router};
use simple_ai_common::{OcrOptions, OcrResponse};

use crate::error::{Error, Result};
use crate::ocr::write_upload;
use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/ocr", post(ocr))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
}

async fn ocr(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<OcrResponse>> {
    let provider = state
        .ocr_provider
        .as_ref()
        .ok_or_else(|| Error::NotSupported("OCR is not enabled on this runner".to_string()))?;

    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename = "upload.bin".to_string();
    let mut options = OcrOptions::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| Error::InvalidRequest(e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            if let Some(field_filename) = field.file_name() {
                filename = field_filename.to_string();
            }
            let bytes = field
                .bytes()
                .await
                .map_err(|e| Error::InvalidRequest(e.to_string()))?;
            if bytes.len() as u64 > provider.max_file_bytes() {
                return Err(Error::InvalidRequest(format!(
                    "file exceeds runner OCR limit of {} bytes",
                    provider.max_file_bytes()
                )));
            }
            file_bytes = Some(bytes.to_vec());
        } else if name == "options" {
            let text = field
                .text()
                .await
                .map_err(|e| Error::InvalidRequest(e.to_string()))?;
            options = serde_json::from_str(&text)
                .map_err(|e| Error::InvalidRequest(format!("invalid OCR options: {}", e)))?;
        }
    }

    let file_bytes =
        file_bytes.ok_or_else(|| Error::InvalidRequest("missing multipart file".to_string()))?;
    let (_temp_dir, path) = write_upload(&file_bytes, &filename).await?;
    let response = provider.recognize(&path, &filename, &options).await?;
    Ok(Json(response))
}
