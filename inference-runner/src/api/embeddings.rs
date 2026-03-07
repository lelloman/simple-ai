//! Embeddings endpoint (OpenAI-compatible).

use std::sync::Arc;

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::state::AppState;

/// OpenAI-compatible embedding request.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    pub model: String,
}

/// Input can be a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Build the embeddings router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/embeddings", post(create_embeddings))
}

/// POST /v1/embeddings - OpenAI-compatible embeddings.
async fn create_embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>> {
    let model = &request.model;

    // Apply alias mapping if configured
    let resolved_model = state
        .config
        .aliases
        .mappings
        .get(model.as_str())
        .map(|s| s.as_str())
        .unwrap_or(model);

    tracing::debug!("Embedding request for model: {} (resolved: {})", model, resolved_model);

    let inputs = request.input.into_vec();

    // Find an engine that can serve this model
    let engine = state
        .engine_registry
        .find_engine_for_model(resolved_model)
        .await
        .ok_or_else(|| Error::ModelNotFound(model.to_string()))?;

    let embeddings = engine.embed(resolved_model, &inputs).await?;

    // Approximate token count
    let prompt_tokens: u32 = inputs.iter()
        .map(|s| (s.split_whitespace().count() as u32).max(1))
        .sum();

    let data: Vec<EmbeddingData> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| EmbeddingData {
            object: "embedding",
            embedding,
            index: i,
        })
        .collect();

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model: model.to_string(),
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}
