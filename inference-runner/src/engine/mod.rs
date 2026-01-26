//! Inference engine abstraction layer.
//!
//! This module defines the `InferenceEngine` trait that abstracts different
//! inference backends (Ollama, llama.cpp, etc.) behind a common interface.

mod llama_cpp;
mod ollama;
mod registry;

pub use llama_cpp::LlamaCppEngine;
pub use ollama::OllamaEngine;
pub use registry::EngineRegistry;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use simple_ai_common::{ChatCompletionRequest, ChatCompletionResponse};

use crate::error::Result;

/// Information about an available model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g., "llama3.2:3b")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model size in bytes (if known)
    #[serde(default)]
    pub size_bytes: Option<u64>,
    /// Parameter count (if known)
    #[serde(default)]
    pub parameter_count: Option<u64>,
    /// Maximum context length
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Quantization type (e.g., "Q4_K_M")
    #[serde(default)]
    pub quantization: Option<String>,
    /// When the model was last modified
    #[serde(default)]
    pub modified_at: Option<String>,
}

/// Health status of an inference engine (used in Phase 2 health endpoint).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct EngineHealth {
    pub is_healthy: bool,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub models_loaded: Vec<String>,
}

/// Primary trait for inference engines.
///
/// Each inference engine (Ollama, llama.cpp, etc.) implements this trait
/// to provide a consistent interface for model management and inference.
///
/// Note: Some methods (`health_check`, `load_model`, `unload_model`) are for
/// Phase 2+ features (health endpoint improvements, model lifecycle management).
#[async_trait]
#[allow(dead_code)]
pub trait InferenceEngine: Send + Sync {
    /// Unique identifier for this engine type (e.g., "ollama", "llama_cpp").
    fn engine_type(&self) -> &'static str;

    /// Maximum batch size for concurrent inference (default: 1 = no batching).
    fn batch_size(&self) -> u32 {
        1
    }

    /// Check if the engine is available and responding (Phase 2).
    async fn health_check(&self) -> Result<EngineHealth>;

    /// List all models available through this engine.
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Get detailed info about a specific model.
    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>>;

    /// Load a model into memory for inference (Phase 2).
    ///
    /// Some engines (like Ollama) load models lazily on first request,
    /// but this method can be used to pre-warm the model.
    async fn load_model(&self, model_id: &str) -> Result<()>;

    /// Unload a model from memory (Phase 2).
    async fn unload_model(&self, model_id: &str) -> Result<()>;

    /// Perform chat completion inference.
    async fn chat_completion(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse>;
}
