//! Process-backed audio embedding engine.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use simple_ai_common::{
    AudioEmbeddingOptions, AudioEmbeddingResponse, ChatCompletionRequest, ChatCompletionResponse,
};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::sync::{RwLock, Semaphore};

use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::config::{AudioEmbeddingEngineConfig, AudioEmbeddingModelConfig};
use crate::error::{Error, Result};

struct AudioEmbeddingServer {
    port: u16,
    process: RwLock<Option<Child>>,
    last_used: RwLock<Instant>,
}

impl AudioEmbeddingServer {
    fn new(port: u16, process: Child) -> Self {
        Self {
            port,
            process: RwLock::new(Some(process)),
            last_used: RwLock::new(Instant::now()),
        }
    }

    async fn is_alive(&self) -> bool {
        let mut process = self.process.write().await;
        if let Some(child) = process.as_mut() {
            matches!(child.try_wait(), Ok(None))
        } else {
            false
        }
    }

    async fn terminate(&self, timeout_secs: u64) {
        let mut process = self.process.write().await;
        let Some(mut child) = process.take() else {
            return;
        };
        let _ = child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await;
    }

    async fn touch(&self) {
        *self.last_used.write().await = Instant::now();
    }
}

fn select_eviction_candidate(
    loaded: &[(String, Instant)],
    target_model: &str,
    cooldown: Duration,
    force: bool,
    now: Instant,
) -> Option<String> {
    loaded
        .iter()
        .filter(|(model_id, _)| model_id != target_model)
        .filter(|(_, last_used)| force || now.duration_since(*last_used) >= cooldown)
        .min_by_key(|(_, last_used)| *last_used)
        .map(|(model_id, _)| model_id.clone())
}

/// Audio embedding models are loaded as long-lived local provider processes.
pub struct AudioEmbeddingEngine {
    config: AudioEmbeddingEngineConfig,
    http_client: Client,
    servers: RwLock<HashMap<String, Arc<AudioEmbeddingServer>>>,
    startup_semaphore: Semaphore,
}

impl AudioEmbeddingEngine {
    pub fn new(config: AudioEmbeddingEngineConfig) -> Self {
        Self {
            config,
            http_client: Client::new(),
            servers: RwLock::new(HashMap::new()),
            startup_semaphore: Semaphore::new(1),
        }
    }

    fn model_config(&self, model_id: &str) -> Option<&AudioEmbeddingModelConfig> {
        self.config.models.iter().find(|model| model.id == model_id)
    }

    fn max_file_bytes(&self) -> u64 {
        self.config.max_file_mb * 1024 * 1024
    }

    async fn allocate_port() -> Result<u16> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;
        let port = listener
            .local_addr()
            .map_err(|e| Error::Internal(e.to_string()))?
            .port();
        drop(listener);
        Ok(port)
    }

    async fn wait_until_ready(&self, model_id: &str, port: u16) -> Result<()> {
        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(self.config.startup_timeout_secs);
        let url = format!("http://127.0.0.1:{}/health", port);
        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::LoadFailed(format!(
                    "audio embedding provider timed out while loading {}",
                    model_id
                )));
            }
            match self.http_client.get(&url).send().await {
                Ok(response) if response.status().is_success() => return Ok(()),
                _ => tokio::time::sleep(Duration::from_millis(250)).await,
            }
        }
    }

    fn build_command(&self, model_id: &str, port: u16) -> Result<Command> {
        if self.config.command.is_empty() {
            return Err(Error::InvalidRequest(
                "engines.audio_embeddings.command is required".to_string(),
            ));
        }
        let mut command = Command::new(&self.config.command[0]);
        for arg in self.config.command.iter().skip(1) {
            command.arg(arg);
        }
        command
            .arg("--model")
            .arg(model_id)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Ok(command)
    }

    async fn loaded_model_ids(&self) -> Vec<String> {
        let servers = self.servers.read().await;
        let mut loaded = Vec::new();
        for (model_id, server) in servers.iter() {
            if server.is_alive().await {
                loaded.push(model_id.clone());
            }
        }
        loaded
    }

    async fn loaded_model_usage(&self) -> Vec<(String, Instant)> {
        let servers = self.servers.read().await;
        let mut loaded = Vec::new();
        for (model_id, server) in servers.iter() {
            if server.is_alive().await {
                loaded.push((model_id.clone(), *server.last_used.read().await));
            }
        }
        loaded
    }

    async fn ensure_capacity(&self, model_to_load: &str) -> Result<()> {
        if self.config.max_loaded_models == 0 {
            return Err(Error::Internal(
                "engines.audio_embeddings.max_loaded_models must be at least 1".to_string(),
            ));
        }

        loop {
            let loaded = self.loaded_model_usage().await;
            if loaded.len() < self.config.max_loaded_models
                || loaded.iter().any(|(model_id, _)| model_id == model_to_load)
            {
                return Ok(());
            }

            let cooldown = Duration::from_secs(self.config.opportunistic_unload_cooldown_secs);
            let now = Instant::now();
            let candidate = select_eviction_candidate(&loaded, model_to_load, cooldown, false, now)
                .or_else(|| select_eviction_candidate(&loaded, model_to_load, cooldown, true, now));
            let Some(model_id) = candidate else {
                return Err(Error::Internal(format!(
                    "Cannot make room for audio embedding model {}: at capacity ({}) with no evictable providers",
                    model_to_load, self.config.max_loaded_models
                )));
            };

            let idle_for = loaded
                .iter()
                .find(|(loaded_model, _)| loaded_model == &model_id)
                .map(|(_, last_used)| now.duration_since(*last_used))
                .unwrap_or_default();
            if idle_for >= cooldown {
                tracing::info!(
                    "Opportunistically unloading idle audio embedding model {} after {:.1}s idle to load {}",
                    model_id,
                    idle_for.as_secs_f64(),
                    model_to_load
                );
            } else {
                tracing::info!(
                    "Force-unloading audio embedding model {} after {:.1}s idle to load {} because provider capacity is full",
                    model_id,
                    idle_for.as_secs_f64(),
                    model_to_load
                );
            }
            self.unload_model(&model_id).await?;
        }
    }
}

#[async_trait]
impl InferenceEngine for AudioEmbeddingEngine {
    fn engine_type(&self) -> &'static str {
        "audio_embeddings"
    }

    async fn health_check(&self) -> Result<EngineHealth> {
        Ok(EngineHealth {
            is_healthy: true,
            version: None,
            models_loaded: self.loaded_model_ids().await,
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(self
            .config
            .models
            .iter()
            .map(|model| ModelInfo {
                id: model.id.clone(),
                name: model.name.clone().unwrap_or_else(|| model.id.clone()),
                size_bytes: model.size_bytes,
                parameter_count: None,
                context_length: None,
                quantization: None,
                modified_at: None,
            })
            .collect())
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        Ok(self.model_config(model_id).map(|model| ModelInfo {
            id: model.id.clone(),
            name: model.name.clone().unwrap_or_else(|| model.id.clone()),
            size_bytes: model.size_bytes,
            parameter_count: None,
            context_length: None,
            quantization: None,
            modified_at: None,
        }))
    }

    async fn load_model(&self, model_id: &str) -> Result<()> {
        let _permit = self
            .startup_semaphore
            .acquire()
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;
        if self.servers.read().await.contains_key(model_id) {
            return Ok(());
        }
        self.model_config(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))?;
        self.ensure_capacity(model_id).await?;

        let port = Self::allocate_port().await?;
        let mut command = self.build_command(model_id, port)?;
        let process = command
            .spawn()
            .map_err(|e| Error::LoadFailed(format!("failed to start provider: {}", e)))?;
        let server = Arc::new(AudioEmbeddingServer::new(port, process));
        self.servers
            .write()
            .await
            .insert(model_id.to_string(), server.clone());
        if let Err(e) = self.wait_until_ready(model_id, port).await {
            server.terminate(self.config.shutdown_timeout_secs).await;
            self.servers.write().await.remove(model_id);
            return Err(e);
        }
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        if let Some(server) = self.servers.write().await.remove(model_id) {
            server.terminate(self.config.shutdown_timeout_secs).await;
        }
        Ok(())
    }

    async fn chat_completion(
        &self,
        _model_id: &str,
        _request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        Err(Error::NotSupported(
            "audio embedding engine does not support chat".to_string(),
        ))
    }

    async fn chat_completion_stream(
        &self,
        _model_id: &str,
        _request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        Err(Error::NotSupported(
            "audio embedding engine does not support chat".to_string(),
        ))
    }

    async fn audio_embedding(
        &self,
        model_id: &str,
        file_name: String,
        file_bytes: Vec<u8>,
        options: &AudioEmbeddingOptions,
    ) -> Result<AudioEmbeddingResponse> {
        if options.model != model_id {
            return Err(Error::InvalidRequest(format!(
                "options.model {} does not match routed model {}",
                options.model, model_id
            )));
        }
        let model = self
            .model_config(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))?;
        if !model
            .namespaces
            .iter()
            .any(|namespace| namespace.namespace == options.namespace)
        {
            return Err(Error::InvalidRequest(format!(
                "namespace {} is not supported by {}",
                options.namespace, model_id
            )));
        }
        if file_bytes.len() as u64 > self.max_file_bytes() {
            return Err(Error::InvalidRequest(format!(
                "file exceeds runner audio embedding limit of {} bytes",
                self.max_file_bytes()
            )));
        }

        let server = self
            .servers
            .read()
            .await
            .get(model_id)
            .cloned()
            .ok_or_else(|| Error::ModelNotLoaded(model_id.to_string()))?;
        if !server.is_alive().await {
            return Err(Error::EngineNotAvailable(format!(
                "audio embedding provider for {} is not running",
                model_id
            )));
        }
        server.touch().await;

        let options_json =
            serde_json::to_string(options).map_err(|e| Error::InvalidRequest(e.to_string()))?;
        let form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(file_bytes).file_name(file_name),
            )
            .text("options", options_json);
        let url = format!("http://127.0.0.1:{}/v1/audio/embeddings", server.port);
        let response = self
            .http_client
            .post(url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::InferenceFailed(format!("HTTP {}: {}", status, body)));
        }
        response
            .json()
            .await
            .map_err(|e| Error::InferenceFailed(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simple_ai_common::AudioEmbeddingNamespaceInfo;

    fn test_engine() -> AudioEmbeddingEngine {
        AudioEmbeddingEngine::new(AudioEmbeddingEngineConfig {
            enabled: true,
            provider: "test-audio".to_string(),
            command: vec!["python3".to_string(), "provider.py".to_string()],
            models: vec![AudioEmbeddingModelConfig {
                id: "musicfm-msd".to_string(),
                name: Some("MusicFM MSD".to_string()),
                provider: None,
                size_bytes: Some(123),
                namespaces: vec![AudioEmbeddingNamespaceInfo {
                    namespace: "musicfm.mean.v1".to_string(),
                    dim: 1024,
                    dtype: "float32".to_string(),
                    description: None,
                }],
            }],
            max_loaded_models: 1,
            opportunistic_unload_cooldown_secs: 600,
            max_file_mb: 10,
            startup_timeout_secs: 1,
            shutdown_timeout_secs: 1,
        })
    }

    #[tokio::test]
    async fn test_list_and_get_audio_embedding_models() {
        let engine = test_engine();
        let models = engine.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "musicfm-msd");
        assert_eq!(models[0].size_bytes, Some(123));

        let model = engine.get_model("musicfm-msd").await.unwrap().unwrap();
        assert_eq!(model.name, "MusicFM MSD");
        assert!(engine.get_model("missing").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_unknown_namespace_is_rejected_before_inference() {
        let engine = test_engine();
        let options = AudioEmbeddingOptions {
            model: "musicfm-msd".to_string(),
            namespace: "missing.namespace".to_string(),
            clip_offset_seconds: None,
            clip_seconds: None,
        };
        let err = engine
            .audio_embedding(
                "musicfm-msd",
                "track.mp3".to_string(),
                vec![1, 2, 3],
                &options,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn test_unknown_model_load_is_model_not_found() {
        let engine = test_engine();
        let err = engine.load_model("missing").await.unwrap_err();
        assert!(matches!(err, Error::ModelNotFound(_)));
    }

    #[test]
    fn test_eviction_candidate_respects_cooldown() {
        let now = Instant::now();
        let loaded = vec![
            ("recent".to_string(), now - Duration::from_secs(30)),
            ("idle".to_string(), now - Duration::from_secs(700)),
        ];

        assert_eq!(
            select_eviction_candidate(&loaded, "target", Duration::from_secs(600), false, now),
            Some("idle".to_string())
        );
    }

    #[test]
    fn test_eviction_candidate_force_uses_lru_before_cooldown() {
        let now = Instant::now();
        let loaded = vec![
            ("newer".to_string(), now - Duration::from_secs(10)),
            ("older".to_string(), now - Duration::from_secs(20)),
        ];

        assert_eq!(
            select_eviction_candidate(&loaded, "target", Duration::from_secs(600), true, now),
            Some("older".to_string())
        );
        assert_eq!(
            select_eviction_candidate(&loaded, "target", Duration::from_secs(600), false, now),
            None
        );
    }
}
