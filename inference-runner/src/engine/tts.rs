//! Process-backed text-to-speech engine.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use simple_ai_common::{
    ChatCompletionRequest, ChatCompletionResponse, SpeechRequest, SpeechResponseFormat,
    SpeechStreamFormat,
};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::sync::{RwLock, Semaphore};

use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::config::{TtsEngineConfig, TtsModelConfig};
use crate::error::{Error, Result};

struct TtsServer {
    port: u16,
    process: RwLock<Option<Child>>,
    last_used: RwLock<Instant>,
}

impl TtsServer {
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

/// TTS models are loaded as long-lived local provider processes.
pub struct TtsEngine {
    config: TtsEngineConfig,
    http_client: Client,
    servers: RwLock<HashMap<String, Arc<TtsServer>>>,
    startup_semaphore: Semaphore,
}

impl TtsEngine {
    pub fn new(config: TtsEngineConfig) -> Self {
        Self {
            config,
            http_client: Client::new(),
            servers: RwLock::new(HashMap::new()),
            startup_semaphore: Semaphore::new(1),
        }
    }

    fn model_config(&self, model_id: &str) -> Option<&TtsModelConfig> {
        self.config.models.iter().find(|model| model.id == model_id)
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
                    "TTS provider timed out while loading {}",
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
        let model = self
            .model_config(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))?;
        let command_config = model.command.as_ref().unwrap_or(&self.config.command);
        if command_config.is_empty() {
            return Err(Error::InvalidRequest(
                "engines.tts.command or engines.tts.models[].command is required".to_string(),
            ));
        }
        let mut command = Command::new(&command_config[0]);
        for arg in command_config.iter().skip(1) {
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
                "engines.tts.max_loaded_models must be at least 1".to_string(),
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
                    "Cannot make room for TTS model {}: at capacity ({}) with no evictable providers",
                    model_to_load, self.config.max_loaded_models
                )));
            };

            tracing::info!("Unloading TTS model {} to load {}", model_id, model_to_load);
            self.unload_model(&model_id).await?;
        }
    }

    fn supported_formats(model: &TtsModelConfig) -> Vec<SpeechResponseFormat> {
        if model.response_formats.is_empty() {
            vec![
                SpeechResponseFormat::Mp3,
                SpeechResponseFormat::Opus,
                SpeechResponseFormat::Aac,
                SpeechResponseFormat::Flac,
                SpeechResponseFormat::Wav,
                SpeechResponseFormat::Pcm,
            ]
        } else {
            model.response_formats.clone()
        }
    }

    fn validate_request(&self, model_id: &str, request: &SpeechRequest) -> Result<()> {
        if request.model != model_id {
            return Err(Error::InvalidRequest(format!(
                "request.model {} does not match routed model {}",
                request.model, model_id
            )));
        }
        let model = self
            .model_config(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))?;
        if request.input.trim().is_empty() {
            return Err(Error::InvalidRequest("input is required".to_string()));
        }
        if request.input.chars().count() > self.config.max_input_chars {
            return Err(Error::InvalidRequest(format!(
                "input exceeds runner TTS limit of {} characters",
                self.config.max_input_chars
            )));
        }
        if let Some(speed) = request.speed {
            if !(0.25..=4.0).contains(&speed) {
                return Err(Error::InvalidRequest(
                    "speed must be between 0.25 and 4.0".to_string(),
                ));
            }
        }
        for (name, value) in [
            ("exaggeration", request.exaggeration),
            ("cfg_weight", request.cfg_weight),
            ("temperature", request.temperature),
            ("top_p", request.top_p),
            ("min_p", request.min_p),
        ] {
            if let Some(value) = value {
                if !(0.0..=2.0).contains(&value) {
                    return Err(Error::InvalidRequest(format!(
                        "{} must be between 0.0 and 2.0",
                        name
                    )));
                }
            }
        }
        if let Some(repetition_penalty) = request.repetition_penalty {
            if !(0.5..=5.0).contains(&repetition_penalty) {
                return Err(Error::InvalidRequest(
                    "repetition_penalty must be between 0.5 and 5.0".to_string(),
                ));
            }
        }
        let voice = request.voice.id();
        if !model.voices.is_empty() && !model.voices.iter().any(|v| v == voice) {
            return Err(Error::InvalidRequest(format!(
                "voice {} is not supported by {}",
                voice, model_id
            )));
        }
        let response_format = request.response_format_or_default();
        if !Self::supported_formats(model).contains(&response_format) {
            return Err(Error::InvalidRequest(format!(
                "response_format {:?} is not supported by {}",
                response_format, model_id
            )));
        }
        if request.stream_format_or_default() == SpeechStreamFormat::Sse && !model.supports_sse {
            return Err(Error::InvalidRequest(format!(
                "stream_format sse is not supported by {}",
                model_id
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl InferenceEngine for TtsEngine {
    fn engine_type(&self) -> &'static str {
        "tts"
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
        if let Some(server) = self.servers.read().await.get(model_id).cloned() {
            if server.is_alive().await {
                return Ok(());
            }
            self.servers.write().await.remove(model_id);
        }
        self.model_config(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))?;
        self.ensure_capacity(model_id).await?;

        let port = Self::allocate_port().await?;
        let mut command = self.build_command(model_id, port)?;
        let process = command
            .spawn()
            .map_err(|e| Error::LoadFailed(format!("failed to start TTS provider: {}", e)))?;
        let server = Arc::new(TtsServer::new(port, process));
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
            "TTS engine does not support chat".to_string(),
        ))
    }

    async fn chat_completion_stream(
        &self,
        _model_id: &str,
        _request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        Err(Error::NotSupported(
            "TTS engine does not support chat".to_string(),
        ))
    }

    async fn speech(&self, model_id: &str, request: &SpeechRequest) -> Result<reqwest::Response> {
        self.validate_request(model_id, request)?;
        let server = self
            .servers
            .read()
            .await
            .get(model_id)
            .cloned()
            .ok_or_else(|| Error::ModelNotLoaded(model_id.to_string()))?;
        if !server.is_alive().await {
            return Err(Error::EngineNotAvailable(format!(
                "TTS provider for {} is not running",
                model_id
            )));
        }
        server.touch().await;

        let url = format!("http://127.0.0.1:{}/v1/audio/speech", server.port);
        let response = self
            .http_client
            .post(url)
            .json(request)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::InferenceFailed(format!("HTTP {}: {}", status, body)));
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simple_ai_common::{SpeechResponseFormat, SpeechVoice};

    fn test_engine() -> TtsEngine {
        TtsEngine::new(TtsEngineConfig {
            enabled: true,
            provider: "test-tts".to_string(),
            command: vec!["python3".to_string(), "provider.py".to_string()],
            models: vec![TtsModelConfig {
                id: "tts-local".to_string(),
                name: Some("Local TTS".to_string()),
                provider: None,
                command: None,
                size_bytes: Some(456),
                voices: vec!["alloy".to_string(), "nova".to_string()],
                response_formats: vec![SpeechResponseFormat::Mp3, SpeechResponseFormat::Wav],
                supports_sse: false,
            }],
            max_loaded_models: 1,
            opportunistic_unload_cooldown_secs: 600,
            max_input_chars: 32,
            startup_timeout_secs: 1,
            shutdown_timeout_secs: 1,
        })
    }

    fn request() -> SpeechRequest {
        SpeechRequest {
            model: "tts-local".to_string(),
            input: "hello".to_string(),
            voice: SpeechVoice::Id("alloy".to_string()),
            instructions: None,
            language: None,
            response_format: Some(SpeechResponseFormat::Mp3),
            speed: Some(1.0),
            exaggeration: None,
            cfg_weight: None,
            temperature: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            seed: None,
            stream_format: None,
        }
    }

    #[tokio::test]
    async fn test_list_and_get_tts_models() {
        let engine = test_engine();
        let models = engine.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "tts-local");
        assert_eq!(models[0].size_bytes, Some(456));

        let model = engine.get_model("tts-local").await.unwrap().unwrap();
        assert_eq!(model.name, "Local TTS");
        assert!(engine.get_model("missing").await.unwrap().is_none());
    }

    #[test]
    fn test_validate_request_rejects_unknown_voice() {
        let engine = test_engine();
        let mut request = request();
        request.voice = SpeechVoice::Id("missing".to_string());
        let err = engine.validate_request("tts-local", &request).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn test_validate_request_rejects_bad_speed() {
        let engine = test_engine();
        let mut request = request();
        request.speed = Some(4.5);
        let err = engine.validate_request("tts-local", &request).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn test_validate_request_rejects_too_much_input() {
        let engine = test_engine();
        let mut request = request();
        request.input = "x".repeat(33);
        let err = engine.validate_request("tts-local", &request).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn test_eviction_candidate_force_uses_lru() {
        let now = Instant::now();
        let loaded = vec![
            ("newer".to_string(), now - Duration::from_secs(10)),
            ("older".to_string(), now - Duration::from_secs(20)),
        ];

        assert_eq!(
            select_eviction_candidate(&loaded, "target", Duration::from_secs(600), true, now),
            Some("older".to_string())
        );
    }
}
