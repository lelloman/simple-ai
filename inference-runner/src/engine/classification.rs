//! Managed process-backed Hugging Face NLI classification engine.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use simple_ai_common::{
    ChatCompletionRequest, ChatCompletionResponse, ClassificationRequest, ClassificationResponse,
};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::sync::{RwLock, Semaphore};

use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::config::{ClassificationEngineConfig, ClassificationModelConfig};
use crate::error::{Error, Result};

struct ClassificationServer {
    port: u16,
    process: RwLock<Option<Child>>,
    last_used: RwLock<Instant>,
}

impl ClassificationServer {
    fn new(port: u16, process: Child) -> Self {
        Self {
            port,
            process: RwLock::new(Some(process)),
            last_used: RwLock::new(Instant::now()),
        }
    }

    async fn is_alive(&self) -> bool {
        let mut process = self.process.write().await;
        process
            .as_mut()
            .is_some_and(|child| matches!(child.try_wait(), Ok(None)))
    }

    async fn touch(&self) {
        *self.last_used.write().await = Instant::now();
    }

    async fn terminate(&self, timeout_secs: u64) {
        let Some(mut child) = self.process.write().await.take() else {
            return;
        };
        let _ = child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await;
    }
}

pub struct ClassificationEngine {
    config: ClassificationEngineConfig,
    http_client: Client,
    servers: RwLock<HashMap<String, Arc<ClassificationServer>>>,
    startup_semaphore: Semaphore,
}

impl ClassificationEngine {
    pub fn new(config: ClassificationEngineConfig) -> Self {
        Self {
            config,
            http_client: Client::new(),
            servers: RwLock::new(HashMap::new()),
            startup_semaphore: Semaphore::new(1),
        }
    }

    fn model_config(&self, model_id: &str) -> Option<&ClassificationModelConfig> {
        self.config.models.iter().find(|model| model.id == model_id)
    }

    async fn allocate_port() -> Result<u16> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|error| Error::Internal(error.to_string()))?;
        let port = listener
            .local_addr()
            .map_err(|error| Error::Internal(error.to_string()))?
            .port();
        drop(listener);
        Ok(port)
    }

    fn build_command(&self, model_id: &str, port: u16) -> Result<Command> {
        if self.config.command.is_empty() {
            return Err(Error::InvalidRequest(
                "engines.classification.command is required".to_string(),
            ));
        }
        let model = self
            .model_config(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))?;
        let batch_size = model.batch_size.unwrap_or(self.config.batch_size);
        let max_length = model.max_length.unwrap_or(self.config.max_length);
        let mut command = Command::new(&self.config.command[0]);
        command.args(self.config.command.iter().skip(1));
        command
            .arg("--model")
            .arg(model_id)
            .arg("--port")
            .arg(port.to_string())
            .arg("--batch-size")
            .arg(batch_size.to_string())
            .arg("--max-length")
            .arg(max_length.to_string())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        Ok(command)
    }

    async fn wait_until_ready(&self, model_id: &str, port: u16) -> Result<()> {
        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(self.config.startup_timeout_secs);
        let url = format!("http://127.0.0.1:{port}/health");
        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::LoadFailed(format!(
                    "classification provider timed out while loading {model_id}"
                )));
            }
            match self.http_client.get(&url).send().await {
                Ok(response) if response.status().is_success() => return Ok(()),
                _ => tokio::time::sleep(Duration::from_millis(250)).await,
            }
        }
    }

    async fn loaded_usage(&self) -> Vec<(String, Instant)> {
        let servers = self.servers.read().await;
        let mut loaded = Vec::new();
        for (model_id, server) in servers.iter() {
            if server.is_alive().await {
                loaded.push((model_id.clone(), *server.last_used.read().await));
            }
        }
        loaded
    }

    async fn ensure_capacity(&self, target_model: &str) -> Result<()> {
        if self.config.max_loaded_models == 0 {
            return Err(Error::InvalidRequest(
                "engines.classification.max_loaded_models must be at least 1".to_string(),
            ));
        }
        let loaded = self.loaded_usage().await;
        if loaded.len() < self.config.max_loaded_models
            || loaded.iter().any(|(model, _)| model == target_model)
        {
            return Ok(());
        }
        let cooldown = Duration::from_secs(self.config.opportunistic_unload_cooldown_secs);
        let now = Instant::now();
        let candidate = loaded
            .iter()
            .filter(|(_, last_used)| now.duration_since(*last_used) >= cooldown)
            .min_by_key(|(_, last_used)| *last_used)
            .map(|(model, _)| model.clone())
            .or_else(|| {
                loaded
                    .iter()
                    .min_by_key(|(_, last_used)| *last_used)
                    .map(|(model, _)| model.clone())
            })
            .ok_or_else(|| Error::Internal("no classification model can be evicted".to_string()))?;
        self.unload_model(&candidate).await
    }

    fn model_info(model: &ClassificationModelConfig) -> ModelInfo {
        ModelInfo {
            id: model.id.clone(),
            name: model.name.clone().unwrap_or_else(|| model.id.clone()),
            size_bytes: model.size_bytes,
            parameter_count: model.parameter_count,
            context_length: model.context_length,
            quantization: Some("F16".to_string()),
            modified_at: None,
            reasoning: None,
        }
    }
}

#[async_trait]
impl InferenceEngine for ClassificationEngine {
    fn engine_type(&self) -> &'static str {
        "classification"
    }

    fn batch_size(&self) -> u32 {
        self.config.batch_size
    }

    async fn health_check(&self) -> Result<EngineHealth> {
        Ok(EngineHealth {
            is_healthy: true,
            version: None,
            models_loaded: self
                .loaded_usage()
                .await
                .into_iter()
                .map(|(model, _)| model)
                .collect(),
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(self.config.models.iter().map(Self::model_info).collect())
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        Ok(self.model_config(model_id).map(Self::model_info))
    }

    async fn load_model(&self, model_id: &str) -> Result<()> {
        let _permit = self
            .startup_semaphore
            .acquire()
            .await
            .map_err(|error| Error::Internal(error.to_string()))?;
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
        let process = self
            .build_command(model_id, port)?
            .spawn()
            .map_err(|error| Error::LoadFailed(error.to_string()))?;
        let server = Arc::new(ClassificationServer::new(port, process));
        self.servers
            .write()
            .await
            .insert(model_id.to_string(), server.clone());
        if let Err(error) = self.wait_until_ready(model_id, port).await {
            server.terminate(self.config.shutdown_timeout_secs).await;
            self.servers.write().await.remove(model_id);
            return Err(error);
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
            "classification engine does not support chat".to_string(),
        ))
    }

    async fn chat_completion_stream(
        &self,
        _model_id: &str,
        _request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        Err(Error::NotSupported(
            "classification engine does not support chat".to_string(),
        ))
    }

    async fn classify(
        &self,
        model_id: &str,
        request: &ClassificationRequest,
    ) -> Result<ClassificationResponse> {
        let server = self
            .servers
            .read()
            .await
            .get(model_id)
            .cloned()
            .ok_or_else(|| Error::ModelNotLoaded(model_id.to_string()))?;
        if !server.is_alive().await {
            self.servers.write().await.remove(model_id);
            return Err(Error::ModelNotLoaded(model_id.to_string()));
        }
        server.touch().await;
        let url = format!("http://127.0.0.1:{}/v1/classifications", server.port);
        let response = self
            .http_client
            .post(url)
            .json(request)
            .send()
            .await
            .map_err(|error| Error::Communication(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::UpstreamResponse {
                status: status.as_u16(),
                body: response.text().await.unwrap_or_default(),
            });
        }
        response
            .json()
            .await
            .map_err(|error| Error::Communication(error.to_string()))
    }
}
