//! Managed GLiNER process. Loading/unloading is exclusive with in-flight requests.
use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::{
    config::ExtractionEngineConfig,
    error::{Error, Result},
};
use async_trait::async_trait;
use reqwest::Client;
use simple_ai_common::{
    ChatCompletionRequest, ChatCompletionResponse, ExtractionRequest, ExtractionResponse,
};
use std::{process::Stdio, sync::Arc, time::Duration};
use tokio::{
    net::TcpListener,
    process::{Child, Command},
    sync::{Mutex, RwLock},
};

struct Server {
    port: u16,
    process: Mutex<Child>,
}
pub struct ExtractionEngine {
    config: ExtractionEngineConfig,
    client: Client,
    server: Arc<RwLock<Option<Server>>>,
}
impl ExtractionEngine {
    pub fn new(config: ExtractionEngineConfig) -> Result<Self> {
        if config.command.is_empty()
            || config.model_id.is_empty()
            || config.revision.is_empty()
            || config.batch_size == 0
            || config.num_threads == 0
            || config.startup_timeout_secs == 0
            || config.request_timeout_secs == 0
            || !["cpu", "cuda"].contains(&config.device.as_str())
        {
            return Err(Error::InvalidRequest("invalid extraction engine configuration: command, model, revision, positive limits and cpu/cuda device required".into()));
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            config,
            client,
            server: Arc::new(RwLock::new(None)),
        })
    }
    fn check_model(&self, id: &str) -> Result<()> {
        if id != self.config.model_id {
            return Err(Error::ModelNotFound(id.into()));
        }
        Ok(())
    }
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: self.config.model_id.clone(),
            name: "GLiNER2.5 Multilingual".into(),
            size_bytes: None,
            parameter_count: Some(287_000_000),
            context_length: Some(4096),
            quantization: Some(
                if self.config.device == "cuda" {
                    "F16"
                } else {
                    "F32"
                }
                .into(),
            ),
            modified_at: None,
            reasoning: None,
        }
    }
}
#[async_trait]
impl InferenceEngine for ExtractionEngine {
    fn engine_type(&self) -> &'static str {
        "extraction"
    }
    // The Python provider serializes HTTP requests and batches within each request.
    fn batch_size(&self) -> u32 {
        1
    }
    async fn health_check(&self) -> Result<EngineHealth> {
        let guard = self.server.read().await;
        let loaded = if let Some(s) = guard.as_ref() {
            matches!(s.process.lock().await.try_wait(), Ok(None))
        } else {
            false
        };
        Ok(EngineHealth {
            is_healthy: true,
            version: None,
            models_loaded: if loaded {
                vec![self.config.model_id.clone()]
            } else {
                vec![]
            },
        })
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![self.info()])
    }
    async fn get_model(&self, id: &str) -> Result<Option<ModelInfo>> {
        Ok((id == self.config.model_id).then(|| self.info()))
    }
    async fn load_model(&self, id: &str) -> Result<()> {
        self.check_model(id)?;
        let mut guard = self.server.write().await;
        if let Some(s) = guard.as_ref() {
            if matches!(s.process.lock().await.try_wait(), Ok(None)) {
                return Ok(());
            }
        }
        *guard = None;
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;
        let port = listener
            .local_addr()
            .map_err(|e| Error::Internal(e.to_string()))?
            .port();
        drop(listener);
        let mut command = Command::new(&self.config.command[0]);
        command.args(&self.config.command[1..]).args([
            "--model",
            id,
            "--revision",
            &self.config.revision,
            "--device",
            &self.config.device,
            "--port",
            &port.to_string(),
            "--batch-size",
            &self.config.batch_size.to_string(),
            "--num-threads",
            &self.config.num_threads.to_string(),
        ]);
        if let Some(path) = &self.config.model_path {
            command.args(["--model-path", path]);
        }
        let mut child = command
            .kill_on_drop(true)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| Error::LoadFailed(e.to_string()))?;
        let url = format!("http://127.0.0.1:{port}/health");
        let ready = async {
            loop {
                if child
                    .try_wait()
                    .map_err(|e| Error::LoadFailed(e.to_string()))?
                    .is_some()
                {
                    return Err(Error::LoadFailed(
                        "extraction provider exited during startup".into(),
                    ));
                }
                if let Ok(r) = self
                    .client
                    .get(&url)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .await
                {
                    if r.status().is_success() {
                        return Ok(());
                    }
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        };
        match tokio::time::timeout(Duration::from_secs(self.config.startup_timeout_secs), ready)
            .await
        {
            Ok(Ok(())) => {
                *guard = Some(Server {
                    port,
                    process: Mutex::new(child),
                });
                Ok(())
            }
            result => {
                let _ = child.kill().await;
                match result {
                    Ok(Err(e)) => Err(e),
                    _ => Err(Error::LoadFailed(
                        "extraction provider startup timed out".into(),
                    )),
                }
            }
        }
    }
    async fn unload_model(&self, id: &str) -> Result<()> {
        self.check_model(id)?;
        if let Some(server) = self.server.write().await.take() {
            server
                .process
                .lock()
                .await
                .kill()
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
        }
        Ok(())
    }
    async fn extract(&self, id: &str, request: &ExtractionRequest) -> Result<ExtractionResponse> {
        self.check_model(id)?;
        request.validate().map_err(Error::InvalidRequest)?;
        let server = self.server.clone();
        let client = self.client.clone();
        let request = request.clone();
        // Retain the read guard until the provider replies, even if the caller disconnects.
        tokio::spawn(async move {
            let guard = server.read().await;
            let server = guard
                .as_ref()
                .ok_or_else(|| Error::ModelNotLoaded(request.model.clone()))?;
            let response = client
                .post(format!("http://127.0.0.1:{}/v1/extractions", server.port))
                .json(&request)
                .send()
                .await
                .map_err(|e| Error::Communication(e.to_string()))?;
            if !response.status().is_success() {
                return Err(Error::UpstreamResponse {
                    status: response.status().as_u16(),
                    body: response.text().await.unwrap_or_default(),
                });
            }
            response
                .json()
                .await
                .map_err(|e| Error::Communication(e.to_string()))
        })
        .await
        .map_err(|e| Error::Internal(e.to_string()))?
    }
    async fn chat_completion(
        &self,
        _: &str,
        _: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        Err(Error::NotSupported(
            "extraction engine does not support chat".into(),
        ))
    }
    async fn chat_completion_stream(
        &self,
        _: &str,
        _: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        Err(Error::NotSupported(
            "extraction engine does not support chat".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unload_waits_for_cancelled_caller_and_provider_can_reload() {
        let config = ExtractionEngineConfig {
            command: vec![
                "python3".into(),
                format!(
                    "{}/../tests/fixtures/extraction-provider.py",
                    env!("CARGO_MANIFEST_DIR")
                ),
            ],
            ..Default::default()
        };
        let id = config.model_id.clone();
        let engine = Arc::new(ExtractionEngine::new(config).unwrap());
        engine.load_model(&id).await.unwrap();
        let port = engine.server.read().await.as_ref().unwrap().port;
        let request: ExtractionRequest = serde_json::from_value(serde_json::json!({
            "model": id, "input": "hello", "schema": {"entities": ["person"]}
        }))
        .unwrap();
        let caller_engine = engine.clone();
        let caller_id = id.clone();
        let caller = tokio::spawn(async move { caller_engine.extract(&caller_id, &request).await });
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let health: serde_json::Value = engine
                    .client
                    .get(format!("http://127.0.0.1:{port}/health"))
                    .send()
                    .await
                    .unwrap()
                    .json()
                    .await
                    .unwrap();
                if health["busy"] == true {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        caller.abort();
        // Detached provider request retains its guard after client cancellation.
        assert!(
            tokio::time::timeout(Duration::from_millis(20), engine.unload_model(&id))
                .await
                .is_err()
        );
        engine.unload_model(&id).await.unwrap();
        assert!(engine
            .health_check()
            .await
            .unwrap()
            .models_loaded
            .is_empty());
        engine.load_model(&id).await.unwrap();
        // Simulate a provider crash and ensure a subsequent load recovers.
        engine
            .server
            .read()
            .await
            .as_ref()
            .unwrap()
            .process
            .lock()
            .await
            .kill()
            .await
            .unwrap();
        assert!(engine
            .health_check()
            .await
            .unwrap()
            .models_loaded
            .is_empty());
        engine.load_model(&id).await.unwrap();
        assert_eq!(
            engine.health_check().await.unwrap().models_loaded,
            vec![id.clone()]
        );
        engine.unload_model(&id).await.unwrap();
    }
}
