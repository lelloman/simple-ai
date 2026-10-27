//! Managed MADLAD process. Loading/unloading is exclusive with in-flight requests.
use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::{
    config::TranslationEngineConfig,
    error::{Error, Result},
};
use async_trait::async_trait;
use reqwest::Client;
use simple_ai_common::{
    ChatCompletionRequest, ChatCompletionResponse, TranslationRequest, TranslationResponse,
    TRANSLATION_MODEL, TRANSLATION_REVISION,
};
use std::{sync::Arc, time::Duration};
use tokio::{
    net::TcpListener,
    process::{Child, Command},
    sync::{Mutex, RwLock},
};

struct Server {
    port: u16,
    process: Mutex<Child>,
}
pub struct TranslationEngine {
    config: TranslationEngineConfig,
    client: Client,
    server: Arc<RwLock<Option<Server>>>,
}
impl TranslationEngine {
    pub fn new(config: TranslationEngineConfig) -> Result<Self> {
        if config.command.is_empty()
            || config.batch_size == 0
            || config.num_threads == 0
            || config.startup_timeout_secs == 0
            || config.request_timeout_secs == 0
            || !["auto", "cpu", "cuda"].contains(&config.device.as_str())
        {
            return Err(Error::InvalidRequest("invalid translation engine configuration: command, model, revision, positive limits and auto/cpu/cuda device required".into()));
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
        if id != TRANSLATION_MODEL {
            return Err(Error::ModelNotFound(id.into()));
        }
        Ok(())
    }
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: TRANSLATION_MODEL.to_string(),
            name: "MADLAD-400-3B-MT".into(),
            size_bytes: None,
            parameter_count: Some(3_000_000_000),
            context_length: Some(1024),
            quantization: match self.config.device.as_str() {
                "cpu" => Some("F32".into()),
                "cuda" => Some("BF16".into()),
                _ => None,
            },
            modified_at: None,
            reasoning: None,
        }
    }
}
#[async_trait]
impl InferenceEngine for TranslationEngine {
    fn engine_type(&self) -> &'static str {
        "translation"
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
                vec![TRANSLATION_MODEL.to_string()]
            } else {
                vec![]
            },
        })
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![self.info()])
    }
    async fn get_model(&self, id: &str) -> Result<Option<ModelInfo>> {
        Ok((id == TRANSLATION_MODEL).then(|| self.info()))
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
            TRANSLATION_REVISION,
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
        let child = super::provider::start(
            &mut command,
            &self.client,
            "translation",
            id,
            port,
            self.config.startup_timeout_secs,
        )
        .await?;
        *guard = Some(Server {
            port,
            process: Mutex::new(child),
        });
        Ok(())
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
    async fn translate(
        &self,
        id: &str,
        request: &TranslationRequest,
    ) -> Result<TranslationResponse> {
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
                .post(format!("http://127.0.0.1:{}/v1/translations", server.port))
                .json(&request)
                .send()
                .await;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    // A timed-out HTTP caller does not stop Python generation. Reap
                    // the process before releasing the lifecycle/resource barrier.
                    tracing::warn!(model = %request.model, %error, "Translation provider request failed; terminating provider");
                    let _ = server.process.lock().await.kill().await;
                    return Err(Error::Communication(error.to_string()));
                }
            };
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
            "translation engine does not support chat".into(),
        ))
    }
    async fn chat_completion_stream(
        &self,
        _: &str,
        _: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        Err(Error::NotSupported(
            "translation engine does not support chat".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unload_waits_for_cancelled_caller_and_provider_can_reload() {
        let config = TranslationEngineConfig {
            command: vec![
                "python3".into(),
                format!(
                    "{}/../tests/fixtures/translation-provider.py",
                    env!("CARGO_MANIFEST_DIR")
                ),
            ],
            ..Default::default()
        };
        let id = TRANSLATION_MODEL.to_string();
        let engine = Arc::new(TranslationEngine::new(config).unwrap());
        engine.load_model(&id).await.unwrap();
        let port = engine.server.read().await.as_ref().unwrap().port;
        let request: TranslationRequest = serde_json::from_value(serde_json::json!({
            "model": id, "input": ["hello"], "target": "en"
        }))
        .unwrap();
        let caller_engine = engine.clone();
        let caller_id = id.clone();
        let caller =
            tokio::spawn(async move { caller_engine.translate(&caller_id, &request).await });
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
        let response = engine
            .translate(
                &id,
                &serde_json::from_value(serde_json::json!({
                    "model": id, "input": ["uno", "due"], "target": "en"
                }))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.data[0].translation, "uno");
        assert_eq!(response.data[1].index, 1);
        assert_eq!(response.data[1].translation, "due");
        engine.unload_model(&id).await.unwrap();
    }
    #[tokio::test]
    async fn timeout_reaps_provider_and_next_load_recovers() {
        let config = TranslationEngineConfig {
            command: vec![
                "python3".into(),
                format!(
                    "{}/../tests/fixtures/translation-provider.py",
                    env!("CARGO_MANIFEST_DIR")
                ),
            ],
            request_timeout_secs: 1,
            ..Default::default()
        };
        let engine = TranslationEngine::new(config).unwrap();
        engine.load_model(TRANSLATION_MODEL).await.unwrap();
        let request: TranslationRequest = serde_json::from_value(
            serde_json::json!({"model": TRANSLATION_MODEL, "input": ["slow"], "target": "en"}),
        )
        .unwrap();
        assert!(engine.translate(TRANSLATION_MODEL, &request).await.is_err());
        assert!(engine
            .health_check()
            .await
            .unwrap()
            .models_loaded
            .is_empty());
        engine.load_model(TRANSLATION_MODEL).await.unwrap();
        assert_eq!(
            engine.health_check().await.unwrap().models_loaded,
            vec![TRANSLATION_MODEL]
        );
        engine.unload_model(TRANSLATION_MODEL).await.unwrap();
    }
}
