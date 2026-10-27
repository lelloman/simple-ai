//! Managed CUDA Qwen embedding process. Loading/unloading is exclusive with in-flight requests.
use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::{
    config::TextEmbeddingEngineConfig,
    error::{Error, Result},
};
use async_trait::async_trait;
use reqwest::Client;
use simple_ai_common::{ChatCompletionRequest, ChatCompletionResponse};
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
pub struct TextEmbeddingEngine {
    config: TextEmbeddingEngineConfig,
    client: Client,
    server: Arc<RwLock<Option<Server>>>,
}
impl TextEmbeddingEngine {
    pub fn new(config: TextEmbeddingEngineConfig) -> Result<Self> {
        if config.command.is_empty() || config.batch_size == 0 || config.batch_size > 256 {
            return Err(Error::InvalidRequest(
                "embedding command and batch size 1–256 required".into(),
            ));
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            config,
            client,
            server: Arc::new(RwLock::new(None)),
        })
    }
    fn check_model(&self, id: &str) -> Result<()> {
        if id != simple_ai_common::text_embeddings::MODEL_ID {
            return Err(Error::ModelNotFound(id.into()));
        }
        Ok(())
    }
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: simple_ai_common::text_embeddings::MODEL_ID.to_string(),
            name: "Qwen3 Embedding 0.6B".into(),
            size_bytes: None,
            parameter_count: Some(600_000_000),
            context_length: Some(512),
            quantization: Some("F16".into()),
            modified_at: None,
            reasoning: None,
        }
    }
}
#[async_trait]
impl InferenceEngine for TextEmbeddingEngine {
    fn engine_type(&self) -> &'static str {
        "text_embeddings"
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
                vec![simple_ai_common::text_embeddings::MODEL_ID.to_string()]
            } else {
                vec![]
            },
        })
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![self.info()])
    }
    async fn get_model(&self, id: &str) -> Result<Option<ModelInfo>> {
        Ok((id == simple_ai_common::text_embeddings::MODEL_ID).then(|| self.info()))
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
            simple_ai_common::text_embeddings::REVISION,
            "--device",
            "cuda",
            "--port",
            &port.to_string(),
            "--batch-size",
            &self.config.batch_size.to_string(),
            "--num-threads",
            "4",
        ]);
        if let Some(path) = &self.config.model_path {
            command.args(["--model-path", path]);
        }
        let child =
            super::provider::start(&mut command, &self.client, "text_embeddings", id, port, 600)
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
    async fn embed(&self, id: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.check_model(id)?;
        simple_ai_common::text_embeddings::validate(inputs).map_err(Error::InvalidRequest)?;
        let server = self.server.clone();
        let client = self.client.clone();
        let request = serde_json::json!({"model": id, "input": inputs});
        // The guard survives caller cancellation so GPU ownership cannot change mid-inference.
        tokio::spawn(async move {
            let guard = server.read().await;
            let server = guard.as_ref().ok_or_else(|| {
                Error::ModelNotLoaded(simple_ai_common::text_embeddings::MODEL_ID.into())
            })?;
            let response = client
                .post(format!("http://127.0.0.1:{}/v1/embeddings", server.port))
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
            #[derive(serde::Deserialize)]
            struct Item {
                index: usize,
                embedding: Vec<f32>,
            }
            #[derive(serde::Deserialize)]
            struct Reply {
                data: Vec<Item>,
            }
            let reply: Reply = response
                .json()
                .await
                .map_err(|e| Error::Communication(e.to_string()))?;
            let count = request["input"].as_array().unwrap().len();
            if reply.data.len() != count
                || reply.data.iter().enumerate().any(|(i, row)| {
                    let norm = row.embedding.iter().map(|v| v * v).sum::<f32>();
                    row.index != i
                        || row.embedding.len() != 1024
                        || !norm.is_finite()
                        || (norm - 1.0).abs() > 0.002
                })
            {
                return Err(Error::Communication(
                    "invalid embedding provider output".into(),
                ));
            }
            Ok(reply.data.into_iter().map(|row| row.embedding).collect())
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
            "embedding engine does not support chat".into(),
        ))
    }
    async fn chat_completion_stream(
        &self,
        _: &str,
        _: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        Err(Error::NotSupported(
            "embedding engine does not support chat".into(),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unload_waits_for_cancelled_caller_and_provider_can_reload() {
        let config = TextEmbeddingEngineConfig {
            command: vec![
                "python3".into(),
                format!(
                    "{}/../tests/fixtures/embedding-provider.py",
                    env!("CARGO_MANIFEST_DIR")
                ),
            ],
            ..Default::default()
        };
        let id = simple_ai_common::text_embeddings::MODEL_ID.to_string();
        let engine = Arc::new(TextEmbeddingEngine::new(config).unwrap());
        engine.load_model(&id).await.unwrap();
        let port = engine.server.read().await.as_ref().unwrap().port;
        let request = vec!["first".to_string(), "second".to_string()];
        let output = engine.embed(&id, &request).await.unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0][0], 1.0);
        assert_eq!(output[1][1], 1.0);
        assert!(engine.embed(&id, &[]).await.is_err());
        assert!(engine.embed(&id, &["bad-dimension".into()]).await.is_err());
        assert!(engine.embed(&id, &["bad-norm".into()]).await.is_err());
        assert!(engine.embed(&id, &["bad-index".into()]).await.is_err());
        let caller_engine = engine.clone();
        let caller_id = id.clone();
        let caller = tokio::spawn(async move { caller_engine.embed(&caller_id, &request).await });
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
