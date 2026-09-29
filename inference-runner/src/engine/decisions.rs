//! Composite JEV provider and vLLM lifecycle, serialized with inference.
use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::{
    config::DecisionEngineConfig,
    error::{Error, Result},
};
use async_trait::async_trait;
use simple_ai_common::*;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tokio::{process::Command, sync::Mutex};

pub struct DecisionEngine {
    config: DecisionEngineConfig,
    client: reqwest::Client,
    lifecycle: Mutex<()>,
    prepared: AtomicBool,
}
impl DecisionEngine {
    pub fn new(config: DecisionEngineConfig) -> Result<Self> {
        if config.compose_command.is_empty()
            || config.compose_command.iter().any(|arg| arg.is_empty())
            || config.compose_dir.is_empty()
            || config.startup_timeout_secs == 0
            || config.request_timeout_secs == 0
            || config.shutdown_timeout_secs == 0
        {
            return Err(Error::InvalidRequest(
                "decisions requires a compose directory and positive timeouts".into(),
            ));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            config,
            lifecycle: Mutex::new(()),
            prepared: AtomicBool::new(false),
        })
    }
    async fn compose(&self, args: &[&str]) -> Result<String> {
        let output = tokio::time::timeout(
            Duration::from_secs(if args.first() == Some(&"up") {
                self.config.startup_timeout_secs
            } else {
                self.config.shutdown_timeout_secs
            }),
            Command::new(&self.config.compose_command[0])
                .args(&self.config.compose_command[1..])
                .args(["-f", "compose.yaml"])
                .args(args)
                .current_dir(&self.config.compose_dir)
                .kill_on_drop(true)
                .output(),
        )
        .await
        .map_err(|_| Error::EngineNotAvailable("JEV compose command timed out".into()))?
        .map_err(|e| Error::EngineNotAvailable(e.to_string()))?;
        if !output.status.success() {
            return Err(Error::EngineNotAvailable(
                String::from_utf8_lossy(&output.stderr).into(),
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into())
    }
    async fn preflight(&self) -> Result<()> {
        if !self.prepared.load(Ordering::Acquire) {
            self.compose(&[
                "run",
                "--rm",
                "--no-deps",
                "--pull",
                "never",
                "provider",
                "--preflight",
            ])
            .await?;
            self.prepared.store(true, Ordering::Release);
        }
        Ok(())
    }
    async fn running(&self) -> Result<bool> {
        Ok(!self
            .compose(&["ps", "--status", "running", "--services"])
            .await?
            .trim()
            .is_empty())
    }
    async fn ready(&self) -> bool {
        let Ok(response) = self
            .client
            .get(format!("{}/health", self.config.base_url))
            .timeout(Duration::from_secs(3))
            .send()
            .await
        else {
            return false;
        };
        if !response.status().is_success() {
            return false;
        }
        let Ok(value) = response.json::<serde_json::Value>().await else {
            return false;
        };
        value["model"] == DECISION_MODEL
            && value["revision"] == DECISION_REVISION
            && value["protocol"] == "decisions-v1"
    }
    async fn stop(&self) -> Result<()> {
        self.compose(&["stop", "-t", "10", "provider", "vllm"])
            .await?;
        if self.running().await? {
            return Err(Error::EngineNotAvailable(
                "JEV containers still running after shutdown".into(),
            ));
        }
        Ok(())
    }
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: DECISION_MODEL.into(),
            name: "JEV-9B semantic decisions".into(),
            size_bytes: None,
            parameter_count: Some(9_000_000_000),
            context_length: Some(18432),
            quantization: Some("BF16".into()),
            modified_at: None,
            reasoning: None,
        }
    }
}
#[async_trait]
impl InferenceEngine for DecisionEngine {
    fn engine_type(&self) -> &'static str {
        "decisions"
    }
    async fn health_check(&self) -> Result<EngineHealth> {
        let running = self.running().await?;
        let ready = running && self.ready().await;
        Ok(EngineHealth {
            is_healthy: !running || ready,
            version: Some(DECISION_REVISION.into()),
            models_loaded: if ready {
                vec![DECISION_MODEL.into()]
            } else {
                vec![]
            },
        })
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.preflight().await?;
        Ok(vec![self.info()])
    }
    async fn get_model(&self, id: &str) -> Result<Option<ModelInfo>> {
        Ok((id == DECISION_MODEL).then(|| self.info()))
    }
    async fn load_model(&self, id: &str) -> Result<()> {
        if id != DECISION_MODEL {
            return Err(Error::ModelNotFound(id.into()));
        }
        let _guard = self.lifecycle.lock().await;
        if self.ready().await {
            return Ok(());
        }
        self.stop().await?;
        let result = async {
            self.preflight().await?;
            self.compose(&["up", "-d", "--no-build", "--pull", "never"])
                .await?;
            tokio::time::timeout(
                Duration::from_secs(self.config.startup_timeout_secs),
                async {
                    while !self.ready().await {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    }
                },
            )
            .await
            .map_err(|_| Error::EngineNotAvailable("JEV startup timed out".into()))
        }
        .await;
        if result.is_err() {
            self.stop().await?;
        }
        result
    }
    async fn unload_model(&self, _: &str) -> Result<()> {
        let _guard = self.lifecycle.lock().await;
        self.stop().await
    }
    async fn decide(&self, _: &str, request: &DecisionRequest) -> Result<DecisionResponse> {
        let _guard = self.lifecycle.lock().await;
        let result = self
            .client
            .post(format!("{}/v1/decisions", self.config.base_url))
            .json(request)
            .timeout(Duration::from_secs(self.config.request_timeout_secs))
            .send()
            .await;
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                self.stop().await?;
                return Err(Error::UpstreamResponse {
                    status: if error.is_timeout() { 504 } else { 502 },
                    body: error.to_string(),
                });
            }
        };
        if !response.status().is_success() {
            return Err(Error::UpstreamResponse {
                status: response.status().as_u16(),
                body: response.text().await.unwrap_or_default(),
            });
        }
        let response: DecisionResponse = response
            .json()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        response
            .validate_for(request, DECISION_REVISION)
            .map_err(Error::Communication)?;
        Ok(response)
    }
    async fn chat_completion(
        &self,
        _: &str,
        _: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        Err(Error::NotSupported("JEV supports decisions".into()))
    }
    async fn chat_completion_stream(
        &self,
        _: &str,
        _: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        Err(Error::NotSupported("JEV supports decisions".into()))
    }
}
