//! Managed async vLLM tag process. Loading/unloading is exclusive with in-flight requests.
use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::{
    config::TagEngineConfig,
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
    process: Mutex<TagProcess>,
}
struct TagProcess {
    child: Child,
    group: super::provider::ProcessGroup,
}
impl TagProcess {
    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }
    async fn kill(&mut self) -> std::io::Result<()> {
        self.group.terminate()?;
        self.child.kill().await?;
        self.group.0 = None;
        Ok(())
    }
}
pub struct TagEngine {
    revisions: super::revision::Revisions,
    config: TagEngineConfig,
    client: Client,
    server: Arc<RwLock<Option<Server>>>,
}
impl TagEngine {
    pub fn new(config: TagEngineConfig) -> Result<Self> {
        if config.command.is_empty() || config.model_path.is_empty() {
            return Err(Error::InvalidRequest(
                "tag command and local model_path required".into(),
            ));
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            revisions: Default::default(),
            config,
            client,
            server: Arc::new(RwLock::new(None)),
        })
    }
    fn check_model(&self, id: &str) -> Result<()> {
        if id != self.config.model {
            return Err(Error::ModelNotFound(id.into()));
        }
        Ok(())
    }
    fn info(&self) -> ModelInfo {
        ModelInfo {
            revision: self.revisions.get(&self.config.model),
            id: self.config.model.clone(),
            name: "Qwen3 4B Site Tags".into(),
            size_bytes: None,
            parameter_count: Some(4_000_000_000),
            context_length: Some(2048),
            quantization: Some("W8A8-INT8".into()),
            modified_at: None,
            reasoning: None,
        }
    }
}
#[async_trait]
impl InferenceEngine for TagEngine {
    fn engine_type(&self) -> &'static str {
        "tag"
    }
    // Independent concurrent HTTP requests enter vLLM continuous batching.
    fn batch_size(&self) -> u32 {
        32
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
                vec![self.config.model.clone()]
            } else {
                vec![]
            },
        })
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![self.info()])
    }
    async fn get_model(&self, id: &str) -> Result<Option<ModelInfo>> {
        Ok((id == self.config.model).then(|| self.info()))
    }
    async fn load_model(&self, id: &str) -> Result<()> {
        self.check_model(id)?;
        // Registry calls load_model for every lease. A healthy provider must not
        // take the write lock while other requests hold lifecycle read guards.
        {
            let guard = self.server.read().await;
            if let Some(server) = guard.as_ref() {
                if matches!(server.process.lock().await.try_wait(), Ok(None)) {
                    return Ok(());
                }
            }
        }
        let mut guard = self.server.write().await;
        if let Some(s) = guard.as_ref() {
            if matches!(s.process.lock().await.try_wait(), Ok(None)) {
                return Ok(());
            }
        }
        *guard = None;
        self.revisions
            .load(id, std::path::Path::new(&self.config.model_path))
            .await;
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
            "--port",
            &port.to_string(),
            "--model-path",
            &self.config.model_path,
        ]);
        let child =
            super::provider::start(&mut command, &self.client, "tag", id, port, 600).await?;
        *guard = Some(Server {
            port,
            process: Mutex::new(TagProcess {
                group: super::provider::ProcessGroup(child.id()),
                child,
            }),
        });
        Ok(())
    }

    async fn unload_model(&self, id: &str) -> Result<()> {
        self.check_model(id)?;
        self.quiesce().await
    }
    async fn quiesce(&self) -> Result<()> {
        // A dead parent can leave engine-core descendants holding GPU memory.
        // Keep the lifecycle barrier through cleanup, regardless of health.
        let mut guard = self.server.write().await;
        if let Some(server) = guard.take() {
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
    async fn chat_completion(
        &self,
        id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        self.check_model(id)?;
        if request.stream.unwrap_or(false) {
            return Err(Error::InvalidRequest(
                "tag supports non-streaming chat only".into(),
            ));
        }
        let server = self.server.clone();
        let client = self.client.clone();
        let mut request = request.clone();
        request.model = Some(id.to_string());
        // Retain the read guard until the provider replies, even if the caller disconnects.
        tokio::spawn(async move {
            let guard = server.read().await;
            let server = guard
                .as_ref()
                .ok_or_else(|| Error::ModelNotLoaded(request.model.clone().unwrap_or_default()))?;
            let response = client
                .post(format!("http://127.0.0.1:{}/v1/chat/completions", server.port))
                .json(&normalized_request(&request)?)
                .send()
                .await;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    // A timed-out HTTP caller does not stop Python generation. Reap
                    // the process before releasing the lifecycle/resource barrier.
                    tracing::warn!(model = ?request.model, %error, "Tag provider request failed; terminating provider");
                    let _ = server.process.lock().await.kill().await;
                    return Err(Error::Communication(error.to_string()));
                }
            };
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                if status.is_server_error() {
                    // A live HTTP server can contain a dead vLLM/CUDA engine.
                    // Reap it while holding the lifecycle barrier so the next
                    // load starts fresh and another engine cannot race CUDA cleanup.
                    tracing::warn!(model = ?request.model, %status, %body, "Tag inference failed; terminating provider");
                    server.process.lock().await.kill().await.map_err(|error| {
                        tracing::error!(model = ?request.model, %error, "Failed to reap tag provider");
                        Error::Internal(error.to_string())
                    })?;
                }
                return Err(Error::UpstreamResponse {
                    status: status.as_u16(),
                    body,
                });
            }
            let reply: ChatCompletionResponse = match response.json().await {
                Ok(reply) => reply,
                Err(error) => {
                    tracing::warn!(model = ?request.model, %error, "Tag provider body failed; terminating provider");
                    let _ = server.process.lock().await.kill().await;
                    return Err(Error::Communication(error.to_string()));
                }
            };
            Ok(reply)
        })
        .await
        .map_err(|e| Error::Internal(e.to_string()))?
    }
    async fn chat_completion_stream(
        &self,
        _: &str,
        _: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        Err(Error::InvalidRequest(
            "tag supports non-streaming chat only".into(),
        ))
    }
}

fn normalized_request(request: &ChatCompletionRequest) -> Result<serde_json::Value> {
    if request.has_images() {
        return Err(Error::InvalidRequest(
            "tag supports text messages only".into(),
        ));
    }
    let mut body =
        serde_json::to_value(request).map_err(|e| Error::InvalidRequest(e.to_string()))?;
    let object = body.as_object_mut().expect("serialized chat object");
    object.retain(|_, value| !value.is_null());
    object.remove("thinking_budget_tokens");
    object.remove("reasoning_effort");
    object.insert(
        "chat_template_kwargs".into(),
        serde_json::json!({"enable_thinking": false}),
    );
    object.entry("max_tokens").or_insert(serde_json::json!(100));
    object.entry("temperature").or_insert(serde_json::json!(0));
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_forwarded_and_thinking_always_disabled() {
        let schema = serde_json::json!({"type":"json_schema", "json_schema":{"name":"tags","schema":{"type":"object"}}});
        let request = serde_json::from_value(serde_json::json!({"messages":[], "response_format":schema, "reasoning_effort":"high", "thinking_budget_tokens":200})).unwrap();
        let body = normalized_request(&request).unwrap();
        assert_eq!(body["response_format"], schema);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["max_tokens"], 100);
        assert!(body.get("thinking_budget_tokens").is_none());
        assert!(body.get("reasoning_effort").is_none());
    }
    fn fixture() -> TagEngine {
        TagEngine::new(TagEngineConfig {
            command: vec![
                "python3".into(),
                format!(
                    "{}/../tests/fixtures/tag-provider.py",
                    env!("CARGO_MANIFEST_DIR")
                ),
            ],
            model_path: "fixture".into(),
            ..Default::default()
        })
        .unwrap()
    }
    fn request(text: &str) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({"messages":[{"role":"user", "content":text}],
            "response_format":{"type":"json_schema", "json_schema":{"name":"tags","schema":{"type":"object"}}}})).unwrap()
    }
    #[tokio::test]
    async fn concurrent_chats_and_failure_reload() {
        let engine = Arc::new(fixture());
        let id = engine.config.model.clone();
        engine.load_model(&id).await.unwrap();
        let port = engine.server.read().await.as_ref().unwrap().port;
        let health: serde_json::Value = engine
            .client
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let child = health["child"].as_u64().unwrap();
        let mut tasks = Vec::new();
        for _ in 0..4 {
            let engine = engine.clone();
            let id = id.clone();
            tasks.push(tokio::spawn(async move {
                engine.chat_completion(&id, &request("slow")).await.unwrap()
            }));
        }
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
                if health["active"] == 4 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("requests must overlap in provider");
        tokio::time::timeout(Duration::from_millis(100), engine.load_model(&id))
            .await
            .expect("acquiring another lease must not wait for active generation")
            .unwrap();
        for task in tasks {
            task.await.unwrap();
        }
        assert!(matches!(
            engine.chat_completion(&id, &request("invalid")).await,
            Err(Error::UpstreamResponse { status: 400, .. })
        ));
        assert!(!engine
            .health_check()
            .await
            .unwrap()
            .models_loaded
            .is_empty());
        assert!(matches!(
            engine.chat_completion(&id, &request("poison")).await,
            Err(Error::UpstreamResponse { status: 500, .. })
        ));
        assert!(engine
            .health_check()
            .await
            .unwrap()
            .models_loaded
            .is_empty());
        #[cfg(target_os = "linux")]
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let status =
                    std::fs::read_to_string(format!("/proc/{child}/status")).unwrap_or_default();
                if status.is_empty()
                    || status
                        .lines()
                        .any(|line| line.starts_with("State:") && line.contains('Z'))
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("engine-core child must be terminated");
        engine.load_model(&id).await.unwrap();
        assert_eq!(
            engine
                .chat_completion(&id, &request("ok"))
                .await
                .unwrap()
                .model,
            id
        );
        engine.unload_model(&id).await.unwrap();
    }
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn quiesce_cleans_descendants_after_parent_exit() {
        use nix::{
            sys::signal::{kill, Signal},
            unistd::Pid,
        };
        let engine = fixture();
        let id = engine.config.model.clone();
        engine.load_model(&id).await.unwrap();
        let (parent, child) = {
            let guard = engine.server.read().await;
            let server = guard.as_ref().unwrap();
            let health: serde_json::Value = engine
                .client
                .get(format!("http://127.0.0.1:{}/health", server.port))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            let parent = server.process.lock().await.child.id().unwrap();
            (parent, health["child"].as_u64().unwrap())
        };
        kill(Pid::from_raw(parent as i32), Signal::SIGKILL).unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while !engine
                .health_check()
                .await
                .unwrap()
                .models_loaded
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let descendant_alive = || {
            std::fs::read_to_string(format!("/proc/{child}/status")).is_ok_and(|status| {
                !status
                    .lines()
                    .any(|line| line.starts_with("State:") && line.contains('Z'))
            })
        };
        assert!(descendant_alive(), "fixture child must survive parent exit");
        engine.quiesce().await.unwrap();
        assert!(engine.server.read().await.is_none());
        tokio::time::timeout(Duration::from_secs(3), async {
            while descendant_alive() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("quiesce must terminate the orphan engine-core child");
        engine.quiesce().await.unwrap();
        engine.load_model(&id).await.unwrap();
        engine.unload_model(&id).await.unwrap();
    }
    #[tokio::test]
    async fn unload_waits_after_caller_cancels() {
        let engine = Arc::new(fixture());
        let id = engine.config.model.clone();
        engine.load_model(&id).await.unwrap();
        let port = engine.server.read().await.as_ref().unwrap().port;
        let caller_engine = engine.clone();
        let caller_id = id.clone();
        let caller = tokio::spawn(async move {
            caller_engine
                .chat_completion(&caller_id, &request("slow"))
                .await
        });
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
                if health["active"] == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        caller.abort();
        assert!(
            tokio::time::timeout(Duration::from_millis(100), engine.unload_model(&id))
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
    }
}
