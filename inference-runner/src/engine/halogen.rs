//! One checkpoint served by an on-demand Halogen systemd user service.
//! GPU ownership and in-flight stream leases belong to EngineRegistry.
use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
use crate::{
    config::HalogenEngineConfig,
    error::{Error, Result},
};
use async_trait::async_trait;
use futures_util::stream;
use serde_json::{json, Value};
use simple_ai_common::{
    format_sse_metrics, ChatCompletionRequest, ChatCompletionResponse, InferenceMetrics,
    ReasoningCapabilities, ReasoningEffort,
};
use simple_server::web::body::Bytes;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tokio::{process::Command, sync::Mutex};

pub struct HalogenEngine {
    config: HalogenEngineConfig,
    client: reqwest::Client,
    lifecycle: Mutex<()>,
    expected_running: AtomicBool,
}

impl HalogenEngine {
    pub fn new(config: HalogenEngineConfig) -> Result<Self> {
        let url = reqwest::Url::parse(&config.base_url)
            .map_err(|e| Error::Internal(format!("invalid Halogen URL: {e}")))?;
        if url.scheme() != "http"
            || !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"))
        {
            return Err(Error::Internal(
                "Halogen's unauthenticated API must bind to loopback".into(),
            ));
        }
        if config.service.starts_with('-')
            || !config.service.ends_with(".service")
            || config.service.contains('/')
            || config.model_id.is_empty()
            || config.served_model.is_empty()
            || config.batch_size == 0
            || config.context_length == 0
        {
            return Err(Error::Internal(
                "invalid Halogen service/model configuration".into(),
            ));
        }
        Ok(Self {
            config,
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .build()
                .map_err(|e| Error::Internal(e.to_string()))?,
            lifecycle: Mutex::new(()),
            expected_running: AtomicBool::new(false),
        })
    }

    fn check_model(&self, id: &str) -> Result<()> {
        if id != self.config.model_id {
            return Err(Error::ModelNotFound(id.into()));
        }
        Ok(())
    }

    async fn ready(&self) -> bool {
        let check = async {
            let health = self
                .client
                .get(format!(
                    "{}/health",
                    self.config.base_url.trim_end_matches('/')
                ))
                .send()
                .await
                .ok()?
                .error_for_status()
                .ok()?;
            drop(health);
            let body: Value = self
                .client
                .get(format!(
                    "{}/v1/models",
                    self.config.base_url.trim_end_matches('/')
                ))
                .send()
                .await
                .ok()?
                .error_for_status()
                .ok()?
                .json()
                .await
                .ok()?;
            Some(
                body["data"]
                    .as_array()?
                    .iter()
                    .any(|m| m["id"].as_str() == Some(&self.config.served_model)),
            )
        };
        matches!(
            tokio::time::timeout(Duration::from_secs(3), check).await,
            Ok(Some(true))
        )
    }

    async fn control(&self, action: &str, timeout: u64) -> Result<()> {
        let output = tokio::time::timeout(
            Duration::from_secs(timeout),
            Command::new("systemctl")
                .args(["--user", action, "--", &self.config.service])
                .kill_on_drop(true)
                .output(),
        )
        .await
        .map_err(|_| Error::LoadFailed(format!("Halogen {action} timed out")))?
        .map_err(|e| Error::LoadFailed(format!("Halogen {action}: {e}")))?;
        if !output.status.success() {
            return Err(Error::LoadFailed(format!(
                "Halogen {action}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(())
    }

    fn request_body(
        &self,
        id: &str,
        request: &ChatCompletionRequest,
        streaming: bool,
    ) -> Result<Value> {
        self.check_model(id)?;
        if request.has_images() {
            return Err(Error::InvalidRequest(
                "this Halogen checkpoint is configured for text, not images".into(),
            ));
        }
        if request.thinking_budget_tokens.is_some_and(|n| n < -1) {
            return Err(Error::InvalidRequest(
                "thinking_budget_tokens must be -1 or greater".into(),
            ));
        }
        let mut body =
            serde_json::to_value(request).map_err(|e| Error::InvalidRequest(e.to_string()))?;
        let obj = body.as_object_mut().expect("chat request is an object");
        // Optional null fields are not equivalent to omission for upstream defaults.
        obj.retain(|_, v| !v.is_null());
        obj.insert("model".into(), json!(self.config.served_model));
        obj.insert("stream".into(), json!(streaming));
        if streaming {
            obj.insert("stream_options".into(), json!({"include_usage":true}));
        }
        let effort = match request.reasoning_effort.unwrap_or(ReasoningEffort::Xhigh) {
            ReasoningEffort::None => "none",
            ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            _ => "xhigh",
        };
        obj.insert(
            "reasoning_effort".into(),
            json!(if request.thinking_budget_tokens == Some(0) {
                "none"
            } else {
                effort
            }),
        );
        // Halogen uses omitted/null for unlimited; simple-ai uses -1.
        obj.remove("thinking_budget_tokens");
        if let Some(budget) = request.thinking_budget_tokens.filter(|n| *n >= 0) {
            obj.insert("max_thinking_tokens".into(), json!(budget));
        }
        Ok(body)
    }

    async fn send(
        &self,
        id: &str,
        request: &ChatCompletionRequest,
        streaming: bool,
    ) -> Result<reqwest::Response> {
        let body = self.request_body(id, request, streaming)?;
        self.load_model(id).await?;
        let response = self
            .client
            .post(format!(
                "{}/v1/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .map_err(|e| Error::Communication(e.to_string()))?;
            return Err(Error::UpstreamResponse { status, body });
        }
        Ok(response)
    }
}

#[async_trait]
impl InferenceEngine for HalogenEngine {
    fn engine_type(&self) -> &'static str {
        "halogen"
    }
    fn batch_size(&self) -> u32 {
        self.config.batch_size
    }
    async fn health_check(&self) -> Result<EngineHealth> {
        let ready = self.ready().await;
        if ready {
            self.expected_running.store(true, Ordering::SeqCst);
        }
        Ok(EngineHealth {
            is_healthy: ready || !self.expected_running.load(Ordering::SeqCst),
            version: Some("halogen-managed".into()),
            models_loaded: if ready {
                vec![self.config.model_id.clone()]
            } else {
                vec![]
            },
        })
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![ModelInfo {
            id: self.config.model_id.clone(),
            name: "Qwen3.8 Flash Next Uncensored (Halogen)".into(),
            size_bytes: None,
            parameter_count: None,
            context_length: Some(self.config.context_length),
            quantization: Some(self.config.quantization.clone()),
            modified_at: None,
            reasoning: Some(ReasoningCapabilities {
                supported_efforts: vec![
                    ReasoningEffort::None,
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::Xhigh,
                ],
                supports_thinking_budget: true,
                default_effort: Some(ReasoningEffort::Xhigh),
                default_thinking_budget_tokens: None,
            }),
        }])
    }
    async fn get_model(&self, id: &str) -> Result<Option<ModelInfo>> {
        Ok(self.list_models().await?.into_iter().find(|m| m.id == id))
    }
    async fn load_model(&self, id: &str) -> Result<()> {
        self.check_model(id)?;
        let _guard = self.lifecycle.lock().await;
        if self.ready().await {
            self.expected_running.store(true, Ordering::SeqCst);
            return Ok(());
        }
        self.expected_running.store(true, Ordering::SeqCst);
        // Starting an already active unit preserves its in-flight requests.
        self.control("start", self.config.startup_timeout_secs)
            .await?;
        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(self.config.startup_timeout_secs);
        while tokio::time::Instant::now() < deadline {
            if self.ready().await {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        self.control("stop", self.config.shutdown_timeout_secs)
            .await?;
        self.expected_running.store(false, Ordering::SeqCst);
        Err(Error::LoadFailed(
            "Halogen did not become ready before the startup deadline".into(),
        ))
    }
    async fn unload_model(&self, id: &str) -> Result<()> {
        self.check_model(id)?;
        self.quiesce().await
    }
    async fn quiesce(&self) -> Result<()> {
        let _guard = self.lifecycle.lock().await;
        // Stop the unit even before health becomes ready, including after runner restart.
        self.control("stop", self.config.shutdown_timeout_secs)
            .await?;
        self.expected_running.store(false, Ordering::SeqCst);
        Ok(())
    }
    async fn chat_completion(
        &self,
        id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let value: Value = self
            .send(id, request, false)
            .await?
            .json()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        let mut metrics = Metrics::new(id, self.config.context_length);
        metrics.observe(&value);
        let mut response: ChatCompletionResponse =
            serde_json::from_value(value).map_err(|e| Error::Communication(e.to_string()))?;
        response.model = id.into();
        response.inference_metrics = Some(metrics.0);
        Ok(response)
    }
    async fn chat_completion_stream(
        &self,
        id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        let response = self.send(id, request, true).await?;
        let state = StreamState {
            response,
            decoder: Decoder::new(id, self.config.context_length),
        };
        Ok(Box::pin(stream::try_unfold(
            state,
            |mut state| async move {
                loop {
                    if let Some(event) = state.decoder.pending.pop_front() {
                        return Ok(Some((event, state)));
                    }
                    if state.decoder.done {
                        return Ok(None);
                    }
                    let chunk = state
                        .response
                        .chunk()
                        .await
                        .map_err(|e| Error::Communication(e.to_string()))?
                        .ok_or_else(|| {
                            Error::Communication("Halogen stream ended before [DONE]".into())
                        })?;
                    state.decoder.push(&chunk)?;
                }
            },
        )))
    }
}

struct StreamState {
    response: reqwest::Response,
    decoder: Decoder,
}
struct Metrics(InferenceMetrics);
impl Metrics {
    fn new(id: &str, ctx: u32) -> Self {
        Self(InferenceMetrics {
            resolved_model: Some(id.into()),
            engine_type: Some("halogen".into()),
            context_window: Some(ctx),
            ..Default::default()
        })
    }
    fn observe(&mut self, v: &Value) {
        let n = |v: &Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
        if let Some(usage) = v.get("usage").filter(|u| u.is_object()) {
            self.0.prompt_tokens = n(&usage["prompt_tokens"]).or(self.0.prompt_tokens);
            self.0.completion_tokens = n(&usage["completion_tokens"]).or(self.0.completion_tokens);
            self.0.cached_prompt_tokens =
                n(&usage["prompt_tokens_details"]["cached_tokens"]).or(self.0.cached_prompt_tokens);
        }
        if let Some(t) = v.get("timings").filter(|t| t.is_object()) {
            self.0.prompt_eval_ms = t["prompt_ms"].as_f64().map(|n| n.max(0.0).round() as u64);
            self.0.completion_eval_ms = t["predicted_ms"]
                .as_f64()
                .map(|n| n.max(0.0).round() as u64);
            self.0.total_inference_ms = self
                .0
                .prompt_eval_ms
                .zip(self.0.completion_eval_ms)
                .map(|(a, b)| a + b);
            // Use the engine's rate over NEW tokens, never cached+new / prefill time.
            self.0.prompt_tokens_per_sec = t["prompt_per_second"].as_f64();
            self.0.completion_tokens_per_sec = t["predicted_per_second"].as_f64();
            self.0.cached_prompt_tokens = n(&t["cache_n"]).or(self.0.cached_prompt_tokens);
        }
    }
}
struct Decoder {
    buffer: Vec<u8>,
    pending: VecDeque<Bytes>,
    done: bool,
    metrics: Metrics,
}
impl Decoder {
    fn new(id: &str, ctx: u32) -> Self {
        Self {
            buffer: vec![],
            pending: VecDeque::new(),
            done: false,
            metrics: Metrics::new(id, ctx),
        }
    }
    fn push(&mut self, bytes: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(bytes);
        loop {
            let lf = self
                .buffer
                .windows(2)
                .position(|w| w == b"\n\n")
                .map(|n| (n, 2));
            let crlf = self
                .buffer
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|n| (n, 4));
            let Some((end, delimiter)) = lf.into_iter().chain(crlf).min_by_key(|(n, _)| *n) else {
                break;
            };
            let raw: Vec<u8> = self.buffer.drain(..end + delimiter).collect();
            let event = std::str::from_utf8(&raw[..end])
                .map_err(|e| Error::Communication(e.to_string()))?;
            let data = event
                .lines()
                .filter_map(|l| l.strip_prefix("data:").map(str::trim_start))
                .collect::<Vec<_>>()
                .join("\n");
            if data.is_empty() {
                self.pending.push_back(Bytes::from(raw));
                continue;
            }
            if data == "[DONE]" {
                self.pending.push_back(Bytes::from(
                    format_sse_metrics(&self.metrics.0)
                        .map_err(|e| Error::Internal(e.to_string()))?,
                ));
                self.pending
                    .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                self.done = true;
                self.buffer.clear();
                break;
            }
            let mut value: Value = serde_json::from_str(&data)
                .map_err(|e| Error::Communication(format!("invalid Halogen SSE: {e}")))?;
            if value.get("error").is_some() {
                return Err(Error::InferenceFailed(value["error"].to_string()));
            }
            self.metrics.observe(&value);
            if value.get("model").is_some() {
                value["model"] = json!(self.metrics.0.resolved_model);
            }
            self.pending
                .push_back(Bytes::from(format!("data: {value}\n\n")));
        }
        if self.buffer.len() > 4 * 1024 * 1024 {
            return Err(Error::Communication(
                "Halogen SSE event exceeds 4 MiB".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use wiremock::{
        matchers::{body_partial_json, method, path},
        Mock, MockServer, ResponseTemplate,
    };
    fn config(url: String) -> HalogenEngineConfig {
        HalogenEngineConfig {
            enabled: true,
            base_url: url,
            service: "simple-ai-halogen.service".into(),
            model_id: "uncensored-iq4xs".into(),
            served_model: "halogen-qwen3.8-flash-next".into(),
            context_length: 262144,
            quantization: "IQ4_XS".into(),
            batch_size: 4,
            startup_timeout_secs: 5,
            shutdown_timeout_secs: 5,
        }
    }
    fn request(extra: Value) -> ChatCompletionRequest {
        let mut value = json!({"messages":[{"role":"user","content":"hello"}]});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(value).unwrap()
    }
    async fn ready_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"data":[{"id":"halogen-qwen3.8-flash-next"}]})),
            )
            .mount(&server)
            .await;
        server
    }
    #[test]
    fn reasoning_and_optional_fields_keep_halogen_defaults() {
        let engine = HalogenEngine::new(config("http://127.0.0.1:8731".into())).unwrap();
        let body = engine
            .request_body(
                "uncensored-iq4xs",
                &request(json!({"thinking_budget_tokens":-1})),
                true,
            )
            .unwrap();
        assert_eq!(body["reasoning_effort"], "xhigh");
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("thinking_budget_tokens").is_none());
        assert!(body.get("max_thinking_tokens").is_none());
        let body = engine
            .request_body(
                "uncensored-iq4xs",
                &request(json!({"reasoning_effort":"high","thinking_budget_tokens":64})),
                false,
            )
            .unwrap();
        assert_eq!(body["reasoning_effort"], "xhigh");
        assert_eq!(body["max_thinking_tokens"], 64);
        let body = engine
            .request_body(
                "uncensored-iq4xs",
                &request(json!({"thinking_budget_tokens":0})),
                false,
            )
            .unwrap();
        assert_eq!(body["reasoning_effort"], "none");
        assert!(engine
            .request_body(
                "uncensored-iq4xs",
                &request(json!({"thinking_budget_tokens":-2})),
                false
            )
            .is_err());
    }
    #[test]
    fn fragmented_crlf_stream_preserves_tools_and_cached_prefill_rates() {
        let raw=concat!("data: {\"model\":\"upstream\",\"choices\":[{\"delta\":{\"reasoning_content\":\"think\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}}]}\r\n\r\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":24597,\"completion_tokens\":128},\"timings\":{\"prompt_ms\":507.0,\"predicted_ms\":2738.6,\"prompt_per_second\":305.72,\"predicted_per_second\":46.739,\"cache_n\":24442}}\n\n",
            "data: [DONE]\n\n");
        let mut decoder = Decoder::new("public", 262144);
        for chunk in raw.as_bytes().chunks(7) {
            decoder.push(chunk).unwrap();
        }
        assert!(decoder.done);
        assert_eq!(decoder.metrics.0.prompt_tokens, Some(24597));
        assert_eq!(decoder.metrics.0.cached_prompt_tokens, Some(24442));
        assert_eq!(decoder.metrics.0.prompt_tokens_per_sec, Some(305.72));
        let output = decoder
            .pending
            .iter()
            .flat_map(|b| b.iter().copied())
            .collect::<Vec<_>>();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"model\":\"public\""));
        assert!(output.contains("reasoning_content"));
        assert!(output.contains("tool_calls"));
        assert!(output.ends_with("data: [DONE]\n\n"));
        assert!(output.find("simple_ai_metrics").unwrap() < output.find("[DONE]").unwrap());
    }
    #[tokio::test]
    async fn adopts_ready_service_forwards_tools_and_upstream_errors() {
        let server = ready_server().await;
        let engine = HalogenEngine::new(config(server.uri())).unwrap();
        assert_eq!(
            engine.health_check().await.unwrap().models_loaded,
            vec!["uncensored-iq4xs"]
        );
        let req = request(
            json!({"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"reasoning_effort":"none"}),
        );
        Mock::given(method("POST")).and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({"model":"halogen-qwen3.8-flash-next","reasoning_effort":"none","tools":req.tools})))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({"error":{"message":"context too long"}}))).expect(1).mount(&server).await;
        assert!(matches!(
            engine.chat_completion("uncensored-iq4xs", &req).await,
            Err(Error::UpstreamResponse { status: 400, .. })
        ));
        assert!(matches!(
            engine.load_model("unknown").await,
            Err(Error::ModelNotFound(_))
        ));
    }
    #[tokio::test]
    async fn interrupted_upstream_stream_is_not_reported_as_success() {
        let server = ready_server().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
                ),
            )
            .mount(&server)
            .await;
        let engine = HalogenEngine::new(config(server.uri())).unwrap();
        let mut stream = engine
            .chat_completion_stream("uncensored-iq4xs", &request(json!({})))
            .await
            .unwrap();
        assert!(stream.next().await.unwrap().is_ok());
        assert!(matches!(
            stream.next().await.unwrap(),
            Err(Error::Communication(_))
        ));
    }
}
