//! Managed vLLM engine backed by an authenticated OpenAI-compatible server.

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{io::Cursor, net::SocketAddr};

use async_trait::async_trait;
use axum::body::Bytes;
use base64::Engine as _;
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};
use simple_ai_common::{
    ChatCompletionRequest, ChatCompletionResponse, InferenceMetrics, ReasoningCapabilities,
    ReasoningEffort,
};
use tokio::process::Command;
use tokio::sync::{Mutex, RwLock};

use crate::config::{VllmEngineConfig, VllmModelConfig};
use crate::error::{Error, Result};

use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};

pub struct VllmEngine {
    config: VllmEngineConfig,
    client: reqwest::Client,
    api_key: String,
    loaded: Arc<RwLock<Option<String>>>,
    lifecycle: Mutex<()>,
}

impl VllmEngine {
    pub fn new(config: VllmEngineConfig) -> Result<Self> {
        let api_key_contents = std::fs::read_to_string(&config.api_key_file)
            .map_err(|e| Error::Internal(format!("cannot read vLLM API key file: {e}")))?;
        let api_key = api_key_contents
            .trim()
            .strip_prefix("VLLM_API_KEY=")
            .unwrap_or_else(|| api_key_contents.trim())
            .to_string();
        if api_key.is_empty() {
            return Err(Error::Internal("vLLM API key is empty".to_string()));
        }
        Ok(Self {
            config,
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .build()
                .map_err(|e| Error::Internal(format!("cannot create vLLM client: {e}")))?,
            api_key,
            loaded: Arc::new(RwLock::new(None)),
            lifecycle: Mutex::new(()),
        })
    }

    fn model(&self, model_id: &str) -> Result<&VllmModelConfig> {
        self.config
            .models
            .get(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))
    }

    async fn compose(&self, args: &[&str]) -> Result<()> {
        let mut command = Command::new("docker");
        command
            .arg("compose")
            .args(args)
            .current_dir(&self.config.compose_dir)
            .kill_on_drop(true);
        let output = command
            .output()
            .await
            .map_err(|e| Error::LoadFailed(format!("failed to run docker compose: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::LoadFailed(format!(
                "docker compose failed: {}",
                stderr.trim()
            )));
        }
        Ok(())
    }

    async fn upstream_healthy(&self, served_model: Option<&str>) -> bool {
        let health = self
            .client
            .get(format!(
                "{}/health",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .send()
            .await;
        if !matches!(health, Ok(response) if response.status().is_success()) {
            return false;
        }
        let Some(served_model) = served_model else {
            return true;
        };
        let response = self
            .client
            .get(format!(
                "{}/v1/models",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .send()
            .await;
        let Ok(response) = response.and_then(reqwest::Response::error_for_status) else {
            return false;
        };
        let Ok(body) = response.json::<Value>().await else {
            return false;
        };
        body.get("data")
            .and_then(Value::as_array)
            .is_some_and(|models| {
                models
                    .iter()
                    .any(|model| model.get("id").and_then(Value::as_str) == Some(served_model))
            })
    }

    async fn wait_ready(&self, served_model: &str) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(self.config.startup_timeout_secs);
        while Instant::now() < deadline {
            if self.upstream_healthy(Some(served_model)).await {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        Err(Error::LoadFailed(format!(
            "vLLM did not become ready within {} seconds",
            self.config.startup_timeout_secs
        )))
    }

    async fn stop_model(&self, model: &VllmModelConfig) -> Result<()> {
        tokio::time::timeout(
            Duration::from_secs(self.config.shutdown_timeout_secs),
            self.compose(&[
                "--env-file",
                &model.env_file,
                "--profile",
                &model.compose_profile,
                "stop",
                &model.compose_service,
            ]),
        )
        .await
        .map_err(|_| {
            Error::LoadFailed(format!(
                "vLLM did not stop within {} seconds",
                self.config.shutdown_timeout_secs
            ))
        })?
    }

    async fn normalized_request(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<Value> {
        let model = self.model(model_id)?;
        if request
            .thinking_budget_tokens
            .is_some_and(|budget| budget > 0)
        {
            return Err(Error::InvalidRequest(
                "positive thinking_budget_tokens is not supported by this vLLM profile; use reasoning_effort"
                    .to_string(),
            ));
        }
        let mut body = serde_json::to_value(request)
            .map_err(|e| Error::InvalidRequest(format!("cannot serialize request: {e}")))?;
        let object = body
            .as_object_mut()
            .ok_or_else(|| Error::InvalidRequest("chat request must be an object".to_string()))?;
        object.insert(
            "model".to_string(),
            Value::String(model.served_model.clone()),
        );
        object.remove("thinking_budget_tokens");

        let requested = request.reasoning_effort.unwrap_or(ReasoningEffort::Medium);
        let (enable_thinking, effort) = match requested {
            ReasoningEffort::None => (false, "medium"),
            ReasoningEffort::Minimal | ReasoningEffort::Low => (true, "low"),
            ReasoningEffort::Medium | ReasoningEffort::Default => (true, "medium"),
            ReasoningEffort::High | ReasoningEffort::Xhigh | ReasoningEffort::Max => {
                (true, "xhigh")
            }
        };
        if request.thinking_budget_tokens == Some(0) {
            object.insert(
                "chat_template_kwargs".to_string(),
                json!({"enable_thinking": false, "reasoning_effort": "medium"}),
            );
        } else {
            object.insert(
                "chat_template_kwargs".to_string(),
                json!({"enable_thinking": enable_thinking, "reasoning_effort": effort}),
            );
        }
        normalize_tool_arguments(&mut body)?;
        normalize_images(&mut body).await?;
        Ok(body)
    }

    async fn ensure_loaded(&self, model_id: &str) -> Result<()> {
        if self.loaded.read().await.as_deref() == Some(model_id) {
            return Ok(());
        }
        self.load_model(model_id).await
    }
}

fn normalize_tool_arguments(body: &mut Value) -> Result<()> {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for message in messages {
        let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for call in calls {
            let Some(arguments) = call.pointer_mut("/function/arguments") else {
                continue;
            };
            if let Some(encoded) = arguments.as_str() {
                *arguments = serde_json::from_str(encoded).map_err(|e| {
                    Error::InvalidRequest(format!("tool call arguments must contain JSON: {e}"))
                })?;
            }
        }
    }
    Ok(())
}

const MAX_IMAGES: usize = 4;
const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const MAX_TOTAL_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_IMAGE_PIXELS: u64 = 2_097_152;

async fn normalize_images(body: &mut Value) -> Result<()> {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    let mut images = 0usize;
    let mut total = 0usize;
    for message in messages {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            images += 1;
            if images > MAX_IMAGES {
                return Err(Error::InvalidRequest(format!(
                    "at most {MAX_IMAGES} images are allowed"
                )));
            }
            let url = part
                .pointer("/image_url/url")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::InvalidRequest("image_url.url is required".to_string()))?;
            let (mime, bytes) = if url.starts_with("data:") {
                decode_data_image(url)?
            } else {
                fetch_https_image(url).await?
            };
            validate_image(&bytes)?;
            total += bytes.len();
            if total > MAX_TOTAL_IMAGE_BYTES {
                return Err(Error::InvalidRequest(format!(
                    "decoded image content exceeds {} MiB total",
                    MAX_TOTAL_IMAGE_BYTES / 1024 / 1024
                )));
            }
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
            *part
                .pointer_mut("/image_url/url")
                .expect("image URL pointer was already validated") =
                Value::String(format!("data:{mime};base64,{encoded}"));
        }
    }
    Ok(())
}

fn decode_data_image(url: &str) -> Result<(String, Vec<u8>)> {
    let (header, encoded) = url
        .split_once(',')
        .ok_or_else(|| Error::InvalidRequest("malformed image data URL".to_string()))?;
    if !header.ends_with(";base64") {
        return Err(Error::InvalidRequest(
            "image data URLs must use base64 encoding".to_string(),
        ));
    }
    let mime = header.trim_start_matches("data:");
    let mime = mime.trim_end_matches(";base64");
    if !matches!(
        mime,
        "image/png" | "image/jpeg" | "image/webp" | "image/gif"
    ) {
        return Err(Error::InvalidRequest(format!(
            "unsupported image MIME type: {mime}"
        )));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| Error::InvalidRequest(format!("invalid image base64: {e}")))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(Error::InvalidRequest(format!(
            "image exceeds {} MiB",
            MAX_IMAGE_BYTES / 1024 / 1024
        )));
    }
    Ok((mime.to_string(), bytes))
}

async fn fetch_https_image(input: &str) -> Result<(String, Vec<u8>)> {
    let mut url = reqwest::Url::parse(input)
        .map_err(|e| Error::InvalidRequest(format!("invalid image URL: {e}")))?;
    for _ in 0..=3 {
        if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
            return Err(Error::InvalidRequest(
                "remote images must use credential-free HTTPS URLs".to_string(),
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| Error::InvalidRequest("image URL has no host".to_string()))?
            .to_string();
        let port = url.port_or_known_default().unwrap_or(443);
        let addresses: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), port))
            .await
            .map_err(|e| Error::InvalidRequest(format!("cannot resolve image host: {e}")))?
            .collect();
        if addresses.is_empty() || addresses.iter().any(|address| !is_public_ip(address.ip())) {
            return Err(Error::InvalidRequest(
                "all resolved image host addresses must be public".to_string(),
            ));
        }
        let address = addresses[0];
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .resolve(&host, address)
            .build()
            .map_err(|e| Error::Internal(format!("cannot build image client: {e}")))?;
        let response = client
            .get(url.clone())
            .header(reqwest::header::USER_AGENT, "simple-ai-image-fetcher/1")
            .send()
            .await
            .map_err(|e| Error::InvalidRequest(format!("cannot fetch image: {e}")))?;
        if response.status().is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| {
                    Error::InvalidRequest("image redirect has no Location".to_string())
                })?;
            url = url
                .join(location)
                .map_err(|e| Error::InvalidRequest(format!("invalid image redirect: {e}")))?;
            continue;
        }
        let response = response
            .error_for_status()
            .map_err(|e| Error::InvalidRequest(format!("image fetch failed: {e}")))?;
        if response
            .content_length()
            .is_some_and(|size| size as usize > MAX_IMAGE_BYTES)
        {
            return Err(Error::InvalidRequest(
                "remote image is too large".to_string(),
            ));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .unwrap_or("application/octet-stream")
            .to_string();
        if !matches!(
            content_type.as_str(),
            "image/png" | "image/jpeg" | "image/webp" | "image/gif"
        ) {
            return Err(Error::InvalidRequest(format!(
                "remote URL returned unsupported content type: {content_type}"
            )));
        }
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| Error::InvalidRequest(format!("image read failed: {e}")))?;
            if bytes.len() + chunk.len() > MAX_IMAGE_BYTES {
                return Err(Error::InvalidRequest(
                    "remote image is too large".to_string(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        return Ok((content_type, bytes));
    }
    Err(Error::InvalidRequest(
        "remote image exceeded three redirects".to_string(),
    ))
}

fn validate_image(bytes: &[u8]) -> Result<()> {
    let format = image::guess_format(bytes)
        .map_err(|e| Error::InvalidRequest(format!("unrecognized image data: {e}")))?;
    if !matches!(
        format,
        image::ImageFormat::Png
            | image::ImageFormat::Jpeg
            | image::ImageFormat::WebP
            | image::ImageFormat::Gif
    ) {
        return Err(Error::InvalidRequest(
            "unsupported image format".to_string(),
        ));
    }
    let (width, height) = image::ImageReader::with_format(Cursor::new(bytes), format)
        .into_dimensions()
        .map_err(|e| Error::InvalidRequest(format!("invalid image data: {e}")))?;
    if u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS {
        return Err(Error::InvalidRequest(format!(
            "image exceeds {MAX_IMAGE_PIXELS} pixels"
        )));
    }
    // Fully decode only after the allocation bound has been checked.
    image::load_from_memory_with_format(bytes, format)
        .map_err(|e| Error::InvalidRequest(format!("invalid image data: {e}")))?;
    Ok(())
}

fn is_public_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(ip) => {
            !(ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.octets()[0] == 0
                || (ip.octets()[0] == 100 && (64..=127).contains(&ip.octets()[1])))
        }
        std::net::IpAddr::V6(ip) => {
            let segments = ip.segments();
            !(ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] == 0x2001 && segments[1] == 0x0db8))
        }
    }
}

#[async_trait]
impl InferenceEngine for VllmEngine {
    fn engine_type(&self) -> &'static str {
        "vllm"
    }

    fn batch_size(&self) -> u32 {
        self.config.batch_size
    }

    async fn health_check(&self) -> Result<EngineHealth> {
        let loaded = self.loaded.read().await.clone();
        let served = loaded
            .as_deref()
            .and_then(|id| self.config.models.get(id))
            .map(|model| model.served_model.as_str());
        let healthy = self.upstream_healthy(served).await;
        Ok(EngineHealth {
            is_healthy: healthy || loaded.is_none(),
            version: Some("vllm-0.27.1-managed".to_string()),
            models_loaded: loaded.into_iter().collect(),
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(self
            .config
            .models
            .iter()
            .map(|(id, model)| ModelInfo {
                id: id.clone(),
                name: format!("{} ({})", id, model.profile),
                size_bytes: None,
                parameter_count: Some(27_000_000_000),
                context_length: Some(model.context_length),
                quantization: model
                    .quantization
                    .clone()
                    .or_else(|| Some("W4A16".to_string())),
                modified_at: None,
                reasoning: model
                    .reasoning
                    .as_ref()
                    .map(|controls| ReasoningCapabilities {
                        supported_efforts: controls.supported_efforts.clone(),
                        supports_thinking_budget: false,
                        default_effort: controls.default_effort,
                        default_thinking_budget_tokens: None,
                    }),
            })
            .collect())
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        Ok(self
            .list_models()
            .await?
            .into_iter()
            .find(|model| model.id == model_id))
    }

    async fn load_model(&self, model_id: &str) -> Result<()> {
        let _guard = self.lifecycle.lock().await;
        if self.loaded.read().await.as_deref() == Some(model_id) {
            return Ok(());
        }
        let model = self.model(model_id)?.clone();
        if !Path::new(&self.config.compose_dir).is_dir() {
            return Err(Error::LoadFailed(format!(
                "compose directory does not exist: {}",
                self.config.compose_dir
            )));
        }
        if let Some(previous) = self.loaded.write().await.take() {
            let previous = self.model(&previous)?.clone();
            self.stop_model(&previous).await?;
        }
        self.compose(&[
            "--env-file",
            &model.env_file,
            "--profile",
            &model.compose_profile,
            "up",
            "-d",
            "--force-recreate",
            &model.compose_service,
        ])
        .await?;
        self.wait_ready(&model.served_model).await?;
        *self.loaded.write().await = Some(model_id.to_string());
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        let _guard = self.lifecycle.lock().await;
        if self.loaded.read().await.as_deref() != Some(model_id) {
            return Ok(());
        }
        let model = self.model(model_id)?.clone();
        self.stop_model(&model).await?;
        *self.loaded.write().await = None;
        Ok(())
    }

    async fn chat_completion(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        self.ensure_loaded(model_id).await?;
        let started = Instant::now();
        let response = self
            .client
            .post(format!(
                "{}/v1/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&self.normalized_request(model_id, request).await?)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        if !status.is_success() {
            return Err(Error::InferenceFailed(format!(
                "vLLM returned {status}: {}",
                String::from_utf8_lossy(&bytes)
            )));
        }
        let mut result: ChatCompletionResponse = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Communication(format!("invalid vLLM response: {e}")))?;
        result.model = model_id.to_string();
        result.inference_metrics = Some(
            InferenceMetrics {
                resolved_model: Some(model_id.to_string()),
                engine_type: Some("vllm".to_string()),
                context_window: Some(self.model(model_id)?.context_length),
                prompt_tokens: result.usage.as_ref().map(|usage| usage.prompt_tokens),
                completion_tokens: result.usage.as_ref().map(|usage| usage.completion_tokens),
                total_inference_ms: Some(started.elapsed().as_millis() as u64),
                ..Default::default()
            }
            .with_computed_rates(),
        );
        Ok(result)
    }

    async fn chat_completion_stream(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        self.ensure_loaded(model_id).await?;
        let response = self
            .client
            .post(format!(
                "{}/v1/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&self.normalized_request(model_id, request).await?)
            .send()
            .await
            .map_err(|e| Error::Communication(e.to_string()))?;
        if !response.status().is_success() {
            return Err(Error::InferenceFailed(format!(
                "vLLM streaming request returned {}",
                response.status()
            )));
        }
        let stream = response.bytes_stream().map(|chunk| {
            chunk.map_err(|e| Error::Communication(format!("vLLM stream failed: {e}")))
        });
        Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_non_public_image_addresses() {
        for address in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.1.1",
            "100.64.0.1",
            "::1",
            "fe80::1",
            "fc00::1",
        ] {
            assert!(!is_public_ip(address.parse().unwrap()), "{address}");
        }
        assert!(is_public_ip("1.1.1.1".parse().unwrap()));
        assert!(is_public_ip("2606:4700:4700::1111".parse().unwrap()));
    }

    #[test]
    fn data_images_require_supported_base64_mime() {
        assert!(decode_data_image("data:text/plain;base64,SGVsbG8=").is_err());
        assert!(decode_data_image("data:image/png,abc").is_err());
        let (_, png) = decode_data_image("data:image/png;base64,iVBORw0KGgo=").unwrap();
        assert_eq!(&png[..4], b"\x89PNG");
    }

    #[test]
    fn historical_tool_arguments_become_objects() {
        let mut body = json!({"messages":[{"role":"assistant","tool_calls":[{
            "id":"call_1","type":"function","function":{"name":"x","arguments":"{\"a\":1}"}
        }]}]});
        normalize_tool_arguments(&mut body).unwrap();
        assert_eq!(
            body.pointer("/messages/0/tool_calls/0/function/arguments/a"),
            Some(&json!(1))
        );
    }
}
