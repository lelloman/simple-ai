//! llama.cpp inference engine implementation.
//!
//! This engine manages llama-server subprocesses to provide inference capabilities.
//! Each loaded model runs in its own llama-server process for isolation.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Constants for polling intervals
const HEALTH_CHECK_INTERVAL_MS: u64 = 200;
const SERVER_STARTING_POLL_MS: u64 = 100;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use simple_ai_common::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::sync::{RwLock, Semaphore};

use super::{EngineHealth, InferenceEngine, ModelInfo};
use crate::config::LlamaCppEngineConfig;
use crate::error::{Error, Result};

/// State of a llama-server process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerState {
    /// Server process is not running.
    Stopped,
    /// Server process is starting up.
    Starting,
    /// Server is ready to accept requests.
    Ready,
    /// Server is running but failing health checks.
    Unhealthy,
    /// Server is being shut down.
    ShuttingDown,
}

/// A running llama-server instance for a specific model.
struct ServerInstance {
    model_id: String,
    port: u16,
    state: RwLock<ServerState>,
    process: RwLock<Option<Child>>,
    last_used: RwLock<Instant>,
}

impl ServerInstance {
    fn new(model_id: String, port: u16, process: Child) -> Self {
        Self {
            model_id,
            port,
            state: RwLock::new(ServerState::Starting),
            process: RwLock::new(Some(process)),
            last_used: RwLock::new(Instant::now()),
        }
    }

    async fn state(&self) -> ServerState {
        *self.state.read().await
    }

    async fn set_state(&self, state: ServerState) {
        *self.state.write().await = state;
    }

    async fn touch(&self) {
        *self.last_used.write().await = Instant::now();
    }

    async fn last_used(&self) -> Instant {
        *self.last_used.read().await
    }

    /// Check if the server process is still alive.
    async fn is_process_alive(&self) -> bool {
        let mut process = self.process.write().await;
        if let Some(ref mut child) = *process {
            match child.try_wait() {
                Ok(None) => true,  // Still running
                Ok(Some(_)) => false,  // Exited
                Err(_) => false,  // Error checking
            }
        } else {
            false
        }
    }

    /// Terminate the server process gracefully.
    async fn terminate(&self, timeout_secs: u64) {
        self.set_state(ServerState::ShuttingDown).await;

        let mut process_guard = self.process.write().await;
        if let Some(mut child) = process_guard.take() {
            // Try SIGTERM first on Unix
            #[cfg(unix)]
            {
                use nix::sys::signal::{kill, Signal};
                use nix::unistd::Pid;

                if let Some(pid) = child.id() {
                    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
                }
            }

            // Wait with timeout
            let wait_result = tokio::time::timeout(
                Duration::from_secs(timeout_secs),
                child.wait(),
            )
            .await;

            match wait_result {
                Ok(Ok(status)) => {
                    tracing::debug!(
                        "llama-server for {} exited with {}",
                        self.model_id,
                        status
                    );
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        "Error waiting for llama-server {}: {}",
                        self.model_id,
                        e
                    );
                }
                Err(_timeout) => {
                    tracing::warn!(
                        "llama-server {} didn't stop gracefully, killing",
                        self.model_id
                    );
                    let _ = child.kill().await;
                }
            }
        }

        self.set_state(ServerState::Stopped).await;
    }
}

/// llama.cpp inference engine.
///
/// Manages llama-server subprocesses to provide inference capabilities.
/// Each loaded model runs in its own llama-server process.
pub struct LlamaCppEngine {
    config: LlamaCppEngineConfig,
    http_client: Client,
    /// Map of model_id -> running server instance
    servers: RwLock<HashMap<String, Arc<ServerInstance>>>,
    /// Semaphore to limit concurrent server starts
    startup_semaphore: Semaphore,
}

impl LlamaCppEngine {
    pub fn new(config: LlamaCppEngineConfig) -> Self {
        Self {
            startup_semaphore: Semaphore::new(1), // Only one server startup at a time
            config,
            http_client: Client::new(),
            servers: RwLock::new(HashMap::new()),
        }
    }

    /// Get the path to a model file.
    ///
    /// Sanitizes model_id to prevent path traversal attacks by taking only
    /// the filename component (no directory separators allowed).
    fn model_path(&self, model_id: &str) -> PathBuf {
        // Sanitize: take only the last path component to prevent traversal
        let sanitized = model_id
            .trim()
            .split(['/', '\\'])
            .last()
            .unwrap_or(model_id)
            .trim_start_matches('.');  // Also prevent hidden files like ".." or "..foo"

        let mut path = PathBuf::from(&self.config.model_dir);
        if sanitized.ends_with(".gguf") {
            path.push(sanitized);
        } else {
            path.push(format!("{}.gguf", sanitized));
        }
        path
    }

    /// Allocate a port for a new server.
    async fn allocate_port(&self) -> Result<u16> {
        if let Some(base) = self.config.base_port {
            // Use sequential ports from base
            // Search range is max_servers or at least 100 to handle edge cases
            let search_range = self.config.max_servers.max(100) as u16;
            let servers = self.servers.read().await;
            let used: HashSet<u16> = servers.values().map(|s| s.port).collect();
            for offset in 0..search_range {
                let port = base.saturating_add(offset);
                if !used.contains(&port) {
                    return Ok(port);
                }
            }
            Err(Error::Internal(format!(
                "No available ports in range {}-{}",
                base,
                base.saturating_add(search_range)
            )))
        } else {
            // Use OS-assigned port
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .map_err(|e| Error::Internal(format!("Failed to bind for port allocation: {}", e)))?;
            let port = listener
                .local_addr()
                .map_err(|e| Error::Internal(format!("Failed to get local addr: {}", e)))?
                .port();
            drop(listener);
            Ok(port)
        }
    }

    /// Wait for a server to become ready.
    async fn wait_for_ready(&self, instance: &ServerInstance) -> Result<()> {
        let timeout = Duration::from_secs(self.config.startup_timeout_secs);
        let start = Instant::now();
        let health_url = format!("http://127.0.0.1:{}/health", instance.port);

        loop {
            if start.elapsed() > timeout {
                return Err(Error::LoadFailed(format!(
                    "llama-server startup timeout for {} after {:?}",
                    instance.model_id,
                    start.elapsed()
                )));
            }

            // Check process still alive
            if !instance.is_process_alive().await {
                return Err(Error::LoadFailed(format!(
                    "llama-server process died during startup for {}",
                    instance.model_id
                )));
            }

            // Check health endpoint
            if let Ok(resp) = self.http_client.get(&health_url).send().await {
                if resp.status().is_success() {
                    instance.set_state(ServerState::Ready).await;
                    tracing::info!(
                        "llama-server ready for {} on port {} ({:?})",
                        instance.model_id,
                        instance.port,
                        start.elapsed()
                    );
                    return Ok(());
                }
            }

            tokio::time::sleep(Duration::from_millis(HEALTH_CHECK_INTERVAL_MS)).await;
        }
    }

    /// Ensure we don't exceed max_servers by unloading the least recently used model.
    async fn enforce_server_limit(&self) -> Result<()> {
        let servers = self.servers.read().await;
        if servers.len() < self.config.max_servers {
            return Ok(());
        }

        // Find LRU server
        let mut oldest: Option<(String, Instant)> = None;
        for (model_id, instance) in servers.iter() {
            let last_used = instance.last_used().await;
            match &oldest {
                None => oldest = Some((model_id.clone(), last_used)),
                Some((_, oldest_time)) if last_used < *oldest_time => {
                    oldest = Some((model_id.clone(), last_used));
                }
                _ => {}
            }
        }
        drop(servers);

        if let Some((model_to_unload, _)) = oldest {
            tracing::info!(
                "Unloading LRU model {} to make room (max_servers={})",
                model_to_unload,
                self.config.max_servers
            );
            self.unload_model(&model_to_unload).await?;
        }

        Ok(())
    }

    /// Start a llama-server process for a model.
    async fn start_server(&self, model_id: &str) -> Result<Arc<ServerInstance>> {
        // Check if model file exists
        let model_path = self.model_path(model_id);
        if !model_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        // Allocate port
        let port = self.allocate_port().await?;

        // Build command
        let mut cmd = Command::new(&self.config.server_binary);

        // Prepend wrapper arguments (e.g., toolbox run -c llamacpp llama-server)
        // These must come before the llama-server specific flags
        for arg in &self.config.server_args {
            cmd.arg(arg);
        }
        cmd.arg("-m")
            .arg(&model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string());

        if let Some(gpu_layers) = self.config.gpu_layers {
            cmd.arg("-ngl").arg(gpu_layers.to_string());
        }

        if let Some(ctx_size) = self.config.context_size {
            cmd.arg("-c").arg(ctx_size.to_string());
        }

        // Configure process I/O
        cmd.stdin(Stdio::null()).kill_on_drop(true);

        if self.config.log_server_output {
            // Inherit stdout/stderr for debugging
            cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        } else {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }

        // Spawn process
        let process = cmd.spawn().map_err(|e| {
            Error::LoadFailed(format!(
                "Failed to spawn llama-server for {}: {}. Binary: {}",
                model_id,
                e,
                self.config.server_binary
            ))
        })?;

        tracing::info!(
            "Spawned llama-server for {} on port {} (pid: {:?})",
            model_id,
            port,
            process.id()
        );

        let instance = Arc::new(ServerInstance::new(model_id.to_string(), port, process));

        // Wait for server to be ready
        if let Err(e) = self.wait_for_ready(&instance).await {
            instance.terminate(self.config.shutdown_timeout_secs).await;
            return Err(e);
        }

        Ok(instance)
    }

    /// Get or start a server for a model.
    async fn ensure_server(&self, model_id: &str) -> Result<Arc<ServerInstance>> {
        // Check if already running and wait for starting servers
        loop {
            let servers = self.servers.read().await;
            if let Some(instance) = servers.get(model_id) {
                let state = instance.state().await;
                if state == ServerState::Ready {
                    instance.touch().await;
                    return Ok(instance.clone());
                } else if state == ServerState::Starting {
                    // Another task is starting this server, wait and retry
                    drop(servers);
                    tokio::time::sleep(Duration::from_millis(SERVER_STARTING_POLL_MS)).await;
                    continue;
                }
                // Server exists but is unhealthy/stopped, will be removed below
            }
            break;
        }

        // Need to start a new server
        // Acquire semaphore to serialize server starts
        let _permit = self
            .startup_semaphore
            .acquire()
            .await
            .map_err(|e| Error::Internal(format!("Semaphore error: {}", e)))?;

        // Double-check after acquiring semaphore
        {
            let servers = self.servers.read().await;
            if let Some(instance) = servers.get(model_id) {
                if instance.state().await == ServerState::Ready {
                    instance.touch().await;
                    return Ok(instance.clone());
                }
            }
        }

        // Enforce server limit
        self.enforce_server_limit().await?;

        // Remove any stale entry
        {
            let mut servers = self.servers.write().await;
            if let Some(old_instance) = servers.remove(model_id) {
                old_instance
                    .terminate(self.config.shutdown_timeout_secs)
                    .await;
            }
        }

        // Start new server
        let instance = self.start_server(model_id).await?;

        // Store in map
        {
            let mut servers = self.servers.write().await;
            servers.insert(model_id.to_string(), instance.clone());
        }

        Ok(instance)
    }

    /// Extract quantization type from filename (e.g., "model-q4_0.gguf" -> "Q4_0").
    fn extract_quantization(filename: &str) -> Option<String> {
        let patterns = [
            "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m",
            "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "f16", "f32",
        ];

        let lower = filename.to_lowercase();
        for pattern in patterns {
            if lower.contains(pattern) {
                return Some(pattern.to_uppercase());
            }
        }
        None
    }
}

// ============================================================================
// llama-server API types (OpenAI-compatible)
// ============================================================================

/// Request body for llama-server /v1/chat/completions endpoint.
#[derive(Debug, Serialize)]
struct LlamaChatRequest {
    model: String,
    messages: Vec<LlamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct LlamaMessage {
    role: String,
    content: String,
}

/// Response from llama-server /v1/chat/completions endpoint.
#[derive(Debug, Deserialize)]
struct LlamaChatResponse {
    choices: Vec<LlamaChoice>,
    #[serde(default)]
    usage: Option<LlamaUsage>,
}

#[derive(Debug, Deserialize)]
struct LlamaChoice {
    message: LlamaResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LlamaResponseMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct LlamaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// ============================================================================
// InferenceEngine implementation
// ============================================================================

#[async_trait]
impl InferenceEngine for LlamaCppEngine {
    fn engine_type(&self) -> &'static str {
        "llama_cpp"
    }

    async fn health_check(&self) -> Result<EngineHealth> {
        // Check if model directory exists
        let model_dir = PathBuf::from(&self.config.model_dir);
        if !model_dir.exists() {
            return Err(Error::EngineNotAvailable(format!(
                "Model directory not found: {}",
                self.config.model_dir
            )));
        }

        // Check if server binary exists (only for absolute paths)
        // For relative paths like "toolbox", we assume it's in PATH
        let binary_path = PathBuf::from(&self.config.server_binary);
        if binary_path.is_absolute() && !binary_path.exists() {
            return Err(Error::EngineNotAvailable(format!(
                "llama-server binary not found: {}",
                self.config.server_binary
            )));
        } else if !binary_path.is_absolute() {
            tracing::debug!(
                "Using relative path binary '{}', assuming it's in PATH",
                self.config.server_binary
            );
        }

        // Collect loaded models
        let servers = self.servers.read().await;
        let mut models_loaded = Vec::new();
        for (model_id, instance) in servers.iter() {
            if instance.state().await == ServerState::Ready {
                models_loaded.push(model_id.clone());
            }
        }

        Ok(EngineHealth {
            is_healthy: true,
            version: Some("llama.cpp".to_string()),
            models_loaded,
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let model_dir = PathBuf::from(&self.config.model_dir);

        if !model_dir.exists() {
            return Ok(vec![]);
        }

        let entries = std::fs::read_dir(&model_dir)
            .map_err(|e| Error::Internal(format!("Failed to read model dir: {}", e)))?;

        let mut models = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("gguf"))
            {
                let filename = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");

                let model_id = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let size_bytes = std::fs::metadata(&path).map(|m| m.len()).ok();

                let quantization = Self::extract_quantization(filename);

                models.push(ModelInfo {
                    id: model_id.clone(),
                    name: model_id,
                    size_bytes,
                    parameter_count: None,
                    context_length: self.config.context_size,
                    quantization,
                    modified_at: None,
                });
            }
        }

        Ok(models)
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        let models = self.list_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }

    async fn load_model(&self, model_id: &str) -> Result<()> {
        // This will start the server if not already running
        self.ensure_server(model_id).await?;
        tracing::info!("Model {} loaded via llama.cpp", model_id);
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        let instance = {
            let mut servers = self.servers.write().await;
            servers.remove(model_id)
        };

        if let Some(instance) = instance {
            instance
                .terminate(self.config.shutdown_timeout_secs)
                .await;
            tracing::info!("Model {} unloaded from llama.cpp", model_id);
        }

        Ok(())
    }

    async fn chat_completion(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        // Ensure server is running
        let instance = self.ensure_server(model_id).await?;

        // Check server is healthy
        if instance.state().await != ServerState::Ready {
            return Err(Error::EngineNotAvailable(format!(
                "llama-server for {} is not ready",
                model_id
            )));
        }

        // Build request
        let messages: Vec<LlamaMessage> = request
            .messages
            .iter()
            .map(|m| LlamaMessage {
                role: m.role.clone(),
                content: m.content.clone().unwrap_or_default(),
            })
            .collect();

        let llama_request = LlamaChatRequest {
            model: model_id.to_string(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
        };

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", instance.port);

        tracing::debug!("Sending chat request to llama-server: {}", url);

        let response = self
            .http_client
            .post(&url)
            .json(&llama_request)
            .send()
            .await
            .map_err(|e| {
                // Server might have crashed
                if e.is_connect() {
                    tracing::warn!(
                        "Connection failed to llama-server for {}, marking unhealthy",
                        model_id
                    );
                    // Mark as unhealthy for next request to restart
                    let instance_clone = instance.clone();
                    tokio::spawn(async move {
                        instance_clone.set_state(ServerState::Unhealthy).await;
                    });
                }
                Error::Communication(e.to_string())
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::InferenceFailed(format!("{}: {}", status, body)));
        }

        let llama_response: LlamaChatResponse = response
            .json()
            .await
            .map_err(|e| Error::InferenceFailed(e.to_string()))?;

        let choice = llama_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::InferenceFailed("No choices in response".to_string()))?;

        let message = ChatMessage {
            role: choice.message.role,
            content: Some(choice.message.content),
            tool_calls: None,
            tool_call_id: None,
        };

        let mut response =
            ChatCompletionResponse::new(model_id.to_string(), message, choice.finish_reason);

        if let Some(usage) = llama_response.usage {
            response = response.with_usage(usage.prompt_tokens, usage.completion_tokens);
        }

        instance.touch().await;

        Ok(response)
    }
}

impl Drop for LlamaCppEngine {
    fn drop(&mut self) {
        // Note: We can't do async cleanup in Drop, so we rely on kill_on_drop(true)
        // set during process spawn to clean up child processes.
        tracing::debug!("LlamaCppEngine dropped, child processes will be cleaned up");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LlamaCppEngineConfig {
        LlamaCppEngineConfig {
            enabled: true,
            model_dir: "/tmp/models".to_string(),
            server_binary: "/usr/bin/llama-server".to_string(),
            server_args: vec![],
            gpu_layers: Some(35),
            context_size: Some(4096),
            base_port: None,
            max_servers: 2,
            startup_timeout_secs: 120,
            shutdown_timeout_secs: 10,
            log_server_output: false,
        }
    }

    #[test]
    fn test_engine_type() {
        let engine = LlamaCppEngine::new(test_config());
        assert_eq!(engine.engine_type(), "llama_cpp");
    }

    #[test]
    fn test_model_path_with_extension() {
        let engine = LlamaCppEngine::new(test_config());
        let path = engine.model_path("model.gguf");
        assert_eq!(path.to_str().unwrap(), "/tmp/models/model.gguf");
    }

    #[test]
    fn test_model_path_without_extension() {
        let engine = LlamaCppEngine::new(test_config());
        let path = engine.model_path("model");
        assert_eq!(path.to_str().unwrap(), "/tmp/models/model.gguf");
    }

    #[test]
    fn test_extract_quantization() {
        assert_eq!(
            LlamaCppEngine::extract_quantization("llama-7b-q4_0.gguf"),
            Some("Q4_0".to_string())
        );
        assert_eq!(
            LlamaCppEngine::extract_quantization("model-Q8_0.gguf"),
            Some("Q8_0".to_string())
        );
        assert_eq!(
            LlamaCppEngine::extract_quantization("model-q4_k_m.gguf"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(
            LlamaCppEngine::extract_quantization("model-f16.gguf"),
            Some("F16".to_string())
        );
        assert_eq!(
            LlamaCppEngine::extract_quantization("model.gguf"),
            None
        );
    }

    #[test]
    fn test_server_state_values() {
        // Verify all server states are distinct
        assert_ne!(ServerState::Stopped, ServerState::Starting);
        assert_ne!(ServerState::Starting, ServerState::Ready);
        assert_ne!(ServerState::Ready, ServerState::Unhealthy);
        assert_ne!(ServerState::Unhealthy, ServerState::ShuttingDown);
    }

    #[tokio::test]
    async fn test_port_allocation_with_base_port() {
        let mut config = test_config();
        config.base_port = Some(9000);
        let engine = LlamaCppEngine::new(config);

        let port = engine.allocate_port().await.unwrap();
        assert_eq!(port, 9000);
    }

    #[tokio::test]
    async fn test_port_allocation_dynamic() {
        let config = test_config();
        let engine = LlamaCppEngine::new(config);

        let port = engine.allocate_port().await.unwrap();
        assert!(port > 0);
    }

    #[tokio::test]
    async fn test_health_check_missing_model_dir() {
        let mut config = test_config();
        config.model_dir = "/nonexistent/path".to_string();
        let engine = LlamaCppEngine::new(config);

        let result = engine.health_check().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_models_missing_dir() {
        let mut config = test_config();
        config.model_dir = "/nonexistent/path".to_string();
        let engine = LlamaCppEngine::new(config);

        let models = engine.list_models().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_get_model_not_found() {
        let mut config = test_config();
        config.model_dir = "/nonexistent/path".to_string();
        let engine = LlamaCppEngine::new(config);

        let model = engine.get_model("nonexistent").await.unwrap();
        assert!(model.is_none());
    }

    #[tokio::test]
    async fn test_load_model_file_not_found() {
        let config = test_config();
        let engine = LlamaCppEngine::new(config);

        let result = engine.load_model("nonexistent-model").await;
        assert!(result.is_err());
        match result {
            Err(Error::ModelNotFound(msg)) => {
                assert!(msg.contains("nonexistent-model"));
            }
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_unload_model_not_loaded() {
        let config = test_config();
        let engine = LlamaCppEngine::new(config);

        // Unloading a model that was never loaded should succeed (no-op)
        let result = engine.unload_model("never-loaded").await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_quantization_various_formats() {
        // Test additional quantization patterns
        assert_eq!(
            LlamaCppEngine::extract_quantization("model-Q2_K.gguf"),
            Some("Q2_K".to_string())
        );
        assert_eq!(
            LlamaCppEngine::extract_quantization("llama-Q5_K_S-instruct.gguf"),
            Some("Q5_K_S".to_string())
        );
        assert_eq!(
            LlamaCppEngine::extract_quantization("MODEL-F32.GGUF"),
            Some("F32".to_string())
        );
        // Edge case: no quantization in name
        assert_eq!(
            LlamaCppEngine::extract_quantization("plain-model.gguf"),
            None
        );
    }

    #[tokio::test]
    async fn test_port_allocation_starts_at_base() {
        let mut config = test_config();
        config.base_port = Some(10000);
        let engine = LlamaCppEngine::new(config);

        // First allocation should get base port
        let port1 = engine.allocate_port().await.unwrap();
        assert_eq!(port1, 10000);
    }

    #[test]
    fn test_model_path_sanitizes_traversal() {
        let engine = LlamaCppEngine::new(test_config());

        // Path traversal attempts should be sanitized
        let path = engine.model_path("../../../etc/passwd");
        assert_eq!(path.to_str().unwrap(), "/tmp/models/passwd.gguf");

        let path = engine.model_path("..\\..\\windows\\system32");
        assert_eq!(path.to_str().unwrap(), "/tmp/models/system32.gguf");

        // Hidden file prefix should be stripped
        let path = engine.model_path("..hidden");
        assert_eq!(path.to_str().unwrap(), "/tmp/models/hidden.gguf");

        // Normal model names should work fine
        let path = engine.model_path("llama-7b-q4_0");
        assert_eq!(path.to_str().unwrap(), "/tmp/models/llama-7b-q4_0.gguf");
    }
}
