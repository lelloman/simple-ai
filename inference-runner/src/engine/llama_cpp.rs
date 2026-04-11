//! llama.cpp inference engine implementation.
//!
//! This engine manages llama-server subprocesses to provide inference capabilities.
//! Each loaded model runs in its own llama-server process for isolation.

use std::collections::{HashMap, HashSet};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Constants for polling intervals
const HEALTH_CHECK_INTERVAL_MS: u64 = 200;
const SERVER_STARTING_POLL_MS: u64 = 100;

use async_trait::async_trait;
use axum::body::Bytes;
use futures_util::stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use simple_ai_common::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolCall, ToolFunction,
};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::sync::{RwLock, Semaphore};

use super::{ChatCompletionStream, EngineHealth, InferenceEngine, ModelInfo};
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
    /// Estimated VRAM usage in GB for this model.
    memory_gb: f32,
    state: RwLock<ServerState>,
    process: RwLock<Option<Child>>,
    last_used: RwLock<Instant>,
}

impl ServerInstance {
    fn new(model_id: String, port: u16, memory_gb: f32, process: Child) -> Self {
        Self {
            model_id,
            port,
            memory_gb,
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

    /// Check if the server process is still alive.
    async fn is_process_alive(&self) -> bool {
        let mut process = self.process.write().await;
        if let Some(ref mut child) = *process {
            match child.try_wait() {
                Ok(None) => true,     // Still running
                Ok(Some(_)) => false, // Exited
                Err(_) => false,      // Error checking
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
            let wait_result =
                tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await;

            match wait_result {
                Ok(Ok(status)) => {
                    tracing::debug!("llama-server for {} exited with {}", self.model_id, status);
                }
                Ok(Err(e)) => {
                    tracing::warn!("Error waiting for llama-server {}: {}", self.model_id, e);
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
    /// Cache of discovered model_id -> full file path
    model_paths: RwLock<HashMap<String, PathBuf>>,
}

impl LlamaCppEngine {
    pub fn new(config: LlamaCppEngineConfig) -> Self {
        Self {
            startup_semaphore: Semaphore::new(1), // Only one server startup at a time
            config,
            http_client: Client::new(),
            servers: RwLock::new(HashMap::new()),
            model_paths: RwLock::new(HashMap::new()),
        }
    }

    /// Recursively discover all GGUF model files in the model directory.
    /// Returns a map of model_id -> full path, filtering out non-primary shards.
    fn discover_models(&self) -> HashMap<String, PathBuf> {
        let mut models = HashMap::new();
        let model_dir = PathBuf::from(&self.config.model_dir);

        if !model_dir.exists() {
            return models;
        }

        self.scan_directory_recursive(&model_dir, &mut models);
        models
    }

    /// Recursively scan a directory for GGUF files.
    fn scan_directory_recursive(&self, dir: &PathBuf, models: &mut HashMap<String, PathBuf>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                self.scan_directory_recursive(&path, models);
            } else if path
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("gguf"))
            {
                let filename = match path.file_name().and_then(|s| s.to_str()) {
                    Some(f) => f,
                    None => continue,
                };

                // Skip non-primary shards (e.g., -00002-of-00003.gguf)
                if Self::is_non_primary_shard(filename) {
                    continue;
                }

                // Create a nice model ID from the filename
                let model_id = Self::create_model_id(filename);
                models.insert(model_id, path);
            }
        }
    }

    /// Check if a filename is a non-primary shard (not the first part of a split model).
    fn is_non_primary_shard(filename: &str) -> bool {
        // Match patterns like -00002-of-00003.gguf, -00003-of-00005.gguf, etc.
        // Primary shards have -00001-of-XXXXX
        if let Some(pos) = filename.find("-of-") {
            // Look backwards from "-of-" to find the shard number
            let prefix = &filename[..pos];
            if let Some(dash_pos) = prefix.rfind('-') {
                let shard_num = &prefix[dash_pos + 1..];
                // If it's not "00001", it's a non-primary shard
                if shard_num.chars().all(|c| c.is_ascii_digit()) && shard_num != "00001" {
                    return true;
                }
            }
        }
        false
    }

    /// Create a nice model ID from a GGUF filename.
    fn create_model_id(filename: &str) -> String {
        let stem = filename.strip_suffix(".gguf").unwrap_or(filename);

        // Remove shard suffix like -00001-of-00002
        let clean = if let Some(pos) = stem.find("-00001-of-") {
            &stem[..pos]
        } else {
            stem
        };

        clean.to_string()
    }

    /// Get the path to a model file from the cache.
    async fn model_path(&self, model_id: &str) -> Option<PathBuf> {
        let paths = self.model_paths.read().await;
        paths.get(model_id).cloned()
    }

    /// Determine the context size to use for a model server.
    ///
    /// A configured value is treated as an explicit operator override. When it
    /// is absent, use the maximum context length declared in GGUF metadata.
    fn effective_context_size(&self, model_path: &Path) -> Option<u32> {
        self.config
            .context_size
            .or_else(|| Self::read_gguf_context_length(model_path))
    }

    /// Refresh the model paths cache by scanning the model directory.
    async fn refresh_model_cache(&self) {
        let discovered = self.discover_models();
        let mut paths = self.model_paths.write().await;
        *paths = discovered;
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
            let listener = TcpListener::bind("127.0.0.1:0").await.map_err(|e| {
                Error::Internal(format!("Failed to bind for port allocation: {}", e))
            })?;
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

    /// Get the memory footprint for a model in GB.
    /// Uses config override if available, otherwise estimates from file size.
    async fn get_model_memory_gb(&self, model_id: &str) -> f32 {
        // Check config override first
        if let Some(&memory) = self.config.model_memory_gb.get(model_id) {
            return memory;
        }

        // Fall back to file size estimation
        if let Some(path) = self.model_path(model_id).await {
            if let Ok(metadata) = std::fs::metadata(&path) {
                let size_gb = metadata.len() as f32 / (1024.0 * 1024.0 * 1024.0);
                // Add ~10% overhead for runtime allocations
                return size_gb * 1.1;
            }
        }

        // Default fallback if we can't determine size
        4.0
    }

    /// Calculate total VRAM currently used by loaded servers.
    async fn get_used_vram_gb(&self) -> f32 {
        let servers = self.servers.read().await;
        let mut total = 0.0;
        for instance in servers.values() {
            let state = instance.state().await;
            // Count memory for servers that are running or starting
            if state != ServerState::ShuttingDown && state != ServerState::Stopped {
                total += instance.memory_gb;
            }
        }
        total
    }

    /// Find the least recently used server to evict, excluding a specific model.
    /// Returns None if there are no servers to evict.
    async fn find_lru_server(&self, exclude_model: &str) -> Option<(String, f32)> {
        let servers = self.servers.read().await;
        let mut oldest: Option<(String, f32, Instant)> = None;

        for (model_id, instance) in servers.iter() {
            // Skip the model we're trying to load
            if model_id == exclude_model {
                continue;
            }

            // Skip servers that are not in a state we can evict
            let state = instance.state().await;
            if state == ServerState::ShuttingDown || state == ServerState::Starting {
                continue;
            }

            let last_used = *instance.last_used.read().await;

            match &oldest {
                None => oldest = Some((model_id.clone(), instance.memory_gb, last_used)),
                Some((_, _, oldest_time)) if last_used < *oldest_time => {
                    oldest = Some((model_id.clone(), instance.memory_gb, last_used));
                }
                _ => {}
            }
        }

        oldest.map(|(model_id, memory, _)| (model_id, memory))
    }

    /// Evict servers to make room for a new model based on memory constraints.
    /// Uses max_vram_gb if configured, otherwise falls back to max_servers count.
    async fn ensure_capacity(&self, model_to_load: &str, required_memory_gb: f32) -> Result<()> {
        // If max_vram_gb is configured, use memory-based eviction
        if let Some(max_vram) = self.config.max_vram_gb {
            loop {
                let used_vram = self.get_used_vram_gb().await;
                let available = max_vram - used_vram;

                if available >= required_memory_gb {
                    tracing::debug!(
                        "Memory check passed: need {:.2}GB, available {:.2}GB (used {:.2}GB / max {:.2}GB)",
                        required_memory_gb,
                        available,
                        used_vram,
                        max_vram
                    );
                    return Ok(());
                }

                // Need to evict - find LRU server
                let lru = self.find_lru_server(model_to_load).await;

                match lru {
                    Some((model_id, memory)) => {
                        tracing::info!(
                            "Evicting LRU model {} ({:.2}GB) to free memory for {} ({:.2}GB needed, {:.2}GB available)",
                            model_id,
                            memory,
                            model_to_load,
                            required_memory_gb,
                            available
                        );
                        self.unload_model(&model_id).await?;
                    }
                    None => {
                        // Distinguish between "model too large" and "no models to evict"
                        if required_memory_gb > max_vram {
                            return Err(Error::Internal(format!(
                                "Cannot load model {}: requires {:.2}GB but max_vram_gb is only {:.2}GB",
                                model_to_load,
                                required_memory_gb,
                                max_vram
                            )));
                        }
                        return Err(Error::Internal(format!(
                            "Cannot load model {}: requires {:.2}GB but only {:.2}GB available and no models to evict",
                            model_to_load,
                            required_memory_gb,
                            available
                        )));
                    }
                }
            }
        } else {
            // Fall back to count-based limiting using max_servers
            loop {
                let current_count = {
                    let servers = self.servers.read().await;
                    let mut count = 0;
                    for instance in servers.values() {
                        let state = instance.state().await;
                        if state != ServerState::ShuttingDown && state != ServerState::Stopped {
                            count += 1;
                        }
                    }
                    count
                };

                if current_count < self.config.max_servers {
                    return Ok(());
                }

                // Need to evict - find LRU server
                let lru = self.find_lru_server(model_to_load).await;

                match lru {
                    Some((model_id, _)) => {
                        tracing::info!(
                            "Evicting LRU model {} to make room (current: {}, max: {})",
                            model_id,
                            current_count,
                            self.config.max_servers
                        );
                        self.unload_model(&model_id).await?;
                    }
                    None => {
                        return Err(Error::Internal(format!(
                            "Cannot make room for model {}: at capacity ({}) with no evictable servers",
                            model_to_load,
                            self.config.max_servers
                        )));
                    }
                }
            }
        }
    }

    /// Start a llama-server process for a model.
    async fn start_server(&self, model_id: &str, memory_gb: f32) -> Result<Arc<ServerInstance>> {
        // Look up model path from cache
        let model_path = self
            .model_path(model_id)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("Model not found: {}", model_id)))?;

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

        if let Some(ctx_size) = self.effective_context_size(&model_path) {
            cmd.arg("-c").arg(ctx_size.to_string());
        }

        // Enable embedding mode for embedding models
        if Self::is_embedding_model(model_id) {
            cmd.arg("--embedding");
        }

        // Append extra llama-server arguments (e.g., --flash-attn on --no-mmap)
        for arg in &self.config.extra_args {
            cmd.arg(arg);
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
                model_id, e, self.config.server_binary
            ))
        })?;

        tracing::info!(
            "Spawned llama-server for {} on port {} (pid: {:?}, memory: {:.2}GB)",
            model_id,
            port,
            process.id(),
            memory_gb
        );

        let instance = Arc::new(ServerInstance::new(
            model_id.to_string(),
            port,
            memory_gb,
            process,
        ));

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

        // Get model memory requirement for capacity planning
        let memory_gb = self.get_model_memory_gb(model_id).await;

        // Ensure we have capacity for a new server (evict LRU if needed)
        self.ensure_capacity(model_id, memory_gb).await?;

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
        let instance = self.start_server(model_id, memory_gb).await?;

        // Store in map
        {
            let mut servers = self.servers.write().await;
            servers.insert(model_id.to_string(), instance.clone());
        }

        Ok(instance)
    }

    /// Check if a model is an embedding model based on its name.
    fn is_embedding_model(model_id: &str) -> bool {
        model_id.to_lowercase().contains("embed")
    }

    /// Extract quantization type from filename (e.g., "model-q4_0.gguf" -> "Q4_0").
    fn extract_quantization(filename: &str) -> Option<String> {
        let patterns = [
            "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0",
            "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "f16", "f32",
        ];

        let lower = filename.to_lowercase();
        for pattern in patterns {
            if lower.contains(pattern) {
                return Some(pattern.to_uppercase());
            }
        }
        None
    }

    fn read_gguf_context_length(path: &Path) -> Option<u32> {
        match Self::read_gguf_context_length_inner(path) {
            Ok(value) => value,
            Err(e) => {
                tracing::debug!(
                    "Could not read GGUF context length from {}: {}",
                    path.display(),
                    e
                );
                None
            }
        }
    }

    fn read_gguf_context_length_inner(path: &Path) -> io::Result<Option<u32>> {
        let mut file = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Ok(None);
        }

        let _version = read_u32(&mut file)?;
        let _tensor_count = read_u64(&mut file)?;
        let metadata_count = read_u64(&mut file)?;

        for _ in 0..metadata_count {
            let key = read_gguf_string(&mut file)?;
            let value_type = read_u32(&mut file)?;

            if key.ends_with(".context_length") || key == "context_length" {
                return read_gguf_u32_value(&mut file, value_type);
            }

            skip_gguf_value(&mut file, value_type)?;
        }

        Ok(None)
    }
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(reader: &mut impl Read) -> io::Result<i64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_gguf_string(reader: &mut impl Read) -> io::Result<String> {
    let len = read_u64(reader)? as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_gguf_u32_value(reader: &mut impl Read, value_type: u32) -> io::Result<Option<u32>> {
    match value_type {
        4 => Ok(Some(read_u32(reader)?)),
        5 => Ok(u32::try_from(read_i32(reader)?).ok()),
        10 => Ok(u32::try_from(read_u64(reader)?).ok()),
        11 => Ok(u32::try_from(read_i64(reader)?).ok()),
        _ => {
            skip_gguf_value(reader, value_type)?;
            Ok(None)
        }
    }
}

fn skip_gguf_value(reader: &mut impl Read, value_type: u32) -> io::Result<()> {
    match value_type {
        0 | 1 | 7 => skip_exact(reader, 1),
        2 | 3 => skip_exact(reader, 2),
        4 | 5 | 6 => skip_exact(reader, 4),
        8 => {
            let len = read_u64(reader)?;
            skip_exact(reader, len)
        }
        9 => {
            let item_type = read_u32(reader)?;
            let len = read_u64(reader)?;
            for _ in 0..len {
                skip_gguf_value(reader, item_type)?;
            }
            Ok(())
        }
        10 | 11 | 12 => skip_exact(reader, 8),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF metadata value type {}", value_type),
        )),
    }
}

fn skip_exact(reader: &mut impl Read, len: u64) -> io::Result<()> {
    let mut remaining = len;
    let mut buffer = [0u8; 8192];

    while remaining > 0 {
        let chunk_len = remaining.min(buffer.len() as u64) as usize;
        reader.read_exact(&mut buffer[..chunk_len])?;
        remaining -= chunk_len as u64;
    }

    Ok(())
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
    tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct LlamaMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<LlamaToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LlamaToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: LlamaToolFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct LlamaToolFunction {
    name: String,
    arguments: LlamaToolArguments,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum LlamaToolArguments {
    Text(String),
    Json(Value),
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
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<LlamaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct LlamaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// Request body for llama-server /v1/embeddings endpoint.
#[derive(Debug, Serialize)]
struct LlamaEmbeddingRequest {
    input: Vec<String>,
}

/// Response from llama-server /v1/embeddings endpoint.
#[derive(Debug, Deserialize)]
struct LlamaEmbeddingResponse {
    data: Vec<LlamaEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct LlamaEmbeddingData {
    embedding: Vec<f32>,
}

// ============================================================================
// InferenceEngine implementation
// ============================================================================

#[async_trait]
impl InferenceEngine for LlamaCppEngine {
    fn engine_type(&self) -> &'static str {
        "llama_cpp"
    }

    fn batch_size(&self) -> u32 {
        self.config.batch_size
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
        // Refresh the model cache
        self.refresh_model_cache().await;

        let paths = self.model_paths.read().await;
        let mut models = Vec::new();

        for (model_id, path) in paths.iter() {
            let filename = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let size_bytes = std::fs::metadata(path).map(|m| m.len()).ok();
            let quantization = Self::extract_quantization(filename);
            let context_length = self.effective_context_size(path);

            models.push(ModelInfo {
                id: model_id.clone(),
                name: model_id.clone(),
                size_bytes,
                parameter_count: None,
                context_length,
                quantization,
                modified_at: None,
            });
        }

        // Sort by name for consistent ordering
        models.sort_by(|a, b| a.name.cmp(&b.name));

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
            instance.terminate(self.config.shutdown_timeout_secs).await;
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
                content: m.content.clone(),
                tool_calls: m.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| LlamaToolCall {
                            id: tc.id.clone(),
                            call_type: tc.call_type.clone(),
                            function: LlamaToolFunction {
                                name: tc.function.name.clone(),
                                arguments: LlamaToolArguments::Text(tc.function.arguments.clone()),
                            },
                        })
                        .collect()
                }),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();

        let llama_request = LlamaChatRequest {
            model: model_id.to_string(),
            messages,
            tools: request.tools.clone(),
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

        let tool_calls = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| ToolCall {
                    id: tc.id,
                    call_type: tc.call_type,
                    function: ToolFunction {
                        name: tc.function.name,
                        arguments: match tc.function.arguments {
                            LlamaToolArguments::Text(value) => value,
                            LlamaToolArguments::Json(value) => value.to_string(),
                        },
                    },
                })
                .collect()
        });

        // For thinking models, use reasoning_content as fallback if content is empty
        let response_content = match (choice.message.content, choice.message.reasoning_content) {
            (Some(content), _) if !content.is_empty() => Some(content),
            (_, Some(reasoning)) if !reasoning.is_empty() => Some(reasoning),
            _ => None,
        };

        let message = ChatMessage {
            role: choice.message.role,
            content: response_content,
            tool_calls,
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

    async fn chat_completion_stream(
        &self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        let instance = self.ensure_server(model_id).await?;

        if instance.state().await != ServerState::Ready {
            return Err(Error::EngineNotAvailable(format!(
                "llama-server for {} is not ready",
                model_id
            )));
        }

        let messages: Vec<LlamaMessage> = request
            .messages
            .iter()
            .map(|m| LlamaMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: m.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| LlamaToolCall {
                            id: tc.id.clone(),
                            call_type: tc.call_type.clone(),
                            function: LlamaToolFunction {
                                name: tc.function.name.clone(),
                                arguments: LlamaToolArguments::Text(tc.function.arguments.clone()),
                            },
                        })
                        .collect()
                }),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();

        let llama_request = LlamaChatRequest {
            model: model_id.to_string(),
            messages,
            tools: request.tools.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: true,
        };

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", instance.port);

        tracing::debug!("Sending streaming chat request to llama-server: {}", url);

        let response = self
            .http_client
            .post(&url)
            .json(&llama_request)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    tracing::warn!(
                        "Connection failed to llama-server for {}, marking unhealthy",
                        model_id
                    );
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

        let stream = stream::try_unfold(response, |mut response| async move {
            let next = response
                .chunk()
                .await
                .map_err(|e| Error::Communication(e.to_string()))?;

            Ok(next.map(|chunk| (Bytes::from(chunk), response)))
        });

        Ok(Box::pin(stream))
    }

    async fn embed(&self, model_id: &str, input: &[String]) -> Result<Vec<Vec<f32>>> {
        if !Self::is_embedding_model(model_id) {
            return Err(Error::InvalidRequest(format!(
                "Model {} is not an embedding model",
                model_id
            )));
        }

        let instance = self.ensure_server(model_id).await?;

        if instance.state().await != ServerState::Ready {
            return Err(Error::EngineNotAvailable(format!(
                "llama-server for {} is not ready",
                model_id
            )));
        }

        let url = format!("http://127.0.0.1:{}/v1/embeddings", instance.port);

        tracing::debug!("Sending embedding request to llama-server: {}", url);

        let request = LlamaEmbeddingRequest {
            input: input.to_vec(),
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    tracing::warn!(
                        "Connection failed to llama-server for {}, marking unhealthy",
                        model_id
                    );
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

        let embed_response: LlamaEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| Error::InferenceFailed(e.to_string()))?;

        let embeddings = embed_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect();

        instance.touch().await;

        Ok(embeddings)
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
            max_vram_gb: None,
            model_memory_gb: HashMap::new(),
            startup_timeout_secs: 120,
            shutdown_timeout_secs: 10,
            log_server_output: false,
            extra_args: vec![],
            batch_size: 1,
        }
    }

    #[test]
    fn test_engine_type() {
        let engine = LlamaCppEngine::new(test_config());
        assert_eq!(engine.engine_type(), "llama_cpp");
    }

    #[test]
    fn test_read_gguf_context_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let mut bytes = Vec::new();

        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&2u64.to_le_bytes()); // metadata count

        write_gguf_string(&mut bytes, "general.name");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string
        write_gguf_string(&mut bytes, "test-model");

        write_gguf_string(&mut bytes, "llama.context_length");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // uint32
        bytes.extend_from_slice(&32768u32.to_le_bytes());

        std::fs::write(&path, bytes).unwrap();

        assert_eq!(LlamaCppEngine::read_gguf_context_length(&path), Some(32768));
    }

    fn write_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn test_is_non_primary_shard() {
        // Primary shards should return false
        assert!(!LlamaCppEngine::is_non_primary_shard(
            "model-00001-of-00002.gguf"
        ));
        assert!(!LlamaCppEngine::is_non_primary_shard(
            "gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"
        ));

        // Non-primary shards should return true
        assert!(LlamaCppEngine::is_non_primary_shard(
            "model-00002-of-00002.gguf"
        ));
        assert!(LlamaCppEngine::is_non_primary_shard(
            "model-00003-of-00005.gguf"
        ));
        assert!(LlamaCppEngine::is_non_primary_shard(
            "gpt-oss-120b-Q4_K_M-00002-of-00002.gguf"
        ));

        // Non-sharded models should return false
        assert!(!LlamaCppEngine::is_non_primary_shard("model.gguf"));
        assert!(!LlamaCppEngine::is_non_primary_shard("llama-7b-q4_0.gguf"));
    }

    #[test]
    fn test_create_model_id() {
        // Non-sharded models
        assert_eq!(
            LlamaCppEngine::create_model_id("llama-7b-q4_0.gguf"),
            "llama-7b-q4_0"
        );
        assert_eq!(
            LlamaCppEngine::create_model_id("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"),
            "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"
        );

        // Sharded models - should strip shard suffix
        assert_eq!(
            LlamaCppEngine::create_model_id("gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"),
            "gpt-oss-120b-Q4_K_M"
        );
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
        assert_eq!(LlamaCppEngine::extract_quantization("model.gguf"), None);
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

    #[tokio::test]
    async fn test_get_model_memory_gb_from_config() {
        let mut config = test_config();
        config.model_memory_gb.insert("test-model".to_string(), 8.5);
        let engine = LlamaCppEngine::new(config);

        let memory = engine.get_model_memory_gb("test-model").await;
        assert!((memory - 8.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_get_model_memory_gb_fallback() {
        let config = test_config();
        let engine = LlamaCppEngine::new(config);

        // Model not in config and file doesn't exist, should return default 4.0
        let memory = engine.get_model_memory_gb("nonexistent").await;
        assert!((memory - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_is_embedding_model() {
        assert!(LlamaCppEngine::is_embedding_model(
            "Qwen3-Embedding-8B-Q4_K_M"
        ));
        assert!(LlamaCppEngine::is_embedding_model("qwen3-embedding-0.6b"));
        assert!(LlamaCppEngine::is_embedding_model("nomic-embed-text"));
        assert!(LlamaCppEngine::is_embedding_model("mxbai-embed-large"));
        assert!(!LlamaCppEngine::is_embedding_model("qwen3-4b"));
        assert!(!LlamaCppEngine::is_embedding_model("gpt-oss-120b-Q4_K_M"));
        assert!(!LlamaCppEngine::is_embedding_model("Qwen3-Coder-30B"));
    }

    #[tokio::test]
    async fn test_get_used_vram_empty() {
        let config = test_config();
        let engine = LlamaCppEngine::new(config);

        let used = engine.get_used_vram_gb().await;
        assert!((used - 0.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_find_lru_server_empty() {
        let config = test_config();
        let engine = LlamaCppEngine::new(config);

        let lru = engine.find_lru_server("any-model").await;
        assert!(lru.is_none());
    }

    #[tokio::test]
    async fn test_ensure_capacity_with_vram_limit_empty() {
        let mut config = test_config();
        config.max_vram_gb = Some(24.0);
        let engine = LlamaCppEngine::new(config);

        // Should succeed when no models loaded and within limit
        let result = engine.ensure_capacity("new-model", 8.0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_ensure_capacity_model_too_large() {
        let mut config = test_config();
        config.max_vram_gb = Some(8.0);
        let engine = LlamaCppEngine::new(config);

        // Model requires more than max_vram_gb allows
        let result = engine.ensure_capacity("huge-model", 12.0).await;
        assert!(result.is_err());

        // Check error message mentions the size issue
        if let Err(Error::Internal(msg)) = result {
            assert!(msg.contains("max_vram_gb is only"));
        } else {
            panic!("Expected Internal error");
        }
    }

    #[tokio::test]
    async fn test_ensure_capacity_count_based_empty() {
        let mut config = test_config();
        config.max_servers = 2;
        config.max_vram_gb = None; // Use count-based
        let engine = LlamaCppEngine::new(config);

        // Should succeed when no models loaded
        let result = engine.ensure_capacity("new-model", 8.0).await;
        assert!(result.is_ok());
    }
}
