//! Configuration for the inference runner.

use config::{Config as ConfigLoader, ConfigError, Environment, File};
use serde::Deserialize;
use simple_ai_common::Capability;
use std::collections::HashMap;

/// Model alias configuration for mapping canonical names to local engine names.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct AliasesConfig {
    /// Maps canonical model names to local engine names.
    /// Example: { "gemma-3-4b-it-Q4_K_M" = "google_gemma-3-4b-it-Q4_K_M" }
    #[serde(default)]
    pub mappings: HashMap<String, String>,
}

/// Main configuration structure for the inference runner.
///
/// Note: Some fields are for Phase 2+ features (gateway connection, capability mappings).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Config {
    pub runner: RunnerConfig,
    pub api: ApiConfig,
    /// Gateway WebSocket connection config (Phase 2)
    #[serde(default)]
    pub gateway: Option<GatewayConfig>,
    #[serde(default)]
    pub engines: EnginesConfig,
    /// Capability-to-model mappings (Phase 2)
    #[serde(default)]
    pub capabilities: CapabilitiesConfig,
    /// Model aliases for mapping canonical names to local engine names.
    #[serde(default)]
    pub aliases: AliasesConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct RunnerConfig {
    pub id: String,
    pub name: String,
    /// Machine type for routing decisions (Phase 2)
    #[serde(default)]
    pub machine_type: Option<String>,
    /// MAC address for Wake-on-LAN (format: AA:BB:CC:DD:EE:FF)
    #[serde(default)]
    pub mac_address: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

/// Gateway WebSocket connection configuration (Phase 2).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct GatewayConfig {
    pub ws_url: String,
    pub auth_token: String,
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay_secs: u64,
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u64,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[allow(dead_code)]
pub struct EnginesConfig {
    pub ollama: Option<OllamaEngineConfig>,
    /// llama.cpp engine configuration (Phase 3)
    pub llama_cpp: Option<LlamaCppEngineConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaEngineConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
    /// Maximum batch size for concurrent inference (default: 1 = no batching).
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
}

/// llama.cpp engine configuration (Phase 3).
///
/// This engine manages llama-server subprocesses to provide inference capabilities.
/// Each loaded model runs in its own llama-server process for isolation.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaCppEngineConfig {
    /// Whether this engine is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Directory containing .gguf model files.
    pub model_dir: String,
    /// Path to the llama-server binary or wrapper command (e.g., "toolbox").
    pub server_binary: String,
    /// Extra arguments inserted BEFORE the model args.
    /// Use this for wrapper commands like toolbox.
    /// Example: `["run", "-c", "llamacpp", "llama-server"]`
    #[serde(default)]
    pub server_args: Vec<String>,
    /// Number of layers to offload to GPU (-ngl flag). 0 = CPU only.
    #[serde(default)]
    pub gpu_layers: Option<u32>,
    /// Context window size (-c flag).
    #[serde(default)]
    pub context_size: Option<u32>,
    /// Base port for server allocation. If not set, OS assigns ports dynamically.
    #[serde(default)]
    pub base_port: Option<u16>,
    /// Maximum number of concurrent model servers (default: 2).
    #[serde(default = "default_max_servers")]
    pub max_servers: usize,
    /// Maximum VRAM available for loading models (in GB).
    /// If not set, falls back to max_servers count-based limiting.
    #[serde(default)]
    pub max_vram_gb: Option<f32>,
    /// Per-model memory overrides (in GB).
    /// Maps model_id to its VRAM footprint. If not specified for a model,
    /// the file size is used as an estimate.
    #[serde(default)]
    pub model_memory_gb: HashMap<String, f32>,
    /// Server startup timeout in seconds (default: 120).
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,
    /// Server graceful shutdown timeout in seconds (default: 10).
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout_secs: u64,
    /// Log llama-server stderr output for debugging (default: false).
    #[serde(default)]
    pub log_server_output: bool,
    /// Extra arguments appended AFTER the standard llama-server flags.
    /// Use this for flags like `["--flash-attn", "on", "--no-mmap"]`
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Maximum batch size for concurrent inference (default: 1 = no batching).
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
}

/// Capability configuration (Phase 2).
#[derive(Debug, Clone, Deserialize, Default)]
#[allow(dead_code)]
pub struct CapabilitiesConfig {
    #[serde(default)]
    pub persistence: PersistenceConfig,
    /// Maps model IDs to capabilities they provide.
    /// Example: { "llama3.2:3b": ["fast_chat"], "qwen2.5:72b": ["large_chat"] }
    #[serde(default)]
    pub mappings: HashMap<String, Vec<Capability>>,
}

/// Model persistence configuration (Phase 2).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct PersistenceConfig {
    /// Models to keep loaded even when idle.
    #[serde(default)]
    pub always_loaded: Vec<String>,
    /// Time before unloading idle models (seconds).
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout_secs: u64,
    /// Maximum number of models to keep loaded simultaneously.
    #[serde(default = "default_max_loaded")]
    pub max_loaded_models: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            always_loaded: vec![],
            idle_timeout_secs: default_idle_timeout(),
            max_loaded_models: default_max_loaded(),
        }
    }
}

// Default values
fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_reconnect_delay() -> u64 {
    5
}
fn default_heartbeat_interval() -> u64 {
    30
}
fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}
fn default_idle_timeout() -> u64 {
    300
}
fn default_max_loaded() -> usize {
    3
}
fn default_true() -> bool {
    true
}
fn default_batch_size() -> u32 {
    1
}
fn default_max_servers() -> usize {
    2
}
fn default_startup_timeout() -> u64 {
    120
}
fn default_shutdown_timeout() -> u64 {
    10
}

impl Config {
    /// Load configuration from file and environment variables.
    ///
    /// Configuration sources (in order of precedence):
    /// 1. Environment variables (RUNNER__SECTION__KEY format)
    /// 2. config.toml file (if present)
    /// 3. Built-in defaults
    pub fn load() -> Result<Self, ConfigError> {
        let config = ConfigLoader::builder()
            // Set defaults
            .set_default("api.host", default_host())?
            .set_default("api.port", default_port() as i64)?
            .set_default("capabilities.persistence.idle_timeout_secs", default_idle_timeout() as i64)?
            .set_default("capabilities.persistence.max_loaded_models", default_max_loaded() as i64)?
            // Load from config.toml if exists
            .add_source(File::with_name("config").required(false))
            // Override with environment variables (RUNNER__SECTION__KEY format)
            .add_source(
                Environment::with_prefix("RUNNER")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?;

        config.try_deserialize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_api_config() {
        let api = ApiConfig::default();
        assert_eq!(api.host, "0.0.0.0");
        assert_eq!(api.port, 8080);
    }

    #[test]
    fn test_default_persistence_config() {
        let persistence = PersistenceConfig::default();
        assert!(persistence.always_loaded.is_empty());
        assert_eq!(persistence.idle_timeout_secs, 300);
        assert_eq!(persistence.max_loaded_models, 3);
    }
}
