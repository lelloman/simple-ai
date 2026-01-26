use std::collections::HashMap;

use config::{Config as ConfigLoader, ConfigError as ConfigCrateError, Environment, File};
use serde::Deserialize;

/// Application configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub ollama: OllamaConfig,
    pub oidc: OidcConfig,
    #[serde(default)]
    pub database: DatabaseConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub cors: CorsConfig,
    #[serde(default)]
    pub language: LanguageConfig,
    /// Gateway configuration for runner fleet management.
    #[serde(default)]
    pub gateway: GatewayConfig,
    /// Wake-on-LAN configuration.
    #[serde(default)]
    pub wol: WolConfig,
    /// Model classification configuration.
    #[serde(default)]
    pub models: ModelsConfig,
    /// Smart routing configuration.
    #[serde(default)]
    pub routing: RoutingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_base_url")]
    pub base_url: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OidcConfig {
    pub issuer: String,
    pub audience: String,
    /// Path to roles in JWT claims (e.g., "roles", "realm_access.roles").
    /// Supports dot-separated paths. Default: "roles"
    #[serde(default = "default_role_claim_path")]
    pub role_claim_path: String,
    /// Name of the admin role. Default: "admin"
    #[serde(default = "default_admin_role")]
    pub admin_role: String,
    /// Explicit list of user subject IDs who are always admins.
    /// Useful when OIDC provider doesn't include roles in tokens.
    #[serde(default)]
    pub admin_users: Vec<String>,
}

fn default_role_claim_path() -> String {
    "roles".to_string()
}

fn default_admin_role() -> String {
    "admin".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    #[serde(default = "default_database_url")]
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CorsConfig {
    #[serde(default = "default_cors_origins")]
    pub origins: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LanguageConfig {
    #[serde(default = "default_language_model_path")]
    pub model_path: String,
}

/// Gateway configuration for runner fleet management.
#[derive(Debug, Clone, Deserialize)]
pub struct GatewayConfig {
    /// Whether gateway mode is enabled.
    /// When enabled, requests are routed to connected inference runners.
    /// When disabled, requests go directly to the configured Ollama instance.
    #[serde(default)]
    pub enabled: bool,
    /// Authentication token for runner connections.
    /// Runners must provide this token to connect.
    #[serde(default = "default_gateway_auth_token")]
    pub auth_token: String,
    /// Timeout for stale runner removal (seconds).
    #[serde(default = "default_runner_timeout")]
    pub runner_timeout_secs: u64,
    /// URL of idle-manager service (e.g., "http://idle-manager:8090").
    /// If set, WOL requests are sent to idle-manager instead of direct WOL.
    #[serde(default)]
    pub idle_manager_url: Option<String>,
    /// Timeout for waiting for runners to wake (seconds). Default: 90
    #[serde(default = "default_wake_timeout")]
    pub wake_timeout_secs: u64,
    /// Whether to auto-wake runners when no runners available. Default: false
    #[serde(default)]
    pub auto_wake_enabled: bool,
    /// Enable request batching for non-streaming requests.
    #[serde(default)]
    pub batching_enabled: bool,
    /// Maximum time to wait for batch to fill (milliseconds). Default: 50
    #[serde(default = "default_batch_timeout_ms")]
    pub batch_timeout_ms: u64,
    /// Minimum batch size before sending (if timeout not reached). Default: 1
    #[serde(default = "default_min_batch_size")]
    pub min_batch_size: u32,
}

/// Wake-on-LAN configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct WolConfig {
    /// Broadcast address for WOL packets (used when sending directly).
    #[serde(default = "default_wol_broadcast")]
    pub broadcast_address: String,
    /// UDP port for WOL packets (typically 9 or 7).
    #[serde(default = "default_wol_port")]
    pub port: u16,
    /// URL of WOL bouncer service. If set, WOL packets are sent via this service
    /// instead of directly. This is useful when running in Docker.
    #[serde(default)]
    pub bouncer_url: Option<String>,
}

/// Model classification configuration.
///
/// Models are classified into classes for routing and permissions:
/// - `big`: Large models (70B+ parameters) - slower but more capable
/// - `fast`: Smaller models - faster inference
///
/// Models not listed in either list are not available for class-based requests.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsConfig {
    /// Models classified as "big" (large/capable).
    /// Exact model IDs (case-insensitive).
    #[serde(default)]
    pub big: Vec<String>,
    /// Models classified as "fast" (small/quick).
    /// Exact model IDs (case-insensitive).
    #[serde(default)]
    pub fast: Vec<String>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            big: vec![],
            fast: vec![],
        }
    }
}

impl ModelsConfig {
    /// Classify a model ID into a class name.
    ///
    /// Returns Some("big") if in the `big` list, Some("fast") if in the `fast` list,
    /// or None if the model is not configured in either list.
    pub fn classify(&self, model_id: &str) -> Option<&'static str> {
        let lower = model_id.to_lowercase();

        for id in &self.big {
            if lower == id.to_lowercase() {
                return Some("big");
            }
        }

        for id in &self.fast {
            if lower == id.to_lowercase() {
                return Some("fast");
            }
        }

        None
    }
}

/// Smart routing configuration.
///
/// Controls how requests are routed to runners, including:
/// - Class-based machine preferences (e.g., `class:fast` prefers gpu-server)
/// - Queue-aware routing (prefer runners with fewer pending requests)
/// - Latency-aware routing (use historical metrics to prefer faster runners)
/// - Speculative wake (wake multiple machines for faster response)
#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    /// Machine preferences by model class.
    /// Key is the class name (e.g., "fast", "big"), value is ordered list of machine types.
    /// First machine type in the list is most preferred.
    #[serde(default)]
    pub class_preferences: HashMap<String, Vec<String>>,
    /// Weight for queue depth in runner scoring (0.0 to 1.0).
    /// Higher values prefer runners with fewer pending requests.
    #[serde(default = "default_queue_weight")]
    pub queue_weight: f64,
    /// Weight for historical latency in runner scoring (0.0 to 1.0).
    /// Higher values prefer runners with lower average latency.
    #[serde(default = "default_latency_weight")]
    pub latency_weight: f64,
    /// Enable speculative wake (wake multiple machines in parallel).
    #[serde(default)]
    pub speculative_wake_enabled: bool,
    /// Speculative wake targets by model class.
    /// Key is the class name, value is list of machine types to wake.
    #[serde(default)]
    pub speculative_wake_targets: HashMap<String, Vec<String>>,
}

fn default_queue_weight() -> f64 { 0.5 }
fn default_latency_weight() -> f64 { 0.3 }
fn default_batch_timeout_ms() -> u64 { 50 }
fn default_min_batch_size() -> u32 { 1 }

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            class_preferences: HashMap::new(),
            queue_weight: default_queue_weight(),
            latency_weight: default_latency_weight(),
            speculative_wake_enabled: false,
            speculative_wake_targets: HashMap::new(),
        }
    }
}

// Defaults
fn default_host() -> String { "0.0.0.0".to_string() }
fn default_port() -> u16 { 8080 }
fn default_ollama_base_url() -> String { "http://localhost:11434".to_string() }
fn default_ollama_model() -> String { "gpt-oss:20b".to_string() }
fn default_database_url() -> String { "sqlite:./data/audit.db".to_string() }
fn default_log_level() -> String { "info".to_string() }
fn default_cors_origins() -> String { "*".to_string() }
fn default_language_model_path() -> String { "/data/lid.176.ftz".to_string() }
fn default_gateway_auth_token() -> String { "change-me-in-production".to_string() }
fn default_runner_timeout() -> u64 { 90 }
fn default_wake_timeout() -> u64 { 90 }
fn default_wol_broadcast() -> String { "255.255.255.255".to_string() }
fn default_wol_port() -> u16 { 9 }

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_base_url(),
            model: default_ollama_model(),
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self { url: default_database_url() }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self { level: default_log_level() }
    }
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self { origins: default_cors_origins() }
    }
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self { model_path: default_language_model_path() }
    }
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auth_token: default_gateway_auth_token(),
            runner_timeout_secs: default_runner_timeout(),
            idle_manager_url: None,
            wake_timeout_secs: default_wake_timeout(),
            auto_wake_enabled: false,
            batching_enabled: false,
            batch_timeout_ms: default_batch_timeout_ms(),
            min_batch_size: default_min_batch_size(),
        }
    }
}

impl Default for WolConfig {
    fn default() -> Self {
        Self {
            broadcast_address: default_wol_broadcast(),
            port: default_wol_port(),
            bouncer_url: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Configuration error: {0}")]
    LoadError(String),
}

impl From<ConfigCrateError> for ConfigError {
    fn from(err: ConfigCrateError) -> Self {
        ConfigError::LoadError(err.to_string())
    }
}

impl Config {
    /// Load configuration from config.toml (if exists) and environment variables.
    /// Environment variables override file settings.
    /// Env var format: SIMPLEAI__SECTION__KEY (e.g., SIMPLEAI__OLLAMA__BASE_URL)
    pub fn load() -> Result<Self, ConfigError> {
        let config = ConfigLoader::builder()
            // Start with defaults
            .set_default("host", default_host())?
            .set_default("port", default_port() as i64)?
            .set_default("ollama.base_url", default_ollama_base_url())?
            .set_default("ollama.model", default_ollama_model())?
            .set_default("database.url", default_database_url())?
            .set_default("logging.level", default_log_level())?
            .set_default("cors.origins", default_cors_origins())?
            .set_default("language.model_path", default_language_model_path())?
            .set_default("gateway.enabled", false)?
            .set_default("gateway.auth_token", default_gateway_auth_token())?
            .set_default("gateway.runner_timeout_secs", default_runner_timeout() as i64)?
            .set_default("gateway.wake_timeout_secs", default_wake_timeout() as i64)?
            .set_default("gateway.auto_wake_enabled", false)?
            .set_default("gateway.batching_enabled", false)?
            .set_default("gateway.batch_timeout_ms", default_batch_timeout_ms() as i64)?
            .set_default("gateway.min_batch_size", default_min_batch_size() as i64)?
            .set_default("wol.broadcast_address", default_wol_broadcast())?
            .set_default("wol.port", default_wol_port() as i64)?
            .set_default("routing.queue_weight", default_queue_weight())?
            .set_default("routing.latency_weight", default_latency_weight())?
            .set_default("routing.speculative_wake_enabled", false)?
            // Load from config.toml if it exists
            .add_source(File::with_name("config").required(false))
            // Override with environment variables (SIMPLEAI__KEY format)
            .add_source(
                Environment::with_prefix("SIMPLEAI")
                    .separator("__")
                    .try_parsing(true)
            )
            .build()?;

        config.try_deserialize().map_err(ConfigError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_host() {
        assert_eq!(default_host(), "0.0.0.0");
    }

    #[test]
    fn test_default_port() {
        assert_eq!(default_port(), 8080);
    }

    #[test]
    fn test_default_ollama_base_url() {
        assert_eq!(default_ollama_base_url(), "http://localhost:11434");
    }

    #[test]
    fn test_default_ollama_model() {
        assert_eq!(default_ollama_model(), "gpt-oss:20b");
    }

    #[test]
    fn test_default_database_url() {
        assert_eq!(default_database_url(), "sqlite:./data/audit.db");
    }

    #[test]
    fn test_default_log_level() {
        assert_eq!(default_log_level(), "info");
    }

    #[test]
    fn test_default_cors_origins() {
        assert_eq!(default_cors_origins(), "*");
    }

    #[test]
    fn test_default_language_model_path() {
        assert_eq!(default_language_model_path(), "/data/lid.176.ftz");
    }

    #[test]
    fn test_oidc_config_requires_issuer() {
        let config = OidcConfig {
            issuer: "".to_string(),
            audience: "test".to_string(),
            role_claim_path: default_role_claim_path(),
            admin_role: default_admin_role(),
            admin_users: vec![],
        };
        assert!(config.issuer.is_empty());
    }

    #[test]
    fn test_oidc_config_with_values() {
        let config = OidcConfig {
            issuer: "https://auth.example.com".to_string(),
            audience: "my-app".to_string(),
            role_claim_path: default_role_claim_path(),
            admin_role: default_admin_role(),
            admin_users: vec![],
        };
        assert_eq!(config.issuer, "https://auth.example.com");
        assert_eq!(config.audience, "my-app");
    }

    #[test]
    fn test_oidc_config_defaults() {
        assert_eq!(default_role_claim_path(), "roles");
        assert_eq!(default_admin_role(), "admin");
    }

    #[test]
    fn test_oidc_config_with_admin_users() {
        let config = OidcConfig {
            issuer: "https://auth.example.com".to_string(),
            audience: "my-app".to_string(),
            role_claim_path: "realm_access.roles".to_string(),
            admin_role: "super-admin".to_string(),
            admin_users: vec!["user-1".to_string(), "user-2".to_string()],
        };
        assert_eq!(config.role_claim_path, "realm_access.roles");
        assert_eq!(config.admin_role, "super-admin");
        assert_eq!(config.admin_users.len(), 2);
    }

    #[test]
    fn test_ollama_config_defaults() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "gpt-oss:20b");
    }

    #[test]
    fn test_database_config_defaults() {
        let config = DatabaseConfig::default();
        assert_eq!(config.url, "sqlite:./data/audit.db");
    }

    #[test]
    fn test_logging_config_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, "info");
    }

    #[test]
    fn test_cors_config_defaults() {
        let config = CorsConfig::default();
        assert_eq!(config.origins, "*");
    }

    #[test]
    fn test_language_config_defaults() {
        let config = LanguageConfig::default();
        assert_eq!(config.model_path, "/data/lid.176.ftz");
    }

    #[test]
    fn test_config_error_load_error() {
        let error = ConfigError::LoadError("test error".to_string());
        assert!(error.to_string().contains("Configuration error"));
    }

    #[test]
    fn test_config_error_from_config_error() {
        let config_err = ConfigCrateError::NotFound("file.toml".to_string());
        let error: ConfigError = config_err.into();
        assert!(error.to_string().contains("Configuration error"));
    }

    #[test]
    fn test_gateway_config_defaults() {
        let config = GatewayConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.auth_token, "change-me-in-production");
        assert_eq!(config.runner_timeout_secs, 90);
        assert_eq!(config.wake_timeout_secs, 90);
        assert!(!config.auto_wake_enabled);
        assert!(config.idle_manager_url.is_none());
        assert!(!config.batching_enabled);
        assert_eq!(config.batch_timeout_ms, 50);
        assert_eq!(config.min_batch_size, 1);
    }

    #[test]
    fn test_routing_config_defaults() {
        let config = RoutingConfig::default();
        assert!(config.class_preferences.is_empty());
        assert!((config.queue_weight - 0.5).abs() < f64::EPSILON);
        assert!((config.latency_weight - 0.3).abs() < f64::EPSILON);
        assert!(!config.speculative_wake_enabled);
        assert!(config.speculative_wake_targets.is_empty());
    }

    #[test]
    fn test_routing_config_with_preferences() {
        let mut prefs = std::collections::HashMap::new();
        prefs.insert("fast".to_string(), vec!["gpu-server".to_string(), "halo".to_string()]);
        prefs.insert("big".to_string(), vec!["halo".to_string(), "gpu-server".to_string()]);

        let config = RoutingConfig {
            class_preferences: prefs.clone(),
            queue_weight: 0.4,
            latency_weight: 0.2,
            speculative_wake_enabled: true,
            speculative_wake_targets: prefs,
        };

        assert_eq!(config.class_preferences.get("fast").unwrap()[0], "gpu-server");
        assert_eq!(config.class_preferences.get("big").unwrap()[0], "halo");
        assert!(config.speculative_wake_enabled);
    }
}
