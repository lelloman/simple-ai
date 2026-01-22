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
            .set_default("wol.broadcast_address", default_wol_broadcast())?
            .set_default("wol.port", default_wol_port() as i64)?
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
        };
        assert!(config.issuer.is_empty());
    }

    #[test]
    fn test_oidc_config_with_values() {
        let config = OidcConfig {
            issuer: "https://auth.example.com".to_string(),
            audience: "my-app".to_string(),
        };
        assert_eq!(config.issuer, "https://auth.example.com");
        assert_eq!(config.audience, "my-app");
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
    }
}
