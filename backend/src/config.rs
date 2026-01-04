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

// Defaults
fn default_host() -> String { "0.0.0.0".to_string() }
fn default_port() -> u16 { 8080 }
fn default_ollama_base_url() -> String { "http://localhost:11434".to_string() }
fn default_ollama_model() -> String { "llama3.2".to_string() }
fn default_database_url() -> String { "sqlite:./data/audit.db".to_string() }
fn default_log_level() -> String { "info".to_string() }
fn default_cors_origins() -> String { "*".to_string() }
fn default_language_model_path() -> String { "/data/lid.176.ftz".to_string() }

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
