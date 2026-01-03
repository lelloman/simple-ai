use std::env;

/// Application configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    /// Server host (default: 0.0.0.0)
    pub host: String,
    /// Server port (default: 8080)
    pub port: u16,
    /// Ollama base URL (default: http://localhost:11434)
    pub ollama_base_url: String,
    /// Default Ollama model (default: llama3.2)
    pub ollama_model: String,
    /// OIDC issuer URL for JWT validation
    pub oidc_issuer: String,
    /// OIDC audience (client ID)
    pub oidc_audience: String,
    /// SQLite database URL
    pub database_url: String,
    /// Log level (default: info)
    pub log_level: String,
    /// CORS allowed origins (comma-separated, default: *)
    pub cors_origins: String,
}

impl Config {
    /// Load configuration from environment variables.
    pub fn from_env() -> Result<Self, ConfigError> {
        Ok(Config {
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidPort)?,
            ollama_base_url: env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            ollama_model: env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string()),
            oidc_issuer: env::var("OIDC_ISSUER")
                .map_err(|_| ConfigError::MissingEnvVar("OIDC_ISSUER"))?,
            oidc_audience: env::var("OIDC_AUDIENCE")
                .map_err(|_| ConfigError::MissingEnvVar("OIDC_AUDIENCE"))?,
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "sqlite:./data/audit.db".to_string()),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            cors_origins: env::var("CORS_ORIGINS").unwrap_or_else(|_| "*".to_string()),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing required environment variable: {0}")]
    MissingEnvVar(&'static str),
    #[error("Invalid port number")]
    InvalidPort,
}
