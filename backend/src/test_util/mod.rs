pub mod mock_ollama;

use jsonwebtoken::{encode, EncodingKey, Header, Algorithm};
use chrono::{Duration, Utc};
use crate::auth::AuthUser;
use crate::config::{Config, OllamaConfig, OidcConfig, DatabaseConfig, LoggingConfig, LanguageConfig, CorsConfig};
use tokio::sync::Mutex;
use fasttext::FastText;
use crate::AppState;
use crate::llm::OllamaClient;
use crate::audit::AuditLogger;
use crate::auth::JwksClient;

pub fn test_config() -> Config {
    Config {
        host: "127.0.0.1".to_string(),
        port: 8080,
        ollama: OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "test-model".to_string(),
        },
        oidc: OidcConfig {
            issuer: "https://test-issuer".to_string(),
            audience: "test-audience".to_string(),
        },
        database: DatabaseConfig {
            url: "sqlite:memory".to_string(),
        },
        logging: LoggingConfig {
            level: "debug".to_string(),
        },
        cors: CorsConfig { origins: "*".to_string() },
        language: LanguageConfig {
            model_path: "/tmp/test-lid.ftz".to_string(),
        },
    }
}

pub async fn create_test_state() -> AppState {
    let config = test_config();
    let jwks_client = JwksClient::new(&config.oidc.issuer).await.unwrap();
    let ollama_client = OllamaClient::new(&config.ollama.base_url, &config.ollama.model);
    let audit_logger = AuditLogger::new(&config.database.url).unwrap();
    let lang_detector = Mutex::new(FastText::default());

    AppState {
        config,
        jwks_client,
        ollama_client,
        audit_logger,
        lang_detector,
    }
}

#[derive(serde::Serialize)]
struct TestClaims {
    sub: String,
    email: Option<String>,
    roles: Vec<String>,
    aud: serde_json::Value,
    exp: u64,
    iat: u64,
}

pub fn generate_test_jwt(
    user_id: &str,
    email: Option<&str>,
    roles: Vec<&str>,
    kid: &str,
    signing_key: &EncodingKey,
) -> String {
    let now = Utc::now();
    let claims = TestClaims {
        sub: user_id.to_string(),
        email: email.map(String::from),
        roles: roles.iter().map(|s| s.to_string()).collect(),
        aud: serde_json::Value::String("test-audience".to_string()),
        exp: (now + Duration::hours(1)).timestamp() as u64,
        iat: now.timestamp() as u64,
    };

    let header = Header {
        alg: Algorithm::RS256,
        kid: Some(kid.to_string()),
        ..Default::default()
    };

    encode(&header, &claims, signing_key).expect("Failed to encode JWT")
}

pub fn generate_expired_jwt(
    user_id: &str,
    kid: &str,
    signing_key: &EncodingKey,
) -> String {
    let now = Utc::now();
    let claims = TestClaims {
        sub: user_id.to_string(),
        email: None,
        roles: vec![],
        aud: serde_json::Value::String("test-audience".to_string()),
        exp: (now - Duration::hours(1)).timestamp() as u64,
        iat: (now - Duration::hours(2)).timestamp() as u64,
    };

    let header = Header {
        alg: Algorithm::RS256,
        kid: Some(kid.to_string()),
        ..Default::default()
    };

    encode(&header, &claims, signing_key).expect("Failed to encode JWT")
}

pub fn test_auth_user(sub: &str, email: Option<&str>, roles: Vec<&str>) -> AuthUser {
    AuthUser {
        sub: sub.to_string(),
        email: email.map(String::from),
        roles: roles.iter().map(|s| s.to_string()).collect(),
    }
}

