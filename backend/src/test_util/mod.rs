pub mod mock_ollama;

use jsonwebtoken::{encode, EncodingKey, Header, Algorithm};
use chrono::{Duration, Utc};
use crate::auth::AuthUser;
use crate::config::{Config, OllamaConfig, OidcConfig, DatabaseConfig, LoggingConfig, LanguageConfig, CorsConfig, GatewayConfig, WolConfig, ModelsConfig, RoutingConfig};
use tokio::sync::Mutex;
use fasttext::FastText;
use std::sync::Arc;
use crate::AppState;
use crate::llm::OllamaClient;
use crate::audit::AuditLogger;
use crate::auth::JwksClient;
use crate::gateway::{InferenceRouter, RunnerRegistry};
use crate::wol::WakeService;

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
            role_claim_path: "roles".to_string(),
            admin_role: "admin".to_string(),
            admin_users: vec![],
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
        gateway: GatewayConfig::default(),
        wol: WolConfig::default(),
        models: ModelsConfig::default(),
        routing: RoutingConfig::default(),
    }
}

pub async fn create_test_state() -> AppState {
    let config = test_config();
    let jwks_client = JwksClient::new(&config.oidc).await.unwrap();
    let ollama_client = OllamaClient::new(&config.ollama.base_url, &config.ollama.model);
    let audit_logger = Arc::new(AuditLogger::new(&config.database.url).unwrap());
    let lang_detector = Mutex::new(FastText::default());

    let runner_registry = Arc::new(RunnerRegistry::new());
    let inference_router = Arc::new(InferenceRouter::new(
        runner_registry.clone(),
        config.models.clone(),
        config.routing.clone(),
        audit_logger.clone(),
    ));
    let wol_config = config.wol.clone();
    let wake_service = Arc::new(WakeService::new(
        runner_registry.clone(),
        audit_logger.clone(),
        config.gateway.clone(),
        config.wol.clone(),
        config.models.clone(),
        config.routing.clone(),
    ));

    let (request_events_tx, _) = tokio::sync::broadcast::channel(64);

    AppState {
        config,
        jwks_client,
        ollama_client,
        audit_logger,
        lang_detector,
        runner_registry,
        inference_router,
        wol_config,
        wake_service,
        request_events: request_events_tx,
        batch_queue: None,
        batch_dispatcher: None,
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
    AuthUser::new(
        sub.to_string(),
        email.map(String::from),
        roles.iter().map(|s| s.to_string()).collect(),
    )
}

