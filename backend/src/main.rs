mod config;
mod routes;
mod auth;
mod llm;
mod audit;
mod models;

use std::sync::Arc;
use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use fasttext::FastText;

use crate::config::Config;
use crate::auth::JwksClient;
use crate::llm::OllamaClient;
use crate::audit::AuditLogger;

/// Shared application state.
pub struct AppState {
    pub config: Config,
    pub jwks_client: JwksClient,
    pub ollama_client: OllamaClient,
    pub audit_logger: AuditLogger,
    pub lang_detector: Mutex<FastText>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::load()?;

    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| config.logging.level.clone().into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting SimpleAI API Gateway");

    // Initialize components
    let jwks_client = JwksClient::new(&config.oidc.issuer).await?;
    let ollama_client = OllamaClient::new(&config.ollama.base_url, &config.ollama.model);
    let audit_logger = AuditLogger::new(&config.database.url)?;

    // Load FastText language detection model
    let mut lang_detector = FastText::default();
    lang_detector.load_model(&config.language.model_path)
        .map_err(|e| format!("Failed to load language model: {}", e))?;
    tracing::info!("Loaded language detection model from {}", config.language.model_path);

    let state = Arc::new(AppState {
        config: config.clone(),
        jwks_client,
        ollama_client,
        audit_logger,
        lang_detector: Mutex::new(lang_detector),
    });

    // Build CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .merge(routes::health::router())
        .merge(routes::chat::router(state.clone()))
        .merge(routes::language::router(state.clone()))
        .nest("/admin", routes::admin::router(state.clone()))
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Listening on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
