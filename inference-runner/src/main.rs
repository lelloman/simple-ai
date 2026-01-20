//! Inference Runner - abstracts local inference engines and exposes OpenAI-compatible API.

use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod api;
mod capability;
mod config;
mod engine;
mod error;
mod state;

use config::Config;
use engine::{EngineRegistry, OllamaEngine};
use state::AppState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::load().map_err(|e| {
        format!(
            "Failed to load configuration: {}. \
             Make sure config.toml exists or set RUNNER__RUNNER__ID and RUNNER__RUNNER__NAME environment variables.",
            e
        )
    })?;
    tracing::info!(
        "Starting inference-runner: {} ({})",
        config.runner.name,
        config.runner.id
    );

    // Create engine registry
    let registry = Arc::new(EngineRegistry::new());

    // Register enabled engines
    if let Some(ref ollama_config) = config.engines.ollama {
        if ollama_config.enabled {
            let engine = Arc::new(OllamaEngine::new(&ollama_config.base_url));
            registry.register(engine).await;
            tracing::info!("Registered Ollama engine at {}", ollama_config.base_url);
        }
    }

    // Create shared state
    let state = Arc::new(AppState::new(config.clone(), registry));

    // Build router
    let app = Router::new()
        .nest("/v1", api::router())
        .route("/health", axum::routing::get(api::health::health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = format!("{}:{}", config.api.host, config.api.port);
    tracing::info!("Listening on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
