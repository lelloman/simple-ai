//! Inference Runner - abstracts local inference engines and exposes OpenAI-compatible API.

use std::env;
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
mod gateway;
mod state;

use config::Config;
use engine::{EngineRegistry, LlamaCppEngine, OllamaEngine};
use gateway::{GatewayClient, StatusCollector};
use state::AppState;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const GIT_HASH: &str = env!("GIT_HASH");

fn print_version() {
    println!("inference-runner {} ({})", VERSION, GIT_HASH);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Handle --version / -V
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        print_version();
        return Ok(());
    }

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

    if let Some(ref llama_config) = config.engines.llama_cpp {
        if llama_config.enabled {
            let engine = Arc::new(LlamaCppEngine::new(llama_config.clone()));
            registry.register(engine).await;
            tracing::info!(
                "Registered llama.cpp engine: model_dir={}, binary={}",
                llama_config.model_dir,
                llama_config.server_binary
            );
        }
    }

    // Create shared state
    let state = Arc::new(AppState::new(config.clone(), registry.clone()));

    // Start gateway client if configured
    if let Some(ref gateway_config) = config.gateway {
        let status_collector = Arc::new(StatusCollector::new(config.clone(), registry.clone()));
        let client = GatewayClient::new(
            gateway_config.clone(),
            config.runner.id.clone(),
            config.runner.name.clone(),
            config.runner.machine_type.clone(),
            config.api.port,
            status_collector,
            registry,
        );

        // Spawn gateway client task
        tokio::spawn(async move {
            client.run().await;
        });
        tracing::info!("Gateway client started, connecting to {}", gateway_config.ws_url);
    } else {
        tracing::info!("No gateway configured, running in standalone mode");
    }

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
