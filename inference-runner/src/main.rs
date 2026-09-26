//! Simple AI Runner - abstracts local inference engines and exposes OpenAI-compatible API.
mod logging_setup;

use std::env;
use std::sync::Arc;

use simple_server::web::Router;
use tracing_subscriber::EnvFilter;

mod api;
mod capability;
mod config;
mod engine;
mod error;
mod gateway;
mod ocr;
mod state;

use config::Config;
use engine::{
    AudioEmbeddingEngine, ClassificationEngine, EngineRegistry, ExtractionEngine, LlamaCppEngine,
    OllamaEngine, TtsEngine, VllmEngine,
};
use gateway::{GatewayClient, StatusCollector};
use ocr::{CliOcrProvider, OcrProvider};
use state::AppState;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const GIT_HASH: &str = env!("GIT_HASH");

fn print_version() {
    println!("simple-ai-runner {} ({})", VERSION, GIT_HASH);
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Service failed: {error}");
        // A timed-out blocking operation must not extend runtime teardown forever.
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Handle --version / -V
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        print_version();
        return Ok(());
    }

    // Initialize tracing
    logging_setup::init(
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        false,
        false,
    )
    .map_err(|error| -> Box<dyn std::error::Error> { error })?;

    use simple_server::lifecycle::{Lifecycle, ShutdownOptions, Signals};
    let signals = Signals::install()?;
    let mut lifecycle = Lifecycle::new(ShutdownOptions {
        grace_period: std::time::Duration::from_secs(30),
    });

    // Load configuration
    let config = Config::load().map_err(|e| {
        format!(
            "Failed to load configuration: {}. \
             Make sure config.toml exists or set RUNNER__RUNNER__ID and RUNNER__RUNNER__NAME environment variables.",
            e
        )
    })?;
    tracing::info!(
        "Starting simple-ai-runner: {} ({})",
        config.runner.name,
        config.runner.id
    );

    // Create engine registry
    let registry = Arc::new(EngineRegistry::new());
    registry
        .configure_routes(config.model_routes.clone())
        .await
        .map_err(|e| format!("Invalid model_routes configuration: {e}"))?;
    registry
        .set_engine_resources(config.engine_resources.clone())
        .await;
    let ocr_provider = if config.ocr.enabled {
        match CliOcrProvider::new(config.ocr.clone()) {
            Ok(provider) => {
                let provider = Arc::new(provider);
                match provider.health_check().await {
                    Ok(info) => {
                        tracing::info!(
                            "Registered OCR provider {} with modes {:?}",
                            info.provider,
                            info.modes
                        );
                        Some(provider as Arc<dyn ocr::OcrProvider>)
                    }
                    Err(e) => {
                        tracing::warn!("OCR enabled but health check failed: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!("OCR enabled but provider config is invalid: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Register enabled engines
    if let Some(ref ollama_config) = config.engines.ollama {
        if ollama_config.enabled {
            let engine = Arc::new(OllamaEngine::with_batch_size(
                &ollama_config.base_url,
                ollama_config.batch_size,
            ));
            registry.register(engine).await;
            tracing::info!(
                "Registered Ollama engine at {} (batch_size={})",
                ollama_config.base_url,
                ollama_config.batch_size
            );
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

    if let Some(ref audio_config) = config.engines.audio_embeddings {
        if audio_config.enabled {
            let engine = Arc::new(AudioEmbeddingEngine::new(audio_config.clone()));
            registry.register(engine).await;
            tracing::info!(
                "Registered audio embedding engine with {} models",
                audio_config.models.len()
            );
        }
    }

    if let Some(ref tts_config) = config.engines.tts {
        if tts_config.enabled {
            let engine = Arc::new(TtsEngine::new(tts_config.clone()));
            registry.register(engine).await;
            tracing::info!(
                "Registered TTS engine with {} models",
                tts_config.models.len()
            );
        }
    }

    if config.engines.extraction.enabled {
        let engine = Arc::new(ExtractionEngine::new(config.engines.extraction.clone())?);
        registry.register(engine).await;
        tracing::info!("Registered information extraction engine");
    }

    if let Some(ref classification_config) = config.engines.classification {
        if classification_config.enabled {
            let engine = Arc::new(ClassificationEngine::new(classification_config.clone()));
            registry.register(engine).await;
            tracing::info!(
                "Registered classification engine with {} models",
                classification_config.models.len()
            );
        }
    }

    if let Some(ref vllm_config) = config.engines.vllm {
        if vllm_config.enabled {
            let engine = Arc::new(VllmEngine::new(vllm_config.clone())?);
            registry.register(engine).await;
            tracing::info!(
                "Registered managed vLLM engine at {} with {} models",
                vllm_config.base_url,
                vllm_config.models.len()
            );
        }
    }

    // Create shared state
    let state = Arc::new(AppState::new(
        config.clone(),
        registry.clone(),
        ocr_provider.clone(),
    ));

    // Start gateway client if configured
    if let Some(ref gateway_config) = config.gateway {
        let status_collector = Arc::new(StatusCollector::new(
            config.clone(),
            registry.clone(),
            ocr_provider.is_some(),
        ));
        let client = GatewayClient::new(
            gateway_config.clone(),
            config.runner.id.clone(),
            config.runner.name.clone(),
            config.runner.machine_type.clone(),
            config.runner.mac_address.clone(),
            config.api.port,
            status_collector,
            registry,
        );

        let stop = lifecycle.shutdown();
        lifecycle.service("gateway", async move {
            client.run_until_shutdown(stop).await;
            Ok::<(), std::io::Error>(())
        })?;
        tracing::info!(
            "Gateway client started, connecting to {}",
            gateway_config.ws_url
        );
    } else {
        tracing::info!("No gateway configured, running in standalone mode");
    }

    // Build router
    let app = Router::new()
        .nest("/v1", api::router())
        .route(
            "/health",
            simple_server::web::routing::get(api::health::health),
        )
        .layer(cors_policy())
        .with_state(state);

    // Start server
    let addr = format!("{}:{}", config.api.host, config.api.port);
    tracing::info!("Listening on {}", addr);

    let listener = simple_server::http::bind(&addr).await?;
    lifecycle.service(
        "http",
        simple_server::web::serve(listener, app, lifecycle.shutdown()),
    )?;
    let report = lifecycle
        .run(signals.wait(), async { Ok::<(), std::io::Error>(()) })
        .await?;
    tracing::info!(?report, "Graceful shutdown complete");

    Ok(())
}

fn cors_policy() -> simple_server::cors::CorsLayer {
    simple_server::cors::CorsConfig::default()
        .allow_any_origin()
        .allow_any_method()
        .allow_any_header()
        .expose_any_header()
        .build()
        .expect("static wildcard CORS policy without credentials")
}

#[cfg(test)]
mod cors_tests {
    use super::cors_policy;
    use simple_server::web::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn production_cors_policy_preserves_preflights_and_error_responses() {
        let app = Router::new()
            .route(
                "/protected",
                get(|| async { (StatusCode::UNAUTHORIZED, "denied") }),
            )
            .layer(cors_policy());
        for origin in ["https://app.example", "https://other.example", "null"] {
            for (method, status, body) in [
                ("GET", StatusCode::UNAUTHORIZED, "denied"),
                ("OPTIONS", StatusCode::OK, ""),
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method(method)
                            .uri("/protected")
                            .header("origin", origin)
                            .header("access-control-request-method", "POST")
                            .header(
                                "access-control-request-headers",
                                "authorization,content-type",
                            )
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), status);
                assert_eq!(response.headers()["access-control-allow-origin"], "*");
                assert!(!response
                    .headers()
                    .contains_key("access-control-allow-credentials"));
                assert_eq!(
                    response.headers()["vary"],
                    "origin, access-control-request-method, access-control-request-headers"
                );
                if method == "OPTIONS" {
                    assert_eq!(response.headers()["access-control-allow-methods"], "*");
                    assert_eq!(response.headers()["access-control-allow-headers"], "*");
                    assert!(!response
                        .headers()
                        .contains_key("access-control-expose-headers"));
                } else {
                    assert_eq!(response.headers()["access-control-expose-headers"], "*");
                }
                assert_eq!(
                    to_bytes(response.into_body(), usize::MAX)
                        .await
                        .unwrap()
                        .as_ref(),
                    body.as_bytes()
                );
            }
        }
    }
}
