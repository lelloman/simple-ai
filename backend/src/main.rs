use std::sync::Arc;
use axum::routing::get;
use axum::response::Html;
use simple_ai_backend::config::Config;
use simple_ai_backend::auth::JwksClient;
use simple_ai_backend::llm::OllamaClient;
use simple_ai_backend::audit::AuditLogger;
use simple_ai_backend::gateway::{
    ws_handler, BatchDispatcher, BatchQueue, BatchQueueConfig, InferenceRouter, RunnerRegistry, WsState,
};
use simple_ai_backend::wol::WakeService;
use simple_ai_backend::AppState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::load()?;

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| config.logging.level.clone().into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting SimpleAI API Gateway");

    let jwks_client = JwksClient::new(&config.oidc).await?;
    let ollama_client = OllamaClient::new(&config.ollama.base_url, &config.ollama.model);
    let audit_logger = Arc::new(AuditLogger::new(&config.database.url)?);

    let mut lang_detector = fasttext::FastText::default();
    lang_detector.load_model(&config.language.model_path)
        .map_err(|e| format!("Failed to load language model: {}", e))?;
    tracing::info!("Loaded language detection model from {}", config.language.model_path);

    // Initialize gateway components
    let runner_registry = Arc::new(RunnerRegistry::new());
    let inference_router = Arc::new(InferenceRouter::new(
        runner_registry.clone(),
        config.models.clone(),
        config.routing.clone(),
        audit_logger.clone(),
    ));

    // Initialize wake service for on-demand runner waking
    let wake_service = Arc::new(WakeService::new(
        runner_registry.clone(),
        audit_logger.clone(),
        config.gateway.clone(),
        config.wol.clone(),
        config.models.clone(),
        config.routing.clone(),
    ));

    if config.gateway.enabled {
        tracing::info!("Gateway mode enabled - will accept runner connections");
        tracing::info!(
            "Smart routing enabled (queue_weight: {}, latency_weight: {})",
            config.routing.queue_weight,
            config.routing.latency_weight
        );
        if config.gateway.auto_wake_enabled {
            tracing::info!(
                "Auto-wake enabled - will wake runners on demand (timeout: {}s)",
                config.gateway.wake_timeout_secs
            );
            if config.routing.speculative_wake_enabled {
                tracing::info!(
                    "Speculative wake enabled - targets: {:?}",
                    config.routing.speculative_wake_targets
                );
            }
        }
    } else {
        tracing::info!("Gateway mode disabled - using direct Ollama connection");
    }

    // Create broadcast channel for request events (admin dashboard)
    let (request_events_tx, _) = tokio::sync::broadcast::channel(64);

    // Initialize batch queue if batching is enabled
    let (batch_queue, batch_dispatcher) = if config.gateway.enabled && config.gateway.batching_enabled {
        let queue_config = BatchQueueConfig::new(
            config.gateway.batch_timeout_ms,
            config.gateway.min_batch_size,
        );
        let queue = Arc::new(BatchQueue::new(queue_config));

        // Create and spawn batch dispatcher (keep Arc for cache invalidation)
        let dispatcher = Arc::new(BatchDispatcher::new(queue.clone(), runner_registry.clone()));
        let dispatcher_clone = dispatcher.clone();
        tokio::spawn(async move {
            dispatcher_clone.run().await;
        });

        tracing::info!(
            "Request batching enabled (timeout={}ms, min_batch_size={})",
            config.gateway.batch_timeout_ms,
            config.gateway.min_batch_size
        );

        (Some(queue), Some(dispatcher))
    } else {
        (None, None)
    };

    let state = Arc::new(AppState {
        config: config.clone(),
        jwks_client,
        ollama_client,
        audit_logger: audit_logger.clone(),
        lang_detector: tokio::sync::Mutex::new(lang_detector),
        runner_registry: runner_registry.clone(),
        inference_router,
        wol_config: config.wol.clone(),
        wake_service,
        request_events: request_events_tx,
        batch_queue,
        batch_dispatcher,
    });

    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    // Create WebSocket state for runner connections
    let ws_state = Arc::new(WsState {
        registry: runner_registry.clone(),
        auth_token: config.gateway.auth_token.clone(),
        audit_logger: audit_logger.clone(),
        batch_dispatcher: state.batch_dispatcher.clone(),
    });

    // Static admin UI HTML
    const ADMIN_UI_HTML: &str = include_str!("../static/admin.html");

    let app = simple_ai_backend::routes::health::router()
        .nest("/v1",
            simple_ai_backend::routes::chat::router(state.clone())
                .merge(simple_ai_backend::routes::language::router(state.clone()))
                .merge(simple_ai_backend::routes::models::router(state.clone()))
        )
        .nest("/admin", simple_ai_backend::routes::admin::router(state.clone()))
        // Admin UI (public, auth handled in browser)
        .route("/admin-ui", get(|| async { Html(ADMIN_UI_HTML) }))
        // WebSocket endpoint for runner connections
        .route("/ws/runners", get(ws_handler).with_state(ws_state))
        .layer(cors)
        .layer(axum::middleware::from_fn(simple_ai_backend::logging::request_logger));

    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await?;

    Ok(())
}
