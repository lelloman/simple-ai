use std::sync::Arc;
use simple_ai_backend::config::Config;
use simple_ai_backend::auth::JwksClient;
use simple_ai_backend::llm::OllamaClient;
use simple_ai_backend::audit::AuditLogger;
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

    let jwks_client = JwksClient::new(&config.oidc.issuer).await?;
    let ollama_client = OllamaClient::new(&config.ollama.base_url, &config.ollama.model);
    let audit_logger = AuditLogger::new(&config.database.url)?;

    let mut lang_detector = fasttext::FastText::default();
    lang_detector.load_model(&config.language.model_path)
        .map_err(|e| format!("Failed to load language model: {}", e))?;
    tracing::info!("Loaded language detection model from {}", config.language.model_path);

    let state = Arc::new(AppState {
        config: config.clone(),
        jwks_client,
        ollama_client,
        audit_logger,
        lang_detector: tokio::sync::Mutex::new(lang_detector),
    });

    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let app = simple_ai_backend::routes::health::router()
        .merge(simple_ai_backend::routes::chat::router(state.clone()))
        .merge(simple_ai_backend::routes::language::router(state.clone()))
        .nest("/admin", simple_ai_backend::routes::admin::router(state.clone()))
        .layer(cors)
        .layer(axum::middleware::from_fn(simple_ai_backend::logging::request_logger));

    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
