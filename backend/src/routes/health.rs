use axum::{routing::get, Json, Router};
use axum::response::{IntoResponse, Response};
use axum::http::{header, StatusCode};
use serde::Serialize;

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

async fn metrics() -> Response {
    let version = env!("CARGO_PKG_VERSION");
    let body = format!(
        "# HELP simpleai_up Whether the service is up\n\
         # TYPE simpleai_up gauge\n\
         simpleai_up 1\n\
         # HELP simpleai_info Service information\n\
         # TYPE simpleai_info gauge\n\
         simpleai_info{{version=\"{}\"}} 1\n",
        version
    );
    (StatusCode::OK, [(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response()
}

pub fn router() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
}
