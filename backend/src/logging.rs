use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};
use std::time::Instant;

/// Middleware that logs HTTP requests at INFO level.
pub async fn request_logger(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let path = uri.path().to_string();

    let response = next.run(request).await;

    let status = response.status();
    let duration = start.elapsed();

    tracing::info!(
        method = %method,
        path = %path,
        status = %status.as_u16(),
        duration_ms = %duration.as_millis(),
        "HTTP request"
    );

    response
}
