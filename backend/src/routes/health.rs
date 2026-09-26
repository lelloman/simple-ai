use serde::Serialize;
use simple_server::web::http::{header, StatusCode};
use simple_server::web::response::{IntoResponse, Response};
use simple_server::web::{
    routing::{get, get_service},
    Json, Router,
};

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
}

fn health(
    _: Result<(), simple_server::health::CheckFailure<std::convert::Infallible>>,
) -> Response {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
    .into_response()
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
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        body,
    )
        .into_response()
}

pub fn router() -> Router {
    Router::new()
        .route(
            "/health",
            get_service(simple_server::health::Probe::liveness().endpoint(health)),
        )
        .route("/metrics", get(metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use simple_server::web::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_preserves_get_head_and_method_contract() {
        let app = router();
        let get = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(get.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(get.into_body(), usize::MAX).await.unwrap()).unwrap();
        assert_eq!(
            body,
            serde_json::json!({"status":"ok", "version":env!("CARGO_PKG_VERSION")})
        );
        let head = app
            .clone()
            .oneshot(Request::head("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(head.status(), StatusCode::OK);
        assert!(to_bytes(head.into_body(), usize::MAX)
            .await
            .unwrap()
            .is_empty());
        let post = app
            .oneshot(Request::post("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(post.status(), StatusCode::METHOD_NOT_ALLOWED);
    }
}
