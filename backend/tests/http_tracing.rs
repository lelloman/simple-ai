use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);
impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Capture {
    fn install(&self) -> tracing::subscriber::DefaultGuard {
        let output = self.clone();
        tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .without_time()
                .with_ansi(false)
                .with_max_level(tracing::Level::TRACE)
                .with_writer(move || output.clone())
                .finish(),
        )
    }
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
    fn events(&self, name: &str) -> Vec<String> {
        self.text()
            .lines()
            .filter(|line| line.contains(name))
            .map(str::to_owned)
            .collect()
    }
}
use simple_server::web::{
    body::{to_bytes, Body, Bytes},
    http::{Request, Response},
    middleware,
    routing::get,
    Router,
};
use tower::ServiceExt;

#[tokio::test]
async fn adapter_preserves_all_status_info_policy_safe_labels_and_response_contract() {
    let output = Capture::default();
    let _guard = output.install();
    let app = Router::new()
        .route(
            "/probe/{status}",
            get(
                |simple_server::web::extract::Path(status): simple_server::web::extract::Path<
                    u16,
                >| async move {
                    Response::builder()
                        .status(status)
                        .header("x-preserved", "yes")
                        .body(Body::from("private-body"))
                        .unwrap()
                },
            ),
        )
        .layer(middleware::from_fn(
            simple_ai_backend::logging::request_logger,
        ));
    for status in [200, 404, 503] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/probe/{status}?private-query=secret"))
                    .header("authorization", "Bearer private-credential")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), status);
        assert_eq!(response.headers()["x-preserved"], "yes");
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await.unwrap(),
            "private-body"
        );
    }
    let headers = output.events("http.response_headers");
    assert_eq!(headers.len(), 3);
    for line in &headers {
        assert!(line.contains("INFO"), "{line}");
        assert!(line.contains("/probe/{status}"), "{line}");
    }
    assert_eq!(output.events("http.finished").len(), 3);
    for secret in ["private-query", "private-credential", "private-body"] {
        assert!(!output.text().contains(secret));
    }
}

#[tokio::test]
async fn streaming_stays_lazy_and_cancellation_is_not_reported_as_completion() {
    let output = Capture::default();
    let _guard = output.install();
    let app = Router::new()
        .route(
            "/stream",
            get(|| async {
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(futures_util::stream::once(async {
                        Ok::<_, io::Error>(Bytes::from_static(b"data: unchanged\n\n"))
                    })))
                    .unwrap()
            }),
        )
        .route(
            "/pending",
            get(|| async {
                Body::from_stream(futures_util::stream::pending::<Result<Bytes, io::Error>>())
            }),
        )
        .layer(middleware::from_fn(
            simple_ai_backend::logging::request_logger,
        ));
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/stream")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(output.events("http.response_headers").len(), 1);
    assert!(output.events("http.finished").is_empty());
    assert_eq!(
        to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        "data: unchanged\n\n"
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/pending")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    drop(response);
    let events = output.events("http.finished");
    assert_eq!(events.len(), 2);
    assert!(events[0].contains("outcome=\"complete\""));
    assert!(events[1].contains("outcome=\"cancelled\""));
}
