use simple_server::web::{
    extract::Request,
    middleware::Next,
    response::Response,
    tracing::{Observer, Outcome, Phase, ResponseInfo, TracingObserver},
};

/// Observe safe route templates and the complete response body lifecycle.
pub async fn request_logger(request: Request, next: Next) -> Response {
    simple_server::web::tracing::trace_with_observer(request, RequestObserver, |request| {
        next.run(request)
    })
    .await
}
struct RequestObserver;
impl Observer for RequestObserver {
    fn on_response(
        &mut self,
        span: &tracing::Span,
        response: &ResponseInfo<'_>,
        latency: std::time::Duration,
    ) {
        let status = response.status().as_u16();
        let header_latency_ms = latency.as_secs_f64() * 1000.0;
        tracing::info!(parent: span, status, header_latency_ms, "http.response_headers");
    }
    fn on_finish(
        &mut self,
        span: &tracing::Span,
        outcome: Outcome,
        phase: Phase,
        duration: std::time::Duration,
    ) {
        TracingObserver.on_finish(span, outcome, phase, duration);
    }
}
