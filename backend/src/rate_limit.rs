use simple_server::axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use simple_server::rate_limit::{KeyedLimiter, Quota, StoreConfig};
use std::{num::NonZeroU32, sync::Arc, time::Duration};

/// Shared GCRA budgets; identity and response policy remain application-owned.
pub struct RateLimiter {
    limiter: KeyedLimiter<String>,
}

impl RateLimiter {
    pub fn new(rpm: u32) -> Self {
        let burst = NonZeroU32::new(rpm).expect("rate_limit_rpm must be > 0");
        let quota = Quota::replenishing(Duration::from_secs(60) / rpm, burst)
            .expect("rate_limit_rpm produces a positive refill interval");
        Self {
            // Preserve the existing unbounded per-IP map. Adding a capacity
            // policy would change admission for previously accepted identities.
            limiter: KeyedLimiter::new(quota, StoreConfig::unbounded()),
        }
    }

    async fn check(&self, key: &str) -> Result<(), u64> {
        self.limiter
            .check(key.to_owned(), NonZeroU32::new(1).unwrap())
            .map_err(|rejection| {
                rejection
                    .retry_after
                    .expect("unit GCRA check has a retry duration")
                    .as_secs()
                    .max(1)
            })
    }
}

/// Extract client IP from request headers or connection info.
fn extract_ip(request: &Request) -> String {
    // Check X-Forwarded-For first
    if let Some(forwarded) = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(first_ip) = forwarded.split(',').next() {
            return first_ip.trim().to_string();
        }
    }
    // Check X-Real-IP
    if let Some(real_ip) = request
        .headers()
        .get("x-real-ip")
        .and_then(|v| v.to_str().ok())
    {
        return real_ip.to_string();
    }
    // Fall back to connection info
    request
        .extensions()
        .get::<simple_server::axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Axum middleware that enforces per-IP rate limiting.
pub async fn rate_limit_middleware(
    simple_server::axum::extract::State(limiter): simple_server::axum::extract::State<
        Arc<RateLimiter>,
    >,
    request: Request,
    next: Next,
) -> Response {
    let ip = extract_ip(&request);

    match limiter.check(&ip).await {
        Ok(()) => next.run(request).await,
        Err(retry_after) => {
            tracing::warn!(ip = %ip, retry_after_secs = retry_after, "Rate limit exceeded");
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("retry-after", retry_after.to_string())],
                "Rate limit exceeded",
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(60);
        // Should allow the first request
        assert!(limiter.check("127.0.0.1").await.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_over_limit() {
        // 1 request per minute
        let limiter = RateLimiter::new(1);
        // First request should succeed
        assert!(limiter.check("127.0.0.1").await.is_ok());
        // Second request should be rate limited
        assert!(limiter.check("127.0.0.1").await.is_err());
    }

    #[tokio::test]
    async fn test_rate_limiter_separate_keys() {
        let limiter = RateLimiter::new(1);
        assert!(limiter.check("127.0.0.1").await.is_ok());
        // Different IP should still be allowed
        assert!(limiter.check("192.168.1.1").await.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiter_returns_retry_after() {
        let limiter = RateLimiter::new(1);
        limiter.check("127.0.0.1").await.ok();
        let err = limiter.check("127.0.0.1").await.unwrap_err();
        assert!(err >= 1);
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use simple_server::axum::{
        body::Body,
        extract::ConnectInfo,
        http::{HeaderValue, Request},
        middleware,
        routing::post,
        Router,
    };
    use std::net::SocketAddr;

    #[test]
    fn client_key_precedence_and_unknown_are_unchanged() {
        let peer: SocketAddr = "192.0.2.1:1234".parse().unwrap();
        let mut request = Request::new(Body::empty());
        assert_eq!(extract_ip(&request), "unknown");
        request.extensions_mut().insert(ConnectInfo(peer));
        assert_eq!(extract_ip(&request), "192.0.2.1");
        request
            .headers_mut()
            .insert("x-real-ip", HeaderValue::from_static("not-normalized"));
        assert_eq!(extract_ip(&request), "not-normalized");
        request.headers_mut().insert(
            "x-forwarded-for",
            HeaderValue::from_static(" 198.51.100.1, 192.0.2.5"),
        );
        request
            .headers_mut()
            .append("x-forwarded-for", HeaderValue::from_static("other"));
        assert_eq!(extract_ip(&request), "198.51.100.1");
        request
            .headers_mut()
            .insert("x-forwarded-for", HeaderValue::from_bytes(b"\xff").unwrap());
        assert_eq!(extract_ip(&request), "not-normalized");
        request
            .headers_mut()
            .insert("x-forwarded-for", HeaderValue::from_static(""));
        assert_eq!(extract_ip(&request), "");
    }

    #[tokio::test]
    async fn real_http_rate_contract_preserves_body_keys_and_public_routes() {
        let app = Router::new()
            .route("/v1/test", post(|body: String| async move { body }))
            .layer(middleware::from_fn_with_state(
                Arc::new(RateLimiter::new(1)),
                rate_limit_middleware,
            ))
            .route("/public", post(|| async { "public" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task =
            tokio::spawn(async move { simple_server::axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let url = format!("http://{addr}/v1/test");
        let ok = client
            .post(&url)
            .header("x-forwarded-for", "a")
            .body("stream-preserved")
            .send()
            .await
            .unwrap();
        assert_eq!(ok.status(), 200);
        assert_eq!(ok.text().await.unwrap(), "stream-preserved");
        let denied = client
            .post(&url)
            .header("x-forwarded-for", "a")
            .send()
            .await
            .unwrap();
        assert_eq!(denied.status(), 429);
        assert!((1..=60).contains(
            &denied.headers()["retry-after"]
                .to_str()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        ));
        assert_eq!(denied.text().await.unwrap(), "Rate limit exceeded");
        assert_eq!(
            client
                .post(&url)
                .header("x-forwarded-for", "b")
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        assert_eq!(client.post(&url).send().await.unwrap().status(), 200);
        assert_eq!(client.post(&url).send().await.unwrap().status(), 429);
        assert_eq!(
            client
                .post(format!("http://{addr}/public"))
                .header("x-forwarded-for", "a")
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        task.abort();
        let _ = task.await;
    }
}
