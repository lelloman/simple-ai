use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter as GovernorRateLimiter,
};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Per-IP rate limiter using the GCRA algorithm.
pub struct RateLimiter {
    limiters:
        Mutex<HashMap<String, Arc<GovernorRateLimiter<NotKeyed, InMemoryState, DefaultClock>>>>,
    quota: Quota,
}

impl RateLimiter {
    /// Create a new rate limiter with the given requests-per-minute limit.
    pub fn new(rpm: u32) -> Self {
        let per_minute = NonZeroU32::new(rpm).expect("rate_limit_rpm must be > 0");
        let quota = Quota::per_minute(per_minute);
        Self {
            limiters: Mutex::new(HashMap::new()),
            quota,
        }
    }

    /// Check if a request from the given key is allowed.
    /// Returns Ok(()) if allowed, Err(retry_after_secs) if rate limited.
    async fn check(&self, key: &str) -> Result<(), u64> {
        let mut limiters = self.limiters.lock().await;
        let limiter = limiters
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(GovernorRateLimiter::direct(self.quota)));

        match limiter.check() {
            Ok(()) => Ok(()),
            Err(not_until) => {
                let retry_after =
                    not_until.wait_time_from(governor::clock::Clock::now(&DefaultClock::default()));
                Err(retry_after.as_secs().max(1))
            }
        }
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
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Axum middleware that enforces per-IP rate limiting.
pub async fn rate_limit_middleware(
    axum::extract::State(limiter): axum::extract::State<Arc<RateLimiter>>,
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
