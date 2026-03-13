use std::net::SocketAddr;

use axum::http::{HeaderMap, StatusCode};

use crate::auth::AuthUser;
use crate::models::user::User;
use crate::AppState;

/// Extract client IP from headers (X-Forwarded-For, X-Real-IP) or connection info.
pub fn extract_client_ip(headers: &HeaderMap, addr: Option<SocketAddr>) -> Option<String> {
    if let Some(forwarded) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        if let Some(first_ip) = forwarded.split(',').next() {
            return Some(first_ip.trim().to_string());
        }
    }
    if let Some(real_ip) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
        return Some(real_ip.to_string());
    }
    addr.map(|a| a.ip().to_string())
}

/// Authenticate a request using API key or JWT.
pub async fn authenticate_request(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(AuthUser, User), (StatusCode, String)> {
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        if token.starts_with("sk-") {
            match state.audit_logger.validate_api_key(token) {
                Ok(Some((user_id, email))) => {
                    let user = state
                        .audit_logger
                        .find_or_create_user(&user_id, email.as_deref())
                        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                    if !user.is_enabled {
                        return Err((StatusCode::FORBIDDEN, "User is disabled".to_string()));
                    }
                    let auth_user = AuthUser::new(user_id, email, vec![]);
                    return Ok((auth_user, user));
                }
                Ok(None) => {
                    return Err((StatusCode::UNAUTHORIZED, "Invalid API key".to_string()));
                }
                Err(e) => {
                    return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
                }
            }
        }
    }

    let auth_user = state
        .jwks_client
        .authenticate(headers)
        .await
        .map_err(|e| (StatusCode::UNAUTHORIZED, e.to_string()))?;

    let user = state
        .audit_logger
        .find_or_create_user(&auth_user.sub, auth_user.email.as_deref())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if !user.is_enabled {
        return Err((StatusCode::FORBIDDEN, "User is disabled".to_string()));
    }

    Ok((auth_user, user))
}
