use std::net::SocketAddr;

use simple_server::auth::{Access, AsyncAccess, HeaderCredential, RepeatedHeaders, SchemeCase};
use simple_server::axum::http::{header::AUTHORIZATION, HeaderMap, StatusCode};

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

/// Authenticate inference requests, allowing the temporary LAN identity only
/// when no Authorization header was supplied.
pub async fn authenticate_inference_request(
    state: &AppState,
    headers: &HeaderMap,
    peer: Option<SocketAddr>,
) -> Result<(AuthUser, User), (StatusCode, String)> {
    authenticate_with_context(AuthContext {
        state,
        headers,
        peer,
        allow_lan: true,
    })
    .await
}

/// Authenticate a request using API key or JWT.
pub async fn authenticate_request(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(AuthUser, User), (StatusCode, String)> {
    authenticate_with_context(AuthContext {
        state,
        headers,
        peer: None,
        allow_lan: false,
    })
    .await
}

struct AuthContext<'a> {
    state: &'a AppState,
    headers: &'a HeaderMap,
    peer: Option<SocketAddr>,
    allow_lan: bool,
}

async fn authenticate_with_context(
    context: AuthContext<'_>,
) -> Result<(AuthUser, User), (StatusCode, String)> {
    // The verifier selects the credential source. A supplied Authorization
    // header, even empty or invalid text, never permits LAN identity.
    let access = AsyncAccess::new(|context: &AuthContext<'_>| {
        Box::pin(async move {
            let state = context.state;
            let headers = context.headers;
            if context.allow_lan
                && !headers.contains_key(AUTHORIZATION)
                && state.lan_local.allows(headers, context.peer)
            {
                let user = state
                    .audit_logger
                    .find_or_create_user("lan-local", None)
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                let auth_user = AuthUser::new(
                    user.id.clone(),
                    None,
                    vec![crate::gateway::model_class::roles::MODEL_SPECIFIC.to_string()],
                );
                return Ok((auth_user, user));
            }

            // Compatibility: the old parser took the first header and exact
            // "Bearer " spelling, including an empty token. JWT verification
            // still owns all malformed-header errors and token validation.
            let bearer = HeaderCredential::new(AUTHORIZATION)
                .with_scheme("Bearer", SchemeCase::Exact)
                .repeated(RepeatedHeaders::First)
                .allow_empty(true)
                .extract(headers);
            if let Ok(credential) = bearer {
                let token = credential.expose();
                if token.starts_with("sk-") {
                    return match state.audit_logger.validate_api_key(token) {
                        Ok(Some((user_id, email, roles))) => {
                            let user = state
                                .audit_logger
                                .find_or_create_user(&user_id, email.as_deref())
                                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                            let auth_user = AuthUser::new(user_id, email, roles);
                            Ok((auth_user, user))
                        }
                        Ok(None) => Err((StatusCode::UNAUTHORIZED, "Invalid API key".to_string())),
                        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
                    };
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
            Ok((auth_user, user))
        })
    })
    .with_check(|(_, user): &(AuthUser, User), _| {
        Box::pin(async move {
            if user.is_enabled {
                Ok(())
            } else {
                Err((StatusCode::FORBIDDEN, "User is disabled".to_string()))
            }
        })
    });
    access.evaluate(&context).await
}

/// Apply the shared authorization flow to a previously verified identity.
pub fn authorize_admin(user: &AuthUser) -> bool {
    Access::new(|user: &AuthUser| Ok::<AuthUser, ()>(user.clone()))
        .with_check(|user, _| if user.is_admin() { Ok(()) } else { Err(()) })
        .evaluate(user)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admin_access_accepts_role_or_configured_user_only() {
        let admin_role = AuthUser::new_for_test(
            "role-user".into(),
            None,
            vec!["operator".into(), "super-admin".into()],
            "super-admin".into(),
            vec![],
        );
        let admin_id = AuthUser::new_for_test(
            "listed-user".into(),
            None,
            vec![],
            "super-admin".into(),
            vec!["listed-user".into()],
        );
        let ordinary = AuthUser::new_for_test(
            "ordinary-user".into(),
            None,
            vec!["admin".into()],
            "super-admin".into(),
            vec!["listed-user".into()],
        );
        assert!(authorize_admin(&admin_role));
        assert!(authorize_admin(&admin_id));
        assert!(!authorize_admin(&ordinary));
    }
}
