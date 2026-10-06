use std::net::SocketAddr;

use simple_server::auth::{Access, AsyncAccess, HeaderCredential, RepeatedHeaders, SchemeCase};
use simple_server::web::http::{header::AUTHORIZATION, HeaderMap, StatusCode};

use crate::auth::AuthUser;
use crate::models::user::User;
use crate::AppState;

/// Attach verified identity and bounded, informational client metadata.
pub fn attribute_request(
    request: &mut crate::models::request::Request,
    user: &AuthUser,
    headers: &HeaderMap,
    peer: Option<SocketAddr>,
    trusted_proxies: &[ipnet::IpNet],
) {
    request.auth_method = Some(user.auth_method.to_string());
    request.api_key_id = user.api_key_id.clone();
    request.api_key_name = user.api_key_name.clone();
    request.peer_ip = peer.map(|p| p.ip().to_string());
    request.proxy_request_id = peer.filter(|p| trusted_proxies.iter().any(|net| net.contains(&p.ip())))
        .and_then(|_| headers.get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| uuid::Uuid::parse_str(value).ok())
        .map(|id| id.to_string());
    request.client_ip = verified_client_ip(headers, peer, trusted_proxies);
    request.user_agent = headers.get("user-agent")
        .and_then(|v| v.to_str().ok())
        .filter(|v| v.len() <= 1024 && !v.chars().any(char::is_control))
        .map(str::to_owned);
}

/// Walk from the actual socket peer toward the client, trusting only configured hops.
fn verified_client_ip(
    headers: &HeaderMap,
    peer: Option<SocketAddr>,
    trusted_proxies: &[ipnet::IpNet],
) -> Option<String> {
    let mut current = peer?.ip();
    let trusted = |ip: &std::net::IpAddr| trusted_proxies.iter().any(|net| net.contains(ip));
    if !trusted(&current) { return Some(current.to_string()); }
    let hops: Result<Vec<std::net::IpAddr>, _> = headers.get_all("x-forwarded-for")
        .iter().flat_map(|v| v.to_str().unwrap_or("").split(','))
        .map(|v| v.trim().parse()).collect();
    // Malformed chains cannot assert a client identity.
    if let Ok(hops) = hops {
        for hop in hops.into_iter().rev() {
            if !trusted(&current) { break; }
            current = hop;
        }
    }
    Some(current.to_string())
}

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
                let mut auth_user = AuthUser::new(
                    user.id.clone(),
                    None,
                    vec![crate::gateway::model_class::roles::MODEL_SPECIFIC.to_string()],
                );
                auth_user.auth_method = "lan_local";
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
                    return match state.audit_logger.validate_api_key_with_identity(token) {
                        Ok(Some(identity)) => {
                            let crate::audit::ValidatedApiKey { user_id, email, roles, key_id, key_name } = identity;
                            let user = state
                                .audit_logger
                                .find_or_create_user(&user_id, email.as_deref())
                                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                            let mut auth_user = AuthUser::new(user_id, email, roles);
                            auth_user.auth_method = "api_key";
                            auth_user.api_key_id = Some(key_id);
                            auth_user.api_key_name = Some(key_name);
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
    fn origin_metadata_requires_a_trusted_socket_peer() {
        let trusted = vec!["172.20.0.19/32".parse().unwrap(), "10.77.0.1/32".parse().unwrap()];
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4, 203.0.113.8, 10.77.0.1".parse().unwrap());
        assert_eq!(verified_client_ip(&headers, Some("172.20.0.19:4444".parse().unwrap()), &trusted).as_deref(), Some("203.0.113.8"));
        assert_eq!(verified_client_ip(&headers, Some("192.168.1.21:4444".parse().unwrap()), &trusted).as_deref(), Some("192.168.1.21"));
        assert_eq!(verified_client_ip(&headers, None, &trusted), None);
        headers.insert("x-forwarded-for", "bogus, 203.0.113.8".parse().unwrap());
        assert_eq!(verified_client_ip(&headers, Some("172.20.0.19:4444".parse().unwrap()), &trusted).as_deref(), Some("172.20.0.19"));
    }

    #[test]
    fn attribution_uses_verified_credential_not_claimed_headers() {
        let mut user = AuthUser::new("owner".into(), None, vec![]);
        user.auth_method = "api_key";
        user.api_key_id = Some("verified-id".into());
        user.api_key_name = Some("qwencode".into());
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer SECRET-NOT-LOGGED".parse().unwrap());
        headers.insert("x-api-key-name", "forged".parse().unwrap());
        headers.insert("user-agent", "probe-client/1".parse().unwrap());
        let mut request = crate::models::request::Request::new("owner".into(), "/v1/chat/completions".into());
        attribute_request(&mut request, &user, &headers, Some("192.168.1.21:8888".parse().unwrap()), &[]);
        assert_eq!(request.api_key_name.as_deref(), Some("qwencode"));
        assert_eq!(request.api_key_id.as_deref(), Some("verified-id"));
        assert_eq!(request.user_agent.as_deref(), Some("probe-client/1"));
        assert!(!serde_json::to_string(&request).unwrap().contains("SECRET-NOT-LOGGED"));
    }

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
