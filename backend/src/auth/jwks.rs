use std::sync::Arc;
use axum::http::HeaderMap;
use jsonwebtoken::{decode, decode_header, DecodingKey, Validation, Algorithm};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use std::collections::HashMap;

use crate::config::OidcConfig as AppOidcConfig;

/// Authenticated user information extracted from JWT.
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub sub: String,
    pub email: Option<String>,
    /// Per-app roles from the OIDC provider.
    pub roles: Vec<String>,
    /// The configured admin role name.
    admin_role: String,
    /// Explicit list of admin user IDs.
    admin_users: Vec<String>,
}

impl AuthUser {
    /// Create a new AuthUser for testing purposes.
    #[cfg(test)]
    pub fn new_for_test(
        sub: String,
        email: Option<String>,
        roles: Vec<String>,
        admin_role: String,
        admin_users: Vec<String>,
    ) -> Self {
        Self {
            sub,
            email,
            roles,
            admin_role,
            admin_users,
        }
    }

    /// Create a new AuthUser with default admin configuration.
    ///
    /// Used for testing and when creating mock users.
    pub fn new(sub: String, email: Option<String>, roles: Vec<String>) -> Self {
        Self {
            sub,
            email,
            roles,
            admin_role: "admin".to_string(),
            admin_users: vec![],
        }
    }

    /// Check if the user has admin access.
    ///
    /// Returns true if either:
    /// - The user has the configured admin role
    /// - The user's subject ID is in the explicit admin_users list
    pub fn is_admin(&self) -> bool {
        self.roles.iter().any(|r| r == &self.admin_role)
            || self.admin_users.iter().any(|id| id == &self.sub)
    }

    /// Check if the user has a specific role.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Missing Authorization header")]
    MissingHeader,
    #[error("Invalid Authorization header format")]
    InvalidFormat,
    #[error("Invalid token: {0}")]
    InvalidToken(String),
    #[error("JWKS fetch error: {0}")]
    JwksFetchError(String),
    #[error("Key not found for kid: {0}")]
    KeyNotFound(String),
}

/// JWKS key set response.
#[derive(Debug, Deserialize)]
struct JwksResponse {
    keys: Vec<Jwk>,
}

#[derive(Debug, Clone, Deserialize)]
struct Jwk {
    kid: String,
    kty: String,
    #[allow(dead_code)]
    alg: Option<String>,
    n: Option<String>,
    e: Option<String>,
}

/// JWT claims.
#[derive(Debug, Deserialize, Serialize)]
struct Claims {
    sub: String,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    aud: Value,
    exp: u64,
    iat: u64,
    /// Capture all other claims for flexible role extraction.
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

/// Client for fetching and caching JWKS keys.
pub struct JwksClient {
    http_client: Client,
    jwks_uri: String,
    keys: Arc<RwLock<HashMap<String, DecodingKey>>>,
    issuer: String,
    /// Path to roles in JWT claims (e.g., "roles", "realm_access.roles").
    role_claim_path: String,
    /// Name of the admin role.
    admin_role: String,
    /// Explicit list of admin user IDs.
    admin_users: Vec<String>,
}

impl JwksClient {
    pub async fn new(config: &AppOidcConfig) -> Result<Self, AuthError> {
        let http_client = Client::new();

        // Fetch OIDC configuration to get JWKS URI
        let config_url = format!(
            "{}/.well-known/openid-configuration",
            config.issuer.trim_end_matches('/')
        );
        let oidc_discovery: OidcDiscovery = http_client
            .get(&config_url)
            .send()
            .await
            .map_err(|e| AuthError::JwksFetchError(e.to_string()))?
            .json()
            .await
            .map_err(|e| AuthError::JwksFetchError(e.to_string()))?;

        let client = Self {
            http_client,
            jwks_uri: oidc_discovery.jwks_uri,
            keys: Arc::new(RwLock::new(HashMap::new())),
            issuer: config.issuer.clone(),
            role_claim_path: config.role_claim_path.clone(),
            admin_role: config.admin_role.clone(),
            admin_users: config.admin_users.clone(),
        };

        // Fetch keys initially
        client.refresh_keys().await?;

        Ok(client)
    }

    /// Extract roles from a nested claim path (e.g., "realm_access.roles").
    fn extract_roles(extra: &HashMap<String, Value>, path: &str) -> Vec<String> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current: Option<&Value> = None;

        for (i, part) in parts.iter().enumerate() {
            if i == 0 {
                current = extra.get(*part);
            } else if let Some(Value::Object(obj)) = current {
                current = obj.get(*part);
            } else {
                return vec![];
            }
        }

        match current {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => vec![],
        }
    }

    async fn refresh_keys(&self) -> Result<(), AuthError> {
        tracing::info!("Fetching JWKS from {}", self.jwks_uri);

        let response: JwksResponse = self.http_client
            .get(&self.jwks_uri)
            .send()
            .await
            .map_err(|e| AuthError::JwksFetchError(e.to_string()))?
            .json()
            .await
            .map_err(|e| AuthError::JwksFetchError(e.to_string()))?;

        let mut keys = self.keys.write().await;
        keys.clear();

        for jwk in response.keys {
            if jwk.kty == "RSA" {
                if let (Some(n), Some(e)) = (&jwk.n, &jwk.e) {
                    match DecodingKey::from_rsa_components(n, e) {
                        Ok(key) => {
                            keys.insert(jwk.kid.clone(), key);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse RSA key {}: {}", jwk.kid, e);
                        }
                    }
                }
            }
        }

        tracing::info!("Loaded {} JWKS keys", keys.len());
        Ok(())
    }

    /// Authenticate a request by validating the Bearer token.
    pub async fn authenticate(&self, headers: &HeaderMap) -> Result<AuthUser, AuthError> {
        let auth_header = headers
            .get("authorization")
            .ok_or(AuthError::MissingHeader)?
            .to_str()
            .map_err(|_| AuthError::InvalidFormat)?;

        if !auth_header.starts_with("Bearer ") {
            return Err(AuthError::InvalidFormat);
        }

        let token = &auth_header[7..];
        self.validate_token(token).await
    }

    /// Validate a JWT token string directly.
    ///
    /// This is useful for SSE endpoints where the token is passed as a query parameter
    /// instead of an Authorization header.
    pub async fn validate_token(&self, token: &str) -> Result<AuthUser, AuthError> {
        // Decode header to get kid
        let header = decode_header(token)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        let kid = header.kid
            .ok_or_else(|| AuthError::InvalidToken("Missing kid in token header".to_string()))?;

        // Get key for kid
        let keys = self.keys.read().await;
        let key = keys.get(&kid)
            .ok_or_else(|| AuthError::KeyNotFound(kid.clone()))?;

        // Validate token
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&[&self.issuer]);
        // Skip audience validation for now (can be added later)
        validation.validate_aud = false;

        let token_data = decode::<Claims>(token, key, &validation)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        // Extract roles from the configured claim path
        let roles = Self::extract_roles(&token_data.claims.extra, &self.role_claim_path);

        tracing::debug!(
            "JWT validated: sub={}, email={:?}, roles={:?}, admin_users={:?}",
            token_data.claims.sub,
            token_data.claims.email,
            roles,
            self.admin_users
        );

        Ok(AuthUser {
            sub: token_data.claims.sub,
            email: token_data.claims.email,
            roles,
            admin_role: self.admin_role.clone(),
            admin_users: self.admin_users.clone(),
        })
    }
}

#[derive(Debug, Deserialize)]
struct OidcDiscovery {
    jwks_uri: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use axum::http::header::AUTHORIZATION;

    fn empty_headers() -> HeaderMap {
        HeaderMap::new()
    }

    fn headers_with_auth(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, token.parse().unwrap());
        headers
    }

    /// Helper to create AuthUser with default admin config.
    fn make_user(sub: &str, email: Option<&str>, roles: Vec<&str>) -> AuthUser {
        AuthUser::new_for_test(
            sub.to_string(),
            email.map(|e| e.to_string()),
            roles.into_iter().map(|r| r.to_string()).collect(),
            "admin".to_string(),
            vec![],
        )
    }

    /// Helper to create AuthUser with explicit admin users.
    fn make_user_with_admin_users(
        sub: &str,
        email: Option<&str>,
        roles: Vec<&str>,
        admin_users: Vec<&str>,
    ) -> AuthUser {
        AuthUser::new_for_test(
            sub.to_string(),
            email.map(|e| e.to_string()),
            roles.into_iter().map(|r| r.to_string()).collect(),
            "admin".to_string(),
            admin_users.into_iter().map(|u| u.to_string()).collect(),
        )
    }

    #[test]
    fn test_auth_user_is_admin_with_admin_role() {
        let user = make_user("user123", Some("user@example.com"), vec!["admin", "user"]);
        assert!(user.is_admin());
    }

    #[test]
    fn test_auth_user_is_admin_without_admin_role() {
        let user = make_user("user123", None, vec!["user"]);
        assert!(!user.is_admin());
    }

    #[test]
    fn test_auth_user_is_admin_with_empty_roles() {
        let user = make_user("user123", None, vec![]);
        assert!(!user.is_admin());
    }

    #[test]
    fn test_auth_user_is_admin_via_admin_users_list() {
        let user = make_user_with_admin_users(
            "user123",
            None,
            vec![],
            vec!["user123"],
        );
        assert!(user.is_admin());
    }

    #[test]
    fn test_auth_user_is_admin_via_admin_users_list_no_match() {
        let user = make_user_with_admin_users(
            "user123",
            None,
            vec![],
            vec!["other-user"],
        );
        assert!(!user.is_admin());
    }

    #[test]
    fn test_auth_user_is_admin_with_custom_role_name() {
        // User has "super-admin" role, and admin_role is configured as "super-admin"
        let user = AuthUser::new_for_test(
            "user123".to_string(),
            None,
            vec!["super-admin".to_string()],
            "super-admin".to_string(),
            vec![],
        );
        assert!(user.is_admin());
    }

    #[test]
    fn test_auth_user_is_not_admin_when_role_name_mismatch() {
        // User has "admin" role, but admin_role is configured as "super-admin"
        let user = AuthUser::new_for_test(
            "user123".to_string(),
            None,
            vec!["admin".to_string()],
            "super-admin".to_string(),
            vec![],
        );
        assert!(!user.is_admin());
    }

    #[test]
    fn test_auth_user_has_role_exact_match() {
        let user = make_user("user123", None, vec!["moderator", "viewer"]);
        assert!(user.has_role("moderator"));
        assert!(user.has_role("viewer"));
        assert!(!user.has_role("admin"));
    }

    #[test]
    fn test_auth_user_has_role_case_sensitive() {
        let user = make_user("user123", None, vec!["Admin"]);
        assert!(!user.has_role("admin"));
        assert!(user.has_role("Admin"));
    }

    #[test]
    fn test_auth_user_has_role_with_empty_roles() {
        let user = make_user("user123", None, vec![]);
        assert!(!user.has_role("any"));
    }

    #[test]
    fn test_auth_user_sub_and_email() {
        let user = make_user("auth0|123456", Some("test@auth0.com"), vec![]);
        assert_eq!(user.sub, "auth0|123456");
        assert_eq!(user.email, Some("test@auth0.com".to_string()));
    }

    #[test]
    fn test_auth_user_without_email() {
        let user = make_user("user123", None, vec![]);
        assert!(user.email.is_none());
    }

    #[test]
    fn test_auth_error_missing_header() {
        let result: Result<AuthUser, AuthError> = Err(AuthError::MissingHeader);
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(e.to_string(), "Missing Authorization header");
        }
    }

    #[test]
    fn test_auth_error_invalid_format() {
        let result: Result<AuthUser, AuthError> = Err(AuthError::InvalidFormat);
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(e.to_string(), "Invalid Authorization header format");
        }
    }

    #[test]
    fn test_auth_error_invalid_token() {
        let result: Result<AuthUser, AuthError> = Err(AuthError::InvalidToken("test error".to_string()));
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid token"));
        }
    }

    #[test]
    fn test_auth_error_jwks_fetch_error() {
        let result: Result<AuthUser, AuthError> = Err(AuthError::JwksFetchError("connection refused".to_string()));
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("JWKS fetch error"));
        }
    }

    #[test]
    fn test_auth_error_key_not_found() {
        let result: Result<AuthUser, AuthError> = Err(AuthError::KeyNotFound("kid123".to_string()));
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Key not found for kid"));
        }
    }

    #[test]
    fn test_auth_user_clone() {
        let original = make_user("user123", Some("user@example.com"), vec!["admin"]);
        let cloned = original.clone();
        assert_eq!(cloned.sub, original.sub);
        assert_eq!(cloned.email, original.email);
        assert_eq!(cloned.roles, original.roles);
    }

    #[test]
    fn test_auth_user_debug_format() {
        let user = make_user("user123", Some("user@example.com"), vec!["admin"]);
        let debug = format!("{:?}", user);
        assert!(debug.contains("user123"));
        assert!(debug.contains("user@example.com"));
        assert!(debug.contains("admin"));
    }

    #[test]
    fn test_bearer_token_extraction_valid() {
        let headers = headers_with_auth("Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test");
        let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());
        assert!(auth_header.is_some());
        assert!(auth_header.unwrap().starts_with("Bearer "));
    }

    #[test]
    fn test_bearer_token_extraction_basic_auth() {
        let headers = headers_with_auth("Basic dXNlcjpwYXNz");
        let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());
        assert!(auth_header.is_some());
        assert!(!auth_header.unwrap().starts_with("Bearer "));
    }

    #[test]
    fn test_empty_headers_has_no_auth() {
        let headers = empty_headers();
        assert!(headers.get("authorization").is_none());
    }

    // Tests for role extraction from claim paths

    #[test]
    fn test_extract_roles_simple_path() {
        let mut extra = HashMap::new();
        extra.insert(
            "roles".to_string(),
            serde_json::json!(["admin", "user"]),
        );
        let roles = JwksClient::extract_roles(&extra, "roles");
        assert_eq!(roles, vec!["admin", "user"]);
    }

    #[test]
    fn test_extract_roles_nested_path() {
        let mut extra = HashMap::new();
        extra.insert(
            "realm_access".to_string(),
            serde_json::json!({
                "roles": ["admin", "user"]
            }),
        );
        let roles = JwksClient::extract_roles(&extra, "realm_access.roles");
        assert_eq!(roles, vec!["admin", "user"]);
    }

    #[test]
    fn test_extract_roles_deeply_nested_path() {
        let mut extra = HashMap::new();
        extra.insert(
            "resource_access".to_string(),
            serde_json::json!({
                "my-app": {
                    "roles": ["editor", "viewer"]
                }
            }),
        );
        let roles = JwksClient::extract_roles(&extra, "resource_access.my-app.roles");
        assert_eq!(roles, vec!["editor", "viewer"]);
    }

    #[test]
    fn test_extract_roles_missing_path() {
        let extra = HashMap::new();
        let roles = JwksClient::extract_roles(&extra, "roles");
        assert!(roles.is_empty());
    }

    #[test]
    fn test_extract_roles_path_not_array() {
        let mut extra = HashMap::new();
        extra.insert("roles".to_string(), serde_json::json!("not-an-array"));
        let roles = JwksClient::extract_roles(&extra, "roles");
        assert!(roles.is_empty());
    }

    #[test]
    fn test_extract_roles_intermediate_not_object() {
        let mut extra = HashMap::new();
        extra.insert("realm_access".to_string(), serde_json::json!("not-an-object"));
        let roles = JwksClient::extract_roles(&extra, "realm_access.roles");
        assert!(roles.is_empty());
    }
}
