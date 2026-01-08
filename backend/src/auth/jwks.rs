use std::sync::Arc;
use axum::http::HeaderMap;
use jsonwebtoken::{decode, decode_header, DecodingKey, Validation, Algorithm};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Authenticated user information extracted from JWT.
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub sub: String,
    pub email: Option<String>,
    /// Per-app roles from the OIDC provider.
    pub roles: Vec<String>,
}

impl AuthUser {
    /// Check if the user has the "admin" role.
    pub fn is_admin(&self) -> bool {
        self.roles.iter().any(|r| r == "admin")
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
    roles: Vec<String>,
    #[serde(default)]
    aud: serde_json::Value,
    exp: u64,
    iat: u64,
}

/// Client for fetching and caching JWKS keys.
pub struct JwksClient {
    http_client: Client,
    jwks_uri: String,
    keys: Arc<RwLock<HashMap<String, DecodingKey>>>,
    issuer: String,
}

impl JwksClient {
    pub async fn new(issuer: &str) -> Result<Self, AuthError> {
        let http_client = Client::new();

        // Fetch OIDC configuration to get JWKS URI
        let config_url = format!("{}/.well-known/openid-configuration", issuer.trim_end_matches('/'));
        let config: OidcConfig = http_client
            .get(&config_url)
            .send()
            .await
            .map_err(|e| AuthError::JwksFetchError(e.to_string()))?
            .json()
            .await
            .map_err(|e| AuthError::JwksFetchError(e.to_string()))?;

        let client = Self {
            http_client,
            jwks_uri: config.jwks_uri,
            keys: Arc::new(RwLock::new(HashMap::new())),
            issuer: issuer.to_string(),
        };

        // Fetch keys initially
        client.refresh_keys().await?;

        Ok(client)
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

        Ok(AuthUser {
            sub: token_data.claims.sub,
            email: token_data.claims.email,
            roles: token_data.claims.roles,
        })
    }
}

#[derive(Debug, Deserialize)]
struct OidcConfig {
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

    #[test]
    fn test_auth_user_is_admin_with_admin_role() {
        let user = AuthUser {
            sub: "user123".to_string(),
            email: Some("user@example.com".to_string()),
            roles: vec!["admin".to_string(), "user".to_string()],
        };
        assert!(user.is_admin());
    }

    #[test]
    fn test_auth_user_is_admin_without_admin_role() {
        let user = AuthUser {
            sub: "user123".to_string(),
            email: None,
            roles: vec!["user".to_string()],
        };
        assert!(!user.is_admin());
    }

    #[test]
    fn test_auth_user_is_admin_with_empty_roles() {
        let user = AuthUser {
            sub: "user123".to_string(),
            email: None,
            roles: vec![],
        };
        assert!(!user.is_admin());
    }

    #[test]
    fn test_auth_user_has_role_exact_match() {
        let user = AuthUser {
            sub: "user123".to_string(),
            email: None,
            roles: vec!["moderator".to_string(), "viewer".to_string()],
        };
        assert!(user.has_role("moderator"));
        assert!(user.has_role("viewer"));
        assert!(!user.has_role("admin"));
    }

    #[test]
    fn test_auth_user_has_role_case_sensitive() {
        let user = AuthUser {
            sub: "user123".to_string(),
            email: None,
            roles: vec!["Admin".to_string()],
        };
        assert!(!user.has_role("admin"));
        assert!(user.has_role("Admin"));
    }

    #[test]
    fn test_auth_user_has_role_with_empty_roles() {
        let user = AuthUser {
            sub: "user123".to_string(),
            email: None,
            roles: vec![],
        };
        assert!(!user.has_role("any"));
    }

    #[test]
    fn test_auth_user_sub_and_email() {
        let user = AuthUser {
            sub: "auth0|123456".to_string(),
            email: Some("test@auth0.com".to_string()),
            roles: vec![],
        };
        assert_eq!(user.sub, "auth0|123456");
        assert_eq!(user.email, Some("test@auth0.com".to_string()));
    }

    #[test]
    fn test_auth_user_without_email() {
        let user = AuthUser {
            sub: "user123".to_string(),
            email: None,
            roles: vec![],
        };
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
        let original = AuthUser {
            sub: "user123".to_string(),
            email: Some("user@example.com".to_string()),
            roles: vec!["admin".to_string()],
        };
        let cloned = original.clone();
        assert_eq!(cloned.sub, original.sub);
        assert_eq!(cloned.email, original.email);
        assert_eq!(cloned.roles, original.roles);
    }

    #[test]
    fn test_auth_user_debug_format() {
        let user = AuthUser {
            sub: "user123".to_string(),
            email: Some("user@example.com".to_string()),
            roles: vec!["admin".to_string()],
        };
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
}
