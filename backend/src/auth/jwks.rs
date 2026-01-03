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
        })
    }
}

#[derive(Debug, Deserialize)]
struct OidcConfig {
    jwks_uri: String,
}
