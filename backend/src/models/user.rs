use chrono::{DateTime, Utc};
use serde::Serialize;

/// User record created on first authentication.
#[derive(Debug, Clone, Serialize)]
pub struct User {
    /// User ID from OIDC (sub claim)
    pub id: String,
    /// Email from OIDC token
    pub email: Option<String>,
    /// When the user first authenticated
    pub created_at: DateTime<Utc>,
    /// When the user last made a request
    pub last_seen_at: DateTime<Utc>,
    /// Whether the user is allowed to make requests
    pub is_enabled: bool,
}
