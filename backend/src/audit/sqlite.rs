use std::path::Path;
use std::sync::Mutex;
use chrono::Utc;
use rusqlite::{Connection, params};

use crate::models::user::User;
use crate::models::request::{Request, Response};

/// SQLite-based audit logger with user management.
pub struct AuditLogger {
    conn: Mutex<Connection>,
}

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("User is disabled")]
    UserDisabled,
}

impl AuditLogger {
    pub fn new(database_url: &str) -> Result<Self, AuditError> {
        // Parse sqlite: prefix if present
        let path = if database_url.starts_with("sqlite:") {
            &database_url[7..]
        } else {
            database_url
        };

        // Create parent directories if needed
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AuditError::IoError(e.to_string()))?;
        }

        let conn = Connection::open(path)
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Create users table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                is_enabled INTEGER NOT NULL DEFAULT 1
            )",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Create requests table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                request_path TEXT NOT NULL,
                request_body TEXT,
                model TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Create responses table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status INTEGER NOT NULL,
                response_body TEXT,
                latency_ms INTEGER NOT NULL,
                tokens_prompt INTEGER,
                tokens_completion INTEGER,
                FOREIGN KEY (request_id) REFERENCES requests(id)
            )",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Create indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id)",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_responses_request_id ON responses(request_id)",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        tracing::info!("Audit logger initialized with database: {}", path);

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Find or create a user. Returns the user.
    pub fn find_or_create_user(&self, user_id: &str, email: Option<&str>) -> Result<User, AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let now = Utc::now();

        // Try to find existing user
        let existing: Option<(String, Option<String>, String, String, bool)> = conn
            .query_row(
                "SELECT id, email, created_at, last_seen_at, is_enabled FROM users WHERE id = ?1",
                params![user_id],
                |row| Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get::<_, i32>(4)? != 0,
                )),
            )
            .ok();

        match existing {
            Some((id, db_email, created_at, _, is_enabled)) => {
                // Update last_seen_at and email if changed
                conn.execute(
                    "UPDATE users SET last_seen_at = ?1, email = COALESCE(?2, email) WHERE id = ?3",
                    params![now.to_rfc3339(), email, user_id],
                ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

                let created = chrono::DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or(now);

                Ok(User {
                    id,
                    email: email.map(String::from).or(db_email),
                    created_at: created,
                    last_seen_at: now,
                    is_enabled,
                })
            }
            None => {
                // Create new user
                conn.execute(
                    "INSERT INTO users (id, email, created_at, last_seen_at, is_enabled) VALUES (?1, ?2, ?3, ?4, 1)",
                    params![user_id, email, now.to_rfc3339(), now.to_rfc3339()],
                ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

                tracing::info!("Created new user: {} ({})", user_id, email.unwrap_or("no email"));

                Ok(User {
                    id: user_id.to_string(),
                    email: email.map(String::from),
                    created_at: now,
                    last_seen_at: now,
                    is_enabled: true,
                })
            }
        }
    }

    /// Log a request (before calling the LLM). Returns the request ID.
    pub fn log_request(&self, request: &Request) -> Result<String, AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "INSERT INTO requests (id, timestamp, user_id, request_path, request_body, model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                request.id,
                request.timestamp.to_rfc3339(),
                request.user_id,
                request.request_path,
                request.request_body,
                request.model,
            ],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        tracing::debug!("Logged request: {}", request.id);
        Ok(request.id.clone())
    }

    /// Log a response (after getting LLM result).
    pub fn log_response(&self, response: &Response) -> Result<(), AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "INSERT INTO responses (id, request_id, timestamp, status, response_body, latency_ms, tokens_prompt, tokens_completion)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                response.id,
                response.request_id,
                response.timestamp.to_rfc3339(),
                response.status,
                response.response_body,
                response.latency_ms as i64,
                response.tokens_prompt.map(|v| v as i64),
                response.tokens_completion.map(|v| v as i64),
            ],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        tracing::debug!("Logged response for request: {}", response.request_id);
        Ok(())
    }
}
