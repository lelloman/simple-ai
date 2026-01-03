use std::path::Path;
use std::sync::Mutex;
use rusqlite::{Connection, params};

use crate::models::audit::AuditLogEntry;

/// SQLite-based audit logger.
pub struct AuditLogger {
    conn: Mutex<Connection>,
}

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("IO error: {0}")]
    IoError(String),
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

        // Create table if not exists
        conn.execute(
            "CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_email TEXT,
                request_path TEXT NOT NULL,
                request_body TEXT,
                response_status INTEGER NOT NULL,
                response_body TEXT,
                latency_ms INTEGER NOT NULL,
                model_used TEXT,
                tokens_prompt INTEGER,
                tokens_completion INTEGER
            )",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Create index on timestamp for efficient queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Create index on user_id for per-user queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        tracing::info!("Audit logger initialized with database: {}", path);

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Log an audit entry.
    pub fn log(&self, entry: &AuditLogEntry) -> Result<(), AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "INSERT INTO audit_logs (
                id, timestamp, user_id, user_email, request_path,
                request_body, response_status, response_body,
                latency_ms, model_used, tokens_prompt, tokens_completion
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                entry.id,
                entry.timestamp.to_rfc3339(),
                entry.user_id,
                entry.user_email,
                entry.request_path,
                entry.request_body,
                entry.response_status,
                entry.response_body,
                entry.latency_ms as i64,
                entry.model_used,
                entry.tokens_prompt.map(|v| v as i64),
                entry.tokens_completion.map(|v| v as i64),
            ],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        tracing::debug!("Logged audit entry: {}", entry.id);
        Ok(())
    }
}
