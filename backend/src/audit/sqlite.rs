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

        // Create runners table for persistent runner tracking
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runners (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                mac_address TEXT,
                machine_type TEXT,
                last_seen_at TEXT NOT NULL,
                available_models TEXT
            )",
            [],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Migration: add available_models column if it doesn't exist
        let _ = conn.execute(
            "ALTER TABLE runners ADD COLUMN available_models TEXT",
            [],
        ); // Ignore error if column already exists

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

    // ========== Admin queries ==========

    /// Get dashboard statistics.
    pub fn get_stats(&self) -> Result<DashboardStats, AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let total_users: i64 = conn
            .query_row("SELECT COUNT(*) FROM users", [], |row| row.get(0))
            .unwrap_or(0);

        let total_requests: i64 = conn
            .query_row("SELECT COUNT(*) FROM requests", [], |row| row.get(0))
            .unwrap_or(0);

        // Requests in last 24 hours
        let cutoff = (Utc::now() - chrono::Duration::hours(24)).to_rfc3339();
        let requests_24h: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM requests WHERE timestamp > ?1",
                params![cutoff],
                |row| row.get(0),
            )
            .unwrap_or(0);

        // Total tokens
        let total_tokens: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(COALESCE(tokens_prompt, 0) + COALESCE(tokens_completion, 0)), 0) FROM responses",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        Ok(DashboardStats {
            total_users: total_users as u64,
            total_requests: total_requests as u64,
            requests_24h: requests_24h as u64,
            total_tokens: total_tokens as u64,
        })
    }

    /// Get recent requests for dashboard.
    pub fn get_recent_requests(&self, limit: u32) -> Result<Vec<RequestSummary>, AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT r.id, r.timestamp, r.user_id, r.request_path, r.model
             FROM requests r
             ORDER BY r.timestamp DESC
             LIMIT ?1"
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt.query_map(params![limit], |row| {
            Ok(RequestSummary {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                user_id: row.get(2)?,
                request_path: row.get(3)?,
                model: row.get(4)?,
            })
        }).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut requests = Vec::new();
        for row in rows {
            requests.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }
        Ok(requests)
    }

    /// Get all users with request counts.
    pub fn get_users_with_stats(&self) -> Result<Vec<UserWithStats>, AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT u.id, u.email, u.created_at, u.last_seen_at, u.is_enabled,
                    (SELECT COUNT(*) FROM requests r WHERE r.user_id = u.id) as request_count
             FROM users u
             ORDER BY u.last_seen_at DESC"
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt.query_map([], |row| {
            Ok(UserWithStats {
                id: row.get(0)?,
                email: row.get(1)?,
                created_at: row.get(2)?,
                last_seen_at: row.get(3)?,
                is_enabled: row.get::<_, i32>(4)? != 0,
                request_count: row.get(5)?,
            })
        }).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut users = Vec::new();
        for row in rows {
            users.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }
        Ok(users)
    }

    /// Get requests with response details, with optional filtering and pagination.
    pub fn get_requests_paginated(
        &self,
        user_id: Option<&str>,
        model: Option<&str>,
        page: u32,
        per_page: u32,
    ) -> Result<(Vec<RequestWithResponse>, u32), AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Build query with filters
        let mut where_clauses = Vec::new();
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(uid) = user_id {
            if !uid.is_empty() {
                where_clauses.push("r.user_id LIKE ?");
                params_vec.push(Box::new(format!("%{}%", uid)));
            }
        }
        if let Some(m) = model {
            if !m.is_empty() {
                where_clauses.push("r.model LIKE ?");
                params_vec.push(Box::new(format!("%{}%", m)));
            }
        }

        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        // Get total count
        let count_sql = format!("SELECT COUNT(*) FROM requests r {}", where_clause);
        let total: i64 = {
            let mut stmt = conn.prepare(&count_sql)
                .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
            let params_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
            stmt.query_row(params_refs.as_slice(), |row| row.get(0))
                .unwrap_or(0)
        };

        let total_pages = ((total as u32) + per_page - 1) / per_page;
        let offset = (page.saturating_sub(1)) * per_page;

        // Get requests
        let query_sql = format!(
            "SELECT r.id, r.timestamp, r.user_id, r.request_path, r.model,
                    resp.status, resp.latency_ms, resp.tokens_prompt, resp.tokens_completion
             FROM requests r
             LEFT JOIN responses resp ON resp.request_id = r.id
             {}
             ORDER BY r.timestamp DESC
             LIMIT ? OFFSET ?",
            where_clause
        );

        params_vec.push(Box::new(per_page as i64));
        params_vec.push(Box::new(offset as i64));

        let mut stmt = conn.prepare(&query_sql)
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let params_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(RequestWithResponse {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                user_id: row.get(2)?,
                request_path: row.get(3)?,
                model: row.get(4)?,
                status: row.get(5)?,
                latency_ms: row.get(6)?,
                tokens_prompt: row.get(7)?,
                tokens_completion: row.get(8)?,
            })
        }).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut requests = Vec::new();
        for row in rows {
            requests.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }

        Ok((requests, total_pages))
    }

    /// Enable a user.
    pub fn enable_user(&self, user_id: &str) -> Result<(), AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "UPDATE users SET is_enabled = 1 WHERE id = ?1",
            params![user_id],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// Disable a user.
    pub fn disable_user(&self, user_id: &str) -> Result<(), AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "UPDATE users SET is_enabled = 0 WHERE id = ?1",
            params![user_id],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        Ok(())
    }
}

/// Dashboard statistics.
#[derive(Debug, Clone)]
pub struct DashboardStats {
    pub total_users: u64,
    pub total_requests: u64,
    pub requests_24h: u64,
    pub total_tokens: u64,
}

/// Request summary for dashboard.
#[derive(Debug, Clone)]
pub struct RequestSummary {
    pub id: String,
    pub timestamp: String,
    pub user_id: String,
    pub request_path: String,
    pub model: Option<String>,
}

/// User with request count.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UserWithStats {
    pub id: String,
    pub email: Option<String>,
    pub created_at: String,
    pub last_seen_at: String,
    pub is_enabled: bool,
    pub request_count: i64,
}

/// Request with response details.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RequestWithResponse {
    pub id: String,
    pub timestamp: String,
    pub user_id: String,
    pub request_path: String,
    pub model: Option<String>,
    pub status: Option<i32>,
    pub latency_ms: Option<i64>,
    pub tokens_prompt: Option<i64>,
    pub tokens_completion: Option<i64>,
}

/// Persistent runner record for WOL and offline tracking.
#[derive(Debug, Clone)]
pub struct RunnerRecord {
    pub id: String,
    pub name: String,
    pub mac_address: Option<String>,
    pub machine_type: Option<String>,
    pub last_seen_at: String,
    /// Available models on this runner (JSON array of model IDs).
    pub available_models: Vec<String>,
}

impl AuditLogger {
    /// Insert or update a runner record.
    pub fn upsert_runner(
        &self,
        id: &str,
        name: &str,
        mac_address: Option<&str>,
        machine_type: Option<&str>,
        available_models: Option<&[String]>,
    ) -> Result<(), AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let now = Utc::now().to_rfc3339();
        let models_json = available_models.map(|m| serde_json::to_string(m).unwrap_or_default());

        conn.execute(
            "INSERT INTO runners (id, name, mac_address, machine_type, last_seen_at, available_models)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                mac_address = COALESCE(excluded.mac_address, runners.mac_address),
                machine_type = COALESCE(excluded.machine_type, runners.machine_type),
                last_seen_at = excluded.last_seen_at,
                available_models = COALESCE(excluded.available_models, runners.available_models)",
            params![id, name, mac_address, machine_type, now, models_json],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        tracing::debug!("Upserted runner: {} with {} models", id, available_models.map(|m| m.len()).unwrap_or(0));
        Ok(())
    }

    /// Get a runner by ID.
    pub fn get_runner(&self, id: &str) -> Result<Option<RunnerRecord>, AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let result = conn.query_row(
            "SELECT id, name, mac_address, machine_type, last_seen_at, available_models FROM runners WHERE id = ?1",
            params![id],
            |row| {
                let models_json: Option<String> = row.get(5)?;
                let available_models = models_json
                    .and_then(|j| serde_json::from_str(&j).ok())
                    .unwrap_or_default();
                Ok(RunnerRecord {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    mac_address: row.get(2)?,
                    machine_type: row.get(3)?,
                    last_seen_at: row.get(4)?,
                    available_models,
                })
            },
        );

        match result {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AuditError::DatabaseError(e.to_string())),
        }
    }

    /// Get all runner records.
    pub fn get_all_runners(&self) -> Result<Vec<RunnerRecord>, AuditError> {
        let conn = self.conn.lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT id, name, mac_address, machine_type, last_seen_at, available_models FROM runners ORDER BY last_seen_at DESC"
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt.query_map([], |row| {
            let models_json: Option<String> = row.get(5)?;
            let available_models = models_json
                .and_then(|j| serde_json::from_str(&j).ok())
                .unwrap_or_default();
            Ok(RunnerRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                mac_address: row.get(2)?,
                machine_type: row.get(3)?,
                last_seen_at: row.get(4)?,
                available_models,
            })
        }).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut runners = Vec::new();
        for row in rows {
            runners.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }
        Ok(runners)
    }

    /// Get runners that have a model of the specified class.
    pub fn get_runners_by_model_class(
        &self,
        class: crate::gateway::ModelClass,
        models_config: &crate::config::ModelsConfig,
    ) -> Result<Vec<RunnerRecord>, AuditError> {
        let all = self.get_all_runners()?;
        Ok(all
            .into_iter()
            .filter(|r| {
                r.available_models
                    .iter()
                    .any(|m| crate::gateway::classify_model(m, models_config) == Some(class))
            })
            .collect())
    }

    /// Get runners that have a specific model.
    pub fn get_runners_by_model(&self, model_id: &str) -> Result<Vec<RunnerRecord>, AuditError> {
        let all = self.get_all_runners()?;
        Ok(all
            .into_iter()
            .filter(|r| r.available_models.iter().any(|m| m == model_id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use uuid::Uuid;

    fn create_test_logger() -> AuditLogger {
        let test_db_path = format!("test_audit_{}.db", Uuid::new_v4().to_string().replace('-', ""));
        AuditLogger::new(&test_db_path).unwrap()
    }

    fn cleanup_db(path: &str) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_new_creates_tables() {
        let test_db_path = format!("test_new_tables_{}.db", Uuid::new_v4().to_string().replace('-', ""));
        let logger = AuditLogger::new(&test_db_path).unwrap();
        drop(logger);
        assert!(fs::metadata(&test_db_path).is_ok());
        cleanup_db(&test_db_path);
    }

    #[test]
    fn test_find_or_create_user_new() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", Some("user@example.com")).unwrap();
        assert_eq!(user.id, "user123");
        assert_eq!(user.email, Some("user@example.com".to_string()));
        assert!(user.is_enabled);
    }

    #[test]
    fn test_find_or_create_user_existing() {
        let logger = create_test_logger();
        let user1 = logger.find_or_create_user("user123", Some("user@example.com")).unwrap();
        let user2 = logger.find_or_create_user("user123", Some("user2@example.com")).unwrap();
        assert_eq!(user1.id, user2.id);
        assert_eq!(user2.email, Some("user2@example.com".to_string()));
    }

    #[test]
    fn test_find_or_create_user_without_email() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        assert_eq!(user.id, "user123");
        assert!(user.email.is_none());
    }

    #[test]
    fn test_user_last_seen_updated() {
        let logger = create_test_logger();
        let user1 = logger.find_or_create_user("user123", None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let user2 = logger.find_or_create_user("user123", None).unwrap();
        assert!(user2.last_seen_at >= user1.last_seen_at);
    }

    #[test]
    fn test_log_request() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let mut request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
        request.request_body = r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_string();
        request.model = Some("llama2".to_string());
        let request_id = logger.log_request(&request).unwrap();
        assert_eq!(request_id, request.id);
    }

    #[test]
    fn test_log_response() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
        let request_id = logger.log_request(&request).unwrap();

        let mut response = Response::new(request_id, 200);
        response.response_body = r#"{"choices":[{"message":{"content":"Hello"}}]}"#.to_string();
        response.latency_ms = 150;
        response.tokens_prompt = Some(10);
        response.tokens_completion = Some(5);
        logger.log_response(&response).unwrap();
    }

    #[test]
    fn test_get_stats_empty() {
        let logger = create_test_logger();
        let stats = logger.get_stats().unwrap();
        assert_eq!(stats.total_users, 0);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.requests_24h, 0);
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn test_get_stats_with_users() {
        let logger = create_test_logger();
        logger.find_or_create_user("user1", None).unwrap();
        logger.find_or_create_user("user2", None).unwrap();
        let stats = logger.get_stats().unwrap();
        assert_eq!(stats.total_users, 2);
    }

    #[test]
    fn test_get_stats_with_requests() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
        logger.log_request(&request).unwrap();
        let stats = logger.get_stats().unwrap();
        assert_eq!(stats.total_requests, 1);
    }

    #[test]
    fn test_get_stats_with_tokens() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
        let request_id = logger.log_request(&request).unwrap();
        let mut response = Response::new(request_id, 200);
        response.tokens_prompt = Some(10);
        response.tokens_completion = Some(20);
        logger.log_response(&response).unwrap();
        let stats = logger.get_stats().unwrap();
        assert_eq!(stats.total_tokens, 30);
    }

    #[test]
    fn test_get_recent_requests_empty() {
        let logger = create_test_logger();
        let requests = logger.get_recent_requests(10).unwrap();
        assert!(requests.is_empty());
    }

    #[test]
    fn test_get_recent_requests_with_data() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        for i in 0..5 {
            let mut request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
            request.model = Some(format!("model-{}", i));
            logger.log_request(&request).unwrap();
        }
        let requests = logger.get_recent_requests(3).unwrap();
        assert_eq!(requests.len(), 3);
    }

    #[test]
    fn test_get_users_with_stats() {
        let logger = create_test_logger();
        logger.find_or_create_user("user1", None).unwrap();
        let user2 = logger.find_or_create_user("user2", None).unwrap();
        let request = Request::new(user2.id.clone(), "/v1/chat/completions".to_string());
        logger.log_request(&request).unwrap();
        let users = logger.get_users_with_stats().unwrap();
        assert_eq!(users.len(), 2);
        let user2_stats = users.iter().find(|u| u.id == "user2").unwrap();
        assert_eq!(user2_stats.request_count, 1);
    }

    #[test]
    fn test_enable_user() {
        let logger = create_test_logger();
        logger.find_or_create_user("user123", None).unwrap();
        logger.disable_user("user123").unwrap();
        logger.enable_user("user123").unwrap();
        let users = logger.get_users_with_stats().unwrap();
        let user = users.iter().find(|u| u.id == "user123").unwrap();
        assert!(user.is_enabled);
    }

    #[test]
    fn test_disable_user() {
        let logger = create_test_logger();
        logger.find_or_create_user("user123", None).unwrap();
        logger.disable_user("user123").unwrap();
        let users = logger.get_users_with_stats().unwrap();
        let user = users.iter().find(|u| u.id == "user123").unwrap();
        assert!(!user.is_enabled);
    }

    #[test]
    fn test_get_requests_paginated() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        for i in 0..15 {
            let mut request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
            request.model = Some(format!("model-{}", i));
            logger.log_request(&request).unwrap();
        }
        let (requests, total_pages) = logger.get_requests_paginated(None, None, 1, 5).unwrap();
        assert_eq!(requests.len(), 5);
        assert_eq!(total_pages, 3);
    }

    #[test]
    fn test_get_requests_paginated_filter_by_user() {
        let logger = create_test_logger();
        let user1 = logger.find_or_create_user("user1", None).unwrap();
        let user2 = logger.find_or_create_user("user2", None).unwrap();
        for _ in 0..3 {
            let request = Request::new(user1.id.clone(), "/v1/chat/completions".to_string());
            logger.log_request(&request).unwrap();
        }
        for _ in 0..2 {
            let request = Request::new(user2.id.clone(), "/v1/chat/completions".to_string());
            logger.log_request(&request).unwrap();
        }
        let (requests, _) = logger.get_requests_paginated(Some("user1"), None, 1, 10).unwrap();
        assert!(requests.iter().all(|r| r.user_id.contains("user1")));
    }

    #[test]
    fn test_get_requests_paginated_filter_by_model() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        for i in 0..5 {
            let mut request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
            request.model = Some(format!("model-{}", i));
            logger.log_request(&request).unwrap();
        }
        let (requests, _) = logger.get_requests_paginated(None, Some("model-2"), 1, 10).unwrap();
        assert!(requests.iter().all(|r| r.model.as_ref().unwrap().contains("model-2")));
    }

    #[test]
    fn test_get_requests_paginated_second_page() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        for i in 0..10 {
            let mut request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
            request.model = Some(format!("model-{}", i));
            logger.log_request(&request).unwrap();
        }
        let (requests, total_pages) = logger.get_requests_paginated(None, None, 2, 5).unwrap();
        assert_eq!(requests.len(), 5);
        assert_eq!(total_pages, 2);
    }

    #[test]
    fn test_audit_error_database_error() {
        let error = AuditError::DatabaseError("test error".to_string());
        assert!(error.to_string().contains("Database error"));
    }

    #[test]
    fn test_audit_error_io_error() {
        let error = AuditError::IoError("permission denied".to_string());
        assert!(error.to_string().contains("IO error"));
    }

    #[test]
    fn test_dashboard_stats_struct() {
        let stats = DashboardStats {
            total_users: 10,
            total_requests: 100,
            requests_24h: 25,
            total_tokens: 5000,
        };
        assert_eq!(stats.total_users, 10);
        assert_eq!(stats.total_requests, 100);
    }

    #[test]
    fn test_request_summary_struct() {
        let summary = RequestSummary {
            id: "req123".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            user_id: "user123".to_string(),
            request_path: "/v1/chat/completions".to_string(),
            model: Some("llama2".to_string()),
        };
        assert_eq!(summary.id, "req123");
        assert!(summary.model.is_some());
    }

    #[test]
    fn test_user_with_stats_struct() {
        let user = UserWithStats {
            id: "user123".to_string(),
            email: Some("user@example.com".to_string()),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            last_seen_at: "2024-01-02T00:00:00Z".to_string(),
            is_enabled: true,
            request_count: 5,
        };
        assert_eq!(user.request_count, 5);
        assert!(user.is_enabled);
    }

    #[test]
    fn test_request_with_response_struct() {
        let req_with_resp = RequestWithResponse {
            id: "req123".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            user_id: "user123".to_string(),
            request_path: "/v1/chat/completions".to_string(),
            model: Some("llama2".to_string()),
            status: Some(200),
            latency_ms: Some(150),
            tokens_prompt: Some(10),
            tokens_completion: Some(5),
        };
        assert_eq!(req_with_resp.status, Some(200));
        assert_eq!(req_with_resp.latency_ms, Some(150));
    }

    #[test]
    fn test_database_url_parsing_sqlite_prefix() {
        let _logger = create_test_logger();
        let url = "sqlite:memory";
        assert!(url.starts_with("sqlite:"));
    }

    #[test]
    fn test_multiple_log_operations() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        for i in 0..10 {
            let request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
            let request_id = logger.log_request(&request).unwrap();
            let mut response = Response::new(request_id, 200);
            response.tokens_prompt = Some(10);
            response.tokens_completion = Some(i as u32);
            logger.log_response(&response).unwrap();
        }
        let stats = logger.get_stats().unwrap();
        assert_eq!(stats.total_requests, 10);
        assert_eq!(stats.total_tokens, 145); // 10 requests * 10 prompt + (0+1+...+9) completion
    }

    #[test]
    fn test_upsert_runner_new() {
        let logger = create_test_logger();
        let models = vec!["llama3:8b".to_string(), "mistral:7b".to_string()];
        logger.upsert_runner(
            "runner-1",
            "Test Runner",
            Some("AA:BB:CC:DD:EE:FF"),
            Some("gpu-server"),
            Some(&models),
        ).unwrap();

        let runner = logger.get_runner("runner-1").unwrap().unwrap();
        assert_eq!(runner.id, "runner-1");
        assert_eq!(runner.name, "Test Runner");
        assert_eq!(runner.mac_address, Some("AA:BB:CC:DD:EE:FF".to_string()));
        assert_eq!(runner.machine_type, Some("gpu-server".to_string()));
        assert_eq!(runner.available_models, models);
    }

    #[test]
    fn test_upsert_runner_update() {
        let logger = create_test_logger();

        // Initial insert
        let models = vec!["llama3:8b".to_string()];
        logger.upsert_runner("runner-1", "Old Name", Some("AA:BB:CC:DD:EE:FF"), None, Some(&models)).unwrap();

        // Update with new name, keeps MAC address and models
        logger.upsert_runner("runner-1", "New Name", None, Some("cpu-server"), None).unwrap();

        let runner = logger.get_runner("runner-1").unwrap().unwrap();
        assert_eq!(runner.name, "New Name");
        assert_eq!(runner.mac_address, Some("AA:BB:CC:DD:EE:FF".to_string())); // Preserved
        assert_eq!(runner.machine_type, Some("cpu-server".to_string()));
        assert_eq!(runner.available_models, models); // Preserved
    }

    #[test]
    fn test_upsert_runner_without_mac() {
        let logger = create_test_logger();
        logger.upsert_runner("runner-1", "Test Runner", None, None, None).unwrap();

        let runner = logger.get_runner("runner-1").unwrap().unwrap();
        assert_eq!(runner.id, "runner-1");
        assert!(runner.mac_address.is_none());
        assert!(runner.available_models.is_empty());
    }

    #[test]
    fn test_get_runner_not_found() {
        let logger = create_test_logger();
        let result = logger.get_runner("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_all_runners_empty() {
        let logger = create_test_logger();
        let runners = logger.get_all_runners().unwrap();
        assert!(runners.is_empty());
    }

    #[test]
    fn test_get_all_runners() {
        let logger = create_test_logger();
        logger.upsert_runner("runner-1", "Runner 1", Some("AA:BB:CC:DD:EE:01"), None, None).unwrap();
        logger.upsert_runner("runner-2", "Runner 2", Some("AA:BB:CC:DD:EE:02"), None, None).unwrap();
        logger.upsert_runner("runner-3", "Runner 3", None, None, None).unwrap();

        let runners = logger.get_all_runners().unwrap();
        assert_eq!(runners.len(), 3);
    }

    #[test]
    fn test_runner_record_struct() {
        let record = RunnerRecord {
            id: "runner-1".to_string(),
            name: "Test Runner".to_string(),
            mac_address: Some("AA:BB:CC:DD:EE:FF".to_string()),
            machine_type: Some("gpu-server".to_string()),
            last_seen_at: "2024-01-01T00:00:00Z".to_string(),
            available_models: vec!["llama3:8b".to_string()],
        };
        assert_eq!(record.id, "runner-1");
        assert!(record.mac_address.is_some());
        assert_eq!(record.available_models.len(), 1);
    }

    #[test]
    fn test_get_runners_by_model_class() {
        let logger = create_test_logger();

        // Runner with big model
        let big_models = vec!["llama3:70b".to_string()];
        logger.upsert_runner("runner-big", "Big Runner", Some("AA:BB:CC:DD:EE:01"), None, Some(&big_models)).unwrap();

        // Runner with fast model
        let fast_models = vec!["llama3:8b".to_string(), "mistral:7b".to_string()];
        logger.upsert_runner("runner-fast", "Fast Runner", Some("AA:BB:CC:DD:EE:02"), None, Some(&fast_models)).unwrap();

        // Query by class with config
        use crate::gateway::ModelClass;
        use crate::config::ModelsConfig;

        let models_config = ModelsConfig {
            big: vec!["llama3:70b".to_string()],
            fast: vec!["llama3:8b".to_string(), "mistral:7b".to_string()],
        };

        let big_runners = logger.get_runners_by_model_class(ModelClass::Big, &models_config).unwrap();
        assert_eq!(big_runners.len(), 1);
        assert_eq!(big_runners[0].id, "runner-big");

        let fast_runners = logger.get_runners_by_model_class(ModelClass::Fast, &models_config).unwrap();
        assert_eq!(fast_runners.len(), 1);
        assert_eq!(fast_runners[0].id, "runner-fast");
    }

    #[test]
    fn test_get_runners_by_model() {
        let logger = create_test_logger();

        let models1 = vec!["llama3:8b".to_string(), "mistral:7b".to_string()];
        logger.upsert_runner("runner-1", "Runner 1", None, None, Some(&models1)).unwrap();

        let models2 = vec!["llama3:8b".to_string()];
        logger.upsert_runner("runner-2", "Runner 2", None, None, Some(&models2)).unwrap();

        // Both have llama3:8b
        let llama_runners = logger.get_runners_by_model("llama3:8b").unwrap();
        assert_eq!(llama_runners.len(), 2);

        // Only runner-1 has mistral
        let mistral_runners = logger.get_runners_by_model("mistral:7b").unwrap();
        assert_eq!(mistral_runners.len(), 1);
        assert_eq!(mistral_runners[0].id, "runner-1");
    }
}
