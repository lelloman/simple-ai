use chrono::Utc;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;

use crate::models::request::{Request, Response};
use crate::models::user::User;
use simple_ai_common::InferenceMetrics;

/// SQLite-based audit logger with user management.
/// Called after every response is persisted, e.g. to track runner failures.
pub type ResponseObserver = Box<dyn Fn(&Response) + Send + Sync>;

pub struct AuditLogger {
    conn: Mutex<Connection>,
    response_observer: std::sync::OnceLock<ResponseObserver>,
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

#[derive(Default)]
pub struct RequestFilters<'a> {
    pub user_id: Option<&'a str>,
    pub model: Option<&'a str>,
    pub origin: Option<&'a str>,
    pub since: Option<&'a str>,
    pub until: Option<&'a str>,
    pub snapshot: Option<i64>,
    /// Only requests whose response status is 400 or above.
    pub failed_only: bool,
}

/// Request counts and token usage for one model or origin within a summary window.
#[derive(Debug, serde::Serialize)]
pub struct UsageRow {
    pub label: String,
    /// "model" for model rows; "app", "key", "user" or "ip" for origin rows.
    pub kind: String,
    pub requests: u64,
    pub failed: u64,
    pub tokens: u64,
}

/// Aggregate activity since a point in time, for the dashboard.
#[derive(Debug, serde::Serialize)]
pub struct ActivitySummary {
    pub requests: u64,
    pub completed: u64,
    pub failed: u64,
    pub tokens_prompt: u64,
    pub tokens_completion: u64,
    /// Latency percentiles over successful (status < 400) requests.
    pub latency_p50_ms: Option<i64>,
    pub latency_p95_ms: Option<i64>,
    pub top_models: Vec<UsageRow>,
    pub top_origins: Vec<UsageRow>,
}

#[derive(serde::Serialize)]
pub struct RequestBodies {
    pub request_body: Option<String>,
    pub response_body: Option<String>,
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
            std::fs::create_dir_all(parent).map_err(|e| AuditError::IoError(e.to_string()))?;
        }

        let conn = Connection::open(path).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Create users table
        super::schema::create_table(&conn, "users")?;

        // Create requests table
        super::schema::create_table(&conn, "requests")?;

        let _ = conn.execute("ALTER TABLE requests ADD COLUMN source_app TEXT", []);

        // Migration: add client_ip column if it doesn't exist (for existing databases)
        let _ = conn.execute("ALTER TABLE requests ADD COLUMN client_ip TEXT", []);

        // Create responses table
        super::schema::create_table(&conn, "responses")?;

        // Migration: add runner_id and wol_sent columns to responses
        let _ = conn.execute("ALTER TABLE responses ADD COLUMN runner_id TEXT", []);
        let _ = conn.execute(
            "ALTER TABLE responses ADD COLUMN wol_sent INTEGER NOT NULL DEFAULT 0",
            [],
        );

        // Migration: add tokens_prompt and tokens_completion columns to responses (for existing databases)
        let _ = conn.execute("ALTER TABLE responses ADD COLUMN tokens_prompt INTEGER", []);
        let _ = conn.execute(
            "ALTER TABLE responses ADD COLUMN tokens_completion INTEGER",
            [],
        );

        // Create API keys table
        super::schema::create_table(&conn, "api_keys")?;

        // Migration: store retrievable API key secrets for keys created after this migration.
        let _ = conn.execute("ALTER TABLE api_keys ADD COLUMN plaintext_key TEXT", []);
        // Migration: existing keys remain class-only until an administrator
        // explicitly grants additional roles.
        let _ = conn.execute(
            "ALTER TABLE api_keys ADD COLUMN roles TEXT NOT NULL DEFAULT '[]'",
            [],
        );

        // Create indexes
        super::schema::create_index(&conn, "requests", "idx_requests_timestamp", "timestamp")?;

        super::schema::create_index(&conn, "requests", "idx_requests_user_id", "user_id")?;

        super::schema::create_index(&conn, "responses", "idx_responses_request_id", "request_id")?;

        // Create runners table for persistent runner tracking
        super::schema::create_table(&conn, "runners")?;

        // Migration: add available_models column if it doesn't exist
        let _ = conn.execute("ALTER TABLE runners ADD COLUMN available_models TEXT", []); // Ignore error if column already exists

        // Migration: add model_class column to responses
        let _ = conn.execute("ALTER TABLE responses ADD COLUMN model_class TEXT", []);

        // Create runner_metrics table for tracking boot time and inference latency
        super::schema::create_table(&conn, "runner_metrics")?;

        super::schema::create_table(&conn, "response_inference_metrics")?;

        super::schema::create_table(&conn, "model_context_metrics")?;

        super::schema::migrate_request_attribution(&conn)?;

        tracing::info!("Audit logger initialized with database: {}", path);

        Ok(Self {
            conn: Mutex::new(conn),
            response_observer: std::sync::OnceLock::new(),
        })
    }

    /// Find or create a user. Returns the user.
    pub fn find_or_create_user(
        &self,
        user_id: &str,
        email: Option<&str>,
    ) -> Result<User, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let now = Utc::now();

        // Try to find existing user
        let existing: Option<(String, Option<String>, String, String, bool)> = conn
            .query_row(
                "SELECT id, email, created_at, last_seen_at, is_enabled FROM users WHERE id = ?1",
                params![user_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get::<_, i32>(4)? != 0,
                    ))
                },
            )
            .ok();

        match existing {
            Some((id, db_email, created_at, _, is_enabled)) => {
                // Update last_seen_at and email if changed
                conn.execute(
                    "UPDATE users SET last_seen_at = ?1, email = COALESCE(?2, email) WHERE id = ?3",
                    params![now.to_rfc3339(), email, user_id],
                )
                .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

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

                tracing::info!(
                    "Created new user: {} ({})",
                    user_id,
                    email.unwrap_or("no email")
                );

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
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "INSERT INTO requests (id, timestamp, user_id, request_path, request_body, model, client_ip, source_app, auth_method, api_key_id, api_key_name, user_agent, peer_ip, proxy_request_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                request.id,
                request.timestamp.to_rfc3339(),
                request.user_id,
                request.request_path,
                request.request_body,
                request.model,
                request.client_ip,
                request.source_app,
                request.auth_method,
                request.api_key_id,
                request.api_key_name,
                request.user_agent,
                request.peer_ip,
                request.proxy_request_id,

            ],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        tracing::debug!("Logged request: {}", request.id);
        Ok(request.id.clone())
    }

    /// Log a response (after getting LLM result).
    /// Install the response observer. Only the first call takes effect.
    pub fn set_response_observer(&self, observer: ResponseObserver) {
        let _ = self.response_observer.set(observer);
    }

    pub fn log_response(&self, response: &Response) -> Result<(), AuditError> {
        if let Some(observer) = self.response_observer.get() {
            observer(response);
        }
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "INSERT INTO responses (id, request_id, timestamp, status, response_body, latency_ms, tokens_prompt, tokens_completion, runner_id, wol_sent, model_class)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                response.id,
                response.request_id,
                response.timestamp.to_rfc3339(),
                response.status,
                response.response_body,
                response.latency_ms as i64,
                response.tokens_prompt.map(|v| v as i64),
                response.tokens_completion.map(|v| v as i64),
                response.runner_id,
                response.wol_sent as i32,
                response.model_class,
            ],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        drop(conn);

        // Record inference metrics if we have runner_id and model_class
        if let (Some(ref runner_id), Some(ref model_class)) =
            (&response.runner_id, &response.model_class)
        {
            let _ = self.record_metric(runner_id, model_class, response.latency_ms);
        }

        if let Some(ref metrics) = response.inference_metrics {
            let _ = self.record_inference_metrics(response, metrics);
        }

        tracing::debug!("Logged response for request: {}", response.request_id);
        Ok(())
    }

    pub fn record_inference_metrics(
        &self,
        response: &Response,
        metrics: &InferenceMetrics,
    ) -> Result<(), AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let requested_model: Option<String> = conn
            .query_row(
                "SELECT model FROM requests WHERE id = ?1",
                params![response.request_id],
                |row| row.get(0),
            )
            .ok();

        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR REPLACE INTO response_inference_metrics (
                response_id, request_id, runner_id, model_class, requested_model,
                resolved_model, engine_type, context_window, prompt_tokens,
                completion_tokens, prompt_eval_ms, completion_eval_ms,
                total_inference_ms, prompt_tokens_per_sec, completion_tokens_per_sec,
                created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                response.id.as_str(),
                response.request_id.as_str(),
                response.runner_id.as_deref(),
                response.model_class.as_deref(),
                requested_model,
                metrics.resolved_model.as_deref(),
                metrics.engine_type.as_deref(),
                metrics.context_window.map(|v| v as i64),
                metrics.prompt_tokens.map(|v| v as i64),
                metrics.completion_tokens.map(|v| v as i64),
                metrics.prompt_eval_ms.map(|v| v as i64),
                metrics.completion_eval_ms.map(|v| v as i64),
                metrics.total_inference_ms.map(|v| v as i64),
                metrics.prompt_tokens_per_sec,
                metrics.completion_tokens_per_sec,
                now,
            ],
        )
        .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        if let (Some(resolved_model), Some(runner_id), Some(context_window)) = (
            metrics.resolved_model.as_deref(),
            response.runner_id.as_deref(),
            metrics.context_window,
        ) {
            Self::record_model_context_metric(
                &conn,
                resolved_model,
                runner_id,
                context_window,
                metrics,
            )?;
        }

        Ok(())
    }

    fn record_model_context_metric(
        conn: &Connection,
        resolved_model: &str,
        runner_id: &str,
        context_window: u32,
        metrics: &InferenceMetrics,
    ) -> Result<(), AuditError> {
        let now = Utc::now().to_rfc3339();
        let prompt_tokens = metrics.prompt_tokens.unwrap_or(0) as i64;
        let completion_tokens = metrics.completion_tokens.unwrap_or(0) as i64;
        let prompt_weighted = metrics
            .prompt_tokens_per_sec
            .map(|tps| tps * prompt_tokens as f64)
            .unwrap_or(0.0);
        let completion_weighted = metrics
            .completion_tokens_per_sec
            .map(|tps| tps * completion_tokens as f64)
            .unwrap_or(0.0);

        conn.execute(
            "INSERT INTO model_context_metrics (
                resolved_model, runner_id, context_window, sample_count,
                prompt_tokens_total, completion_tokens_total,
                prompt_weighted_tps_total, completion_weighted_tps_total,
                prompt_tps_min, prompt_tps_max, completion_tps_min, completion_tps_max,
                last_updated_at
             ) VALUES (?1, ?2, ?3, 1, ?4, ?5, ?6, ?7, ?8, ?8, ?9, ?9, ?10)
             ON CONFLICT(resolved_model, runner_id, context_window) DO UPDATE SET
                sample_count = sample_count + 1,
                prompt_tokens_total = prompt_tokens_total + ?4,
                completion_tokens_total = completion_tokens_total + ?5,
                prompt_weighted_tps_total = prompt_weighted_tps_total + ?6,
                completion_weighted_tps_total = completion_weighted_tps_total + ?7,
                prompt_tps_min = CASE
                    WHEN ?8 IS NULL THEN prompt_tps_min
                    WHEN prompt_tps_min IS NULL THEN ?8
                    ELSE MIN(prompt_tps_min, ?8)
                END,
                prompt_tps_max = CASE
                    WHEN ?8 IS NULL THEN prompt_tps_max
                    WHEN prompt_tps_max IS NULL THEN ?8
                    ELSE MAX(prompt_tps_max, ?8)
                END,
                completion_tps_min = CASE
                    WHEN ?9 IS NULL THEN completion_tps_min
                    WHEN completion_tps_min IS NULL THEN ?9
                    ELSE MIN(completion_tps_min, ?9)
                END,
                completion_tps_max = CASE
                    WHEN ?9 IS NULL THEN completion_tps_max
                    WHEN completion_tps_max IS NULL THEN ?9
                    ELSE MAX(completion_tps_max, ?9)
                END,
                last_updated_at = ?10",
            params![
                resolved_model,
                runner_id,
                context_window as i64,
                prompt_tokens,
                completion_tokens,
                prompt_weighted,
                completion_weighted,
                metrics.prompt_tokens_per_sec,
                metrics.completion_tokens_per_sec,
                now,
            ],
        )
        .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// Record a metric sample (boot time or inference latency).
    pub fn record_metric(
        &self,
        runner_id: &str,
        model_class: &str,
        ms: u64,
    ) -> Result<(), AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO runner_metrics (runner_id, model_class, sample_count, total_ms, min_ms, max_ms, last_updated_at)
             VALUES (?1, ?2, 1, ?3, ?3, ?3, ?4)
             ON CONFLICT(runner_id, model_class) DO UPDATE SET
                sample_count = sample_count + 1,
                total_ms = total_ms + ?3,
                min_ms = MIN(min_ms, ?3),
                max_ms = MAX(max_ms, ?3),
                last_updated_at = ?4",
            params![runner_id, model_class, ms as i64, now],
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        tracing::debug!(
            "Recorded {} metric for runner {}: {}ms",
            model_class,
            runner_id,
            ms
        );
        Ok(())
    }

    // ========== Admin queries ==========

    /// Get dashboard statistics.
    pub fn get_stats(&self) -> Result<DashboardStats, AuditError> {
        let conn = self
            .conn
            .lock()
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

        // Token counts
        let (tokens_prompt, tokens_completion): (i64, i64) = conn
            .query_row(
                "SELECT COALESCE(SUM(COALESCE(tokens_prompt, 0)), 0), COALESCE(SUM(COALESCE(tokens_completion, 0)), 0) FROM responses",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap_or((0, 0));

        let total_tokens = tokens_prompt + tokens_completion;

        Ok(DashboardStats {
            total_users: total_users as u64,
            total_requests: total_requests as u64,
            requests_24h: requests_24h as u64,
            total_tokens: total_tokens as u64,
            tokens_prompt: tokens_prompt as u64,
            tokens_completion: tokens_completion as u64,
        })
    }

    /// Get recent requests for dashboard.
    pub fn get_recent_requests(&self, limit: u32) -> Result<Vec<RequestSummary>, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT r.id, r.timestamp, r.user_id, r.request_path, r.model
             FROM requests r
             ORDER BY r.timestamp DESC, r.rowid DESC
             LIMIT ?1",
            )
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt
            .query_map(params![limit], |row| {
                Ok(RequestSummary {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    user_id: row.get(2)?,
                    request_path: row.get(3)?,
                    model: row.get(4)?,
                })
            })
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut requests = Vec::new();
        for row in rows {
            requests.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }
        Ok(requests)
    }

    /// Get all users with request counts.
    pub fn get_users_with_stats(&self) -> Result<Vec<UserWithStats>, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT u.id, u.email, u.created_at, u.last_seen_at, u.is_enabled,
                    (SELECT COUNT(*) FROM requests r WHERE r.user_id = u.id) as request_count
             FROM users u
             ORDER BY u.last_seen_at DESC",
            )
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(UserWithStats {
                    id: row.get(0)?,
                    email: row.get(1)?,
                    created_at: row.get(2)?,
                    last_seen_at: row.get(3)?,
                    is_enabled: row.get::<_, i32>(4)? != 0,
                    request_count: row.get(5)?,
                })
            })
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

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
        self.get_request_history(
            &RequestFilters {
                user_id,
                model,
                ..Default::default()
            },
            page,
            per_page,
        )
        .map(|(requests, pages, _)| (requests, pages))
    }

    pub(super) fn save_stream_body(&self, id: &str, body: &str) -> Result<(), AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
        conn.execute(
            "UPDATE requests SET stream_body = ? WHERE id = ?",
            params![body, id],
        )
        .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub fn get_request_bodies(&self, id: &str) -> Result<Option<RequestBodies>, AuditError> {
        use rusqlite::OptionalExtension;
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
        conn.query_row(
            "SELECT r.request_body, COALESCE(r.stream_body, resp.response_body) FROM requests r LEFT JOIN responses resp ON resp.request_id = r.id WHERE r.id = ?",
            [id], |row| Ok(RequestBodies { request_body: row.get(0)?, response_body: row.get(1)? })
        ).optional().map_err(|e| AuditError::DatabaseError(e.to_string()))
    }

    pub fn get_request_history(
        &self,
        filters: &RequestFilters<'_>,
        page: u32,
        per_page: u32,
    ) -> Result<(Vec<RequestWithResponse>, u32, i64), AuditError> {
        let per_page = per_page.clamp(1, 100);
        let user_id = filters.user_id;
        let model = filters.model;
        let conn = self
            .conn
            .lock()
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

        let snapshot = match filters.snapshot {
            Some(value) => value,
            None => conn
                .query_row("SELECT COALESCE(MAX(rowid), 0) FROM requests", [], |row| {
                    row.get(0)
                })
                .map_err(|e| AuditError::DatabaseError(e.to_string()))?,
        };
        where_clauses.push("r.rowid <= ?");
        params_vec.push(Box::new(snapshot));
        if filters.failed_only {
            where_clauses.push("resp.status >= 400");
        }
        if let Some(origin) = filters.origin.filter(|value| !value.is_empty()) {
            where_clauses.push("(instr(lower(COALESCE(r.source_app,'')), lower(?)) > 0 OR instr(lower(COALESCE(r.client_ip,'')), lower(?)) > 0 OR instr(lower(COALESCE(r.api_key_name,'')), lower(?)) > 0 OR instr(lower(COALESCE(r.user_agent,'')), lower(?)) > 0)");
            for _ in 0..4 {
                params_vec.push(Box::new(origin.to_owned()));
            }
        }
        for (value, clause) in [
            (filters.since, "julianday(r.timestamp) >= julianday(?)"),
            (filters.until, "julianday(r.timestamp) <= julianday(?)"),
        ] {
            if let Some(value) = value {
                where_clauses.push(clause);
                params_vec.push(Box::new(value.to_owned()));
            }
        }

        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        // Get total count
        let count_sql = format!(
            "SELECT COUNT(*) FROM requests r LEFT JOIN responses resp ON resp.request_id = r.id {}",
            where_clause
        );
        let total: i64 = {
            let mut stmt = conn
                .prepare(&count_sql)
                .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
            let params_refs: Vec<&dyn rusqlite::ToSql> =
                params_vec.iter().map(|p| p.as_ref()).collect();
            stmt.query_row(params_refs.as_slice(), |row| row.get(0))
                .map_err(|e| AuditError::DatabaseError(e.to_string()))?
        };

        let total_pages = ((total as u32) + per_page - 1) / per_page;
        let offset = u64::from(page.saturating_sub(1)) * u64::from(per_page);

        // Get requests with user email
        let query_sql = format!(
            "SELECT r.id, r.timestamp, r.user_id, u.email, r.request_path, r.model, r.client_ip,
                    resp.status, resp.latency_ms, resp.tokens_prompt, resp.tokens_completion,
                    resp.runner_id, COALESCE(resp.wol_sent, 0), r.source_app,
                    r.auth_method, r.api_key_id, r.api_key_name, r.user_agent, r.peer_ip, r.proxy_request_id
             FROM requests r
             LEFT JOIN users u ON u.id = r.user_id
             LEFT JOIN responses resp ON resp.request_id = r.id
             {}
             ORDER BY r.timestamp DESC, r.rowid DESC
             LIMIT ? OFFSET ?",
            where_clause
        );

        params_vec.push(Box::new(per_page as i64));
        params_vec.push(Box::new(offset as i64));

        let mut stmt = conn
            .prepare(&query_sql)
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(params_refs.as_slice(), |row| {
                Ok(RequestWithResponse {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    user_id: row.get(2)?,
                    user_email: row.get(3)?,
                    request_path: row.get(4)?,
                    model: row.get(5)?,
                    client_ip: row.get(6)?,
                    status: row.get(7)?,
                    latency_ms: row.get(8)?,
                    tokens_prompt: row.get(9)?,
                    tokens_completion: row.get(10)?,
                    runner_id: row.get(11)?,
                    wol_sent: row.get::<_, i32>(12)? != 0,
                    source_app: row.get(13)?,
                    auth_method: row.get(14)?,
                    api_key_id: row.get(15)?,
                    api_key_name: row.get(16)?,
                    user_agent: row.get(17)?,
                    peer_ip: row.get(18)?,
                    proxy_request_id: row.get(19)?,
                })
            })
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut requests = Vec::new();
        for row in rows {
            requests.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }

        Ok((requests, total_pages, snapshot))
    }

    /// Summarise requests at or after `since` (RFC 3339), with the `top` busiest models and origins.
    pub fn get_activity_summary(
        &self,
        since: &str,
        top: u32,
    ) -> Result<ActivitySummary, AuditError> {
        let db_err = |e: rusqlite::Error| AuditError::DatabaseError(e.to_string());
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
        const WINDOW: &str = "FROM requests r
             LEFT JOIN responses resp ON resp.request_id = r.id
             LEFT JOIN users u ON u.id = r.user_id
             WHERE julianday(r.timestamp) >= julianday(?1)";

        let (requests, completed, failed, tokens_prompt, tokens_completion): (i64, i64, i64, i64, i64) = conn
            .query_row(
                &format!(
                    "SELECT COUNT(*), COUNT(resp.status), COALESCE(SUM(resp.status >= 400), 0),
                            COALESCE(SUM(resp.tokens_prompt), 0), COALESCE(SUM(resp.tokens_completion), 0)
                     {WINDOW}"
                ),
                params![since],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .map_err(db_err)?;

        let latencies = conn
            .prepare(&format!(
                "SELECT resp.latency_ms {WINDOW} AND resp.status < 400 ORDER BY resp.latency_ms"
            ))
            .map_err(db_err)?
            .query_map(params![since], |row| row.get::<_, i64>(0))
            .map_err(db_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(db_err)?;
        // Nearest-rank percentile over the sorted latencies.
        let percentile = |p: usize| {
            (!latencies.is_empty())
                .then(|| latencies[((latencies.len() * p).div_ceil(100)).max(1) - 1])
        };

        let usage = |kind_sql: &str, label_sql: &str| -> Result<Vec<UsageRow>, AuditError> {
            conn.prepare(&format!(
                "SELECT {kind_sql} AS kind, {label_sql} AS label, COUNT(*) AS n,
                        COALESCE(SUM(resp.status >= 400), 0),
                        COALESCE(SUM(COALESCE(resp.tokens_prompt, 0) + COALESCE(resp.tokens_completion, 0)), 0)
                 {WINDOW}
                 GROUP BY kind, label ORDER BY n DESC, label LIMIT ?2"
            ))
            .map_err(db_err)?
            .query_map(params![since, top], |row| {
                Ok(UsageRow {
                    kind: row.get(0)?,
                    label: row.get(1)?,
                    requests: row.get::<_, i64>(2)? as u64,
                    failed: row.get::<_, i64>(3)? as u64,
                    tokens: row.get::<_, i64>(4)? as u64,
                })
            })
            .map_err(db_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(db_err)
        };
        // An origin is the most specific attribution available for the request.
        let top_origins = usage(
            "CASE WHEN COALESCE(r.source_app, '') != '' THEN 'app'
                  WHEN COALESCE(r.api_key_name, '') != '' THEN 'key'
                  WHEN COALESCE(u.email, '') != '' THEN 'user'
                  ELSE 'ip' END",
            "COALESCE(NULLIF(r.source_app, ''), NULLIF(r.api_key_name, ''), NULLIF(u.email, ''), r.client_ip, 'unknown')",
        )?;
        let top_models = usage("'model'", "COALESCE(NULLIF(r.model, ''), r.request_path)")?;

        Ok(ActivitySummary {
            requests: requests as u64,
            completed: completed as u64,
            failed: failed as u64,
            tokens_prompt: tokens_prompt as u64,
            tokens_completion: tokens_completion as u64,
            latency_p50_ms: percentile(50),
            latency_p95_ms: percentile(95),
            top_models,
            top_origins,
        })
    }

    /// Enable a user.
    pub fn enable_user(&self, user_id: &str) -> Result<(), AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "UPDATE users SET is_enabled = 1 WHERE id = ?1",
            params![user_id],
        )
        .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// Disable a user.
    pub fn disable_user(&self, user_id: &str) -> Result<(), AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "UPDATE users SET is_enabled = 0 WHERE id = ?1",
            params![user_id],
        )
        .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    // ==================== API Key Management ====================

    /// Create a new API key. Returns the plaintext key (only shown once).
    pub fn create_api_key(
        &self,
        user_id: &str,
        name: &str,
        roles: &[String],
    ) -> Result<(ApiKey, String), AuditError> {
        use sha2::{Digest, Sha256};

        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Generate a random key: sk-<32 hex chars>
        let raw_key: [u8; 16] = rand::random();
        let plaintext_key = format!("sk-{}", hex::encode(raw_key));

        // Hash the key for storage
        let mut hasher = Sha256::new();
        hasher.update(plaintext_key.as_bytes());
        let key_hash = hex::encode(hasher.finalize());

        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let roles_json =
            serde_json::to_string(roles).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        conn.execute(
            "INSERT INTO api_keys (id, key_hash, plaintext_key, user_id, name, roles, created_at, revoked)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0)",
            params![id, key_hash, plaintext_key, user_id, name, roles_json, now],
        )
        .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Get user email for the response
        let user_email: Option<String> = conn
            .query_row(
                "SELECT email FROM users WHERE id = ?1",
                params![user_id],
                |row| row.get(0),
            )
            .ok();

        let api_key = ApiKey {
            id,
            key_hash,
            user_id: user_id.to_string(),
            user_email,
            name: name.to_string(),
            roles: roles.to_vec(),
            created_at: now,
            last_used_at: None,
            revoked: false,
            secret_available: true,
        };

        Ok((api_key, plaintext_key))
    }

    /// Validate an API key and return the associated user info.
    /// Updates last_used_at on successful validation.
    pub fn validate_api_key(
        &self,
        plaintext_key: &str,
    ) -> Result<Option<(String, Option<String>, Vec<String>)>, AuditError> {
        Ok(self
            .validate_api_key_with_identity(plaintext_key)?
            .map(|key| (key.user_id, key.email, key.roles)))
    }

    /// Validate and capture the key identity from the same authenticated record.
    pub fn validate_api_key_with_identity(
        &self,
        plaintext_key: &str,
    ) -> Result<Option<ValidatedApiKey>, AuditError> {
        use sha2::{Digest, Sha256};

        if !plaintext_key.starts_with("sk-") {
            return Ok(None);
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        // Hash the provided key
        let mut hasher = Sha256::new();
        hasher.update(plaintext_key.as_bytes());
        let key_hash = hex::encode(hasher.finalize());

        // Look up the key
        let result: Result<(String, String, Option<String>, String, String), _> = conn.query_row(
            "SELECT ak.id, ak.user_id, u.email, ak.roles, ak.name
             FROM api_keys ak
             LEFT JOIN users u ON u.id = ak.user_id
             WHERE ak.key_hash = ?1 AND ak.revoked = 0",
            params![key_hash],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        );

        match result {
            Ok((key_id, user_id, email, roles_json, key_name)) => {
                let roles = serde_json::from_str(&roles_json)
                    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
                // Update last_used_at
                let now = chrono::Utc::now().to_rfc3339();
                let _ = conn.execute(
                    "UPDATE api_keys SET last_used_at = ?1 WHERE id = ?2",
                    params![now, key_id],
                );
                Ok(Some(ValidatedApiKey {
                    user_id,
                    email,
                    roles,
                    key_id,
                    key_name,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AuditError::DatabaseError(e.to_string())),
        }
    }

    /// List all API keys (for admin).
    pub fn list_api_keys(&self) -> Result<Vec<ApiKey>, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT ak.id, ak.key_hash, ak.user_id, u.email, ak.name, ak.roles, ak.created_at, ak.last_used_at, ak.revoked, ak.plaintext_key IS NOT NULL
             FROM api_keys ak
             LEFT JOIN users u ON u.id = ak.user_id
             ORDER BY ak.created_at DESC"
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                let roles_json: String = row.get(5)?;
                let roles = serde_json::from_str(&roles_json).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        5,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?;
                Ok(ApiKey {
                    id: row.get(0)?,
                    key_hash: row.get(1)?,
                    user_id: row.get(2)?,
                    user_email: row.get(3)?,
                    name: row.get(4)?,
                    roles,
                    created_at: row.get(6)?,
                    last_used_at: row.get(7)?,
                    revoked: row.get::<_, i32>(8)? != 0,
                    secret_available: row.get::<_, i32>(9)? != 0,
                })
            })
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut keys = Vec::new();
        for row in rows {
            keys.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }

        Ok(keys)
    }

    /// Replace the roles assigned to an active API key.
    pub fn update_api_key_roles(&self, key_id: &str, roles: &[String]) -> Result<bool, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
        let roles_json =
            serde_json::to_string(roles).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows_affected = conn
            .execute(
                "UPDATE api_keys SET roles = ?1 WHERE id = ?2 AND revoked = 0",
                params![roles_json, key_id],
            )
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        Ok(rows_affected > 0)
    }

    /// Get the stored plaintext secret for an active API key, if available.
    pub fn get_api_key_secret(&self, key_id: &str) -> Result<Option<String>, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        match conn.query_row(
            "SELECT plaintext_key FROM api_keys WHERE id = ?1 AND revoked = 0",
            params![key_id],
            |row| row.get(0),
        ) {
            Ok(secret) => Ok(secret),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AuditError::DatabaseError(e.to_string())),
        }
    }

    /// Revoke an API key.
    pub fn revoke_api_key(&self, key_id: &str) -> Result<bool, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows_affected = conn
            .execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ?1",
                params![key_id],
            )
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        Ok(rows_affected > 0)
    }
}

/// Dashboard statistics.
#[derive(Debug, Clone)]
pub struct DashboardStats {
    pub total_users: u64,
    pub total_requests: u64,
    pub requests_24h: u64,
    pub total_tokens: u64,
    pub tokens_prompt: u64,
    pub tokens_completion: u64,
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
    pub user_email: Option<String>,
    pub request_path: String,
    pub model: Option<String>,
    pub client_ip: Option<String>,
    pub source_app: Option<String>,
    pub auth_method: Option<String>,
    pub api_key_id: Option<String>,
    pub api_key_name: Option<String>,
    pub user_agent: Option<String>,
    pub peer_ip: Option<String>,
    pub proxy_request_id: Option<String>,

    pub status: Option<i32>,
    pub latency_ms: Option<i64>,
    pub tokens_prompt: Option<i64>,
    pub tokens_completion: Option<i64>,
    pub runner_id: Option<String>,
    pub wol_sent: bool,
}

/// Identity from a successfully validated API key, never credential material.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ValidatedApiKey {
    pub user_id: String,
    pub email: Option<String>,
    pub roles: Vec<String>,
    pub key_id: String,
    pub key_name: String,
}

/// API key for programmatic access.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ApiKey {
    pub id: String,
    #[serde(skip_serializing)]
    pub key_hash: String,
    pub user_id: String,
    pub user_email: Option<String>,
    pub name: String,
    pub roles: Vec<String>,
    pub created_at: String,
    pub last_used_at: Option<String>,
    pub revoked: bool,
    pub secret_available: bool,
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

/// Runner metrics for boot time or inference latency.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunnerMetricRow {
    pub runner_id: String,
    pub model_class: String,
    pub sample_count: u64,
    pub total_ms: u64,
    pub min_ms: Option<u64>,
    pub max_ms: Option<u64>,
    pub last_updated_at: String,
}

/// Aggregate inference speed metrics per model, runner, and context window.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelContextMetricRow {
    pub resolved_model: String,
    pub runner_id: String,
    pub context_window: u32,
    pub sample_count: u64,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    pub avg_prompt_tokens_per_sec: Option<f64>,
    pub avg_completion_tokens_per_sec: Option<f64>,
    pub prompt_tps_min: Option<f64>,
    pub prompt_tps_max: Option<f64>,
    pub completion_tps_min: Option<f64>,
    pub completion_tps_max: Option<f64>,
    pub last_updated_at: String,
}

impl RunnerMetricRow {
    /// Calculate the average latency in milliseconds.
    pub fn avg_ms(&self) -> Option<f64> {
        if self.sample_count > 0 {
            Some(self.total_ms as f64 / self.sample_count as f64)
        } else {
            None
        }
    }
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
        let conn = self
            .conn
            .lock()
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

        tracing::debug!(
            "Upserted runner: {} with {} models",
            id,
            available_models.map(|m| m.len()).unwrap_or(0)
        );
        Ok(())
    }

    /// Get a runner by ID.
    pub fn get_runner(&self, id: &str) -> Result<Option<RunnerRecord>, AuditError> {
        let conn = self
            .conn
            .lock()
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
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT id, name, mac_address, machine_type, last_seen_at, available_models FROM runners ORDER BY last_seen_at DESC"
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
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
            })
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

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

    /// Get all metrics for a specific runner.
    pub fn get_runner_metrics(&self, runner_id: &str) -> Result<Vec<RunnerMetricRow>, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT runner_id, model_class, sample_count, total_ms, min_ms, max_ms, last_updated_at
             FROM runner_metrics WHERE runner_id = ?1 ORDER BY model_class"
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt
            .query_map(params![runner_id], |row| {
                Ok(RunnerMetricRow {
                    runner_id: row.get(0)?,
                    model_class: row.get(1)?,
                    sample_count: row.get::<_, i64>(2)? as u64,
                    total_ms: row.get::<_, i64>(3)? as u64,
                    min_ms: row.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                    max_ms: row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
                    last_updated_at: row.get(6)?,
                })
            })
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut metrics = Vec::new();
        for row in rows {
            metrics.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }
        Ok(metrics)
    }

    /// Get all metrics across all runners.
    pub fn get_all_metrics(&self) -> Result<Vec<RunnerMetricRow>, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT runner_id, model_class, sample_count, total_ms, min_ms, max_ms, last_updated_at
             FROM runner_metrics ORDER BY runner_id, model_class"
        ).map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(RunnerMetricRow {
                    runner_id: row.get(0)?,
                    model_class: row.get(1)?,
                    sample_count: row.get::<_, i64>(2)? as u64,
                    total_ms: row.get::<_, i64>(3)? as u64,
                    min_ms: row.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                    max_ms: row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
                    last_updated_at: row.get(6)?,
                })
            })
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut metrics = Vec::new();
        for row in rows {
            metrics.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }
        Ok(metrics)
    }

    /// Get aggregate model speed metrics.
    pub fn get_model_context_metrics(
        &self,
        limit: u32,
    ) -> Result<Vec<ModelContextMetricRow>, AuditError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT
                resolved_model,
                runner_id,
                context_window,
                sample_count,
                prompt_tokens_total,
                completion_tokens_total,
                CASE
                    WHEN prompt_tokens_total > 0 THEN prompt_weighted_tps_total / prompt_tokens_total
                    ELSE NULL
                END AS avg_prompt_tps,
                CASE
                    WHEN completion_tokens_total > 0 THEN completion_weighted_tps_total / completion_tokens_total
                    ELSE NULL
                END AS avg_completion_tps,
                prompt_tps_min,
                prompt_tps_max,
                completion_tps_min,
                completion_tps_max,
                last_updated_at
             FROM model_context_metrics
             ORDER BY last_updated_at DESC
             LIMIT ?1",
        )
        .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let rows = stmt
            .query_map(params![limit.min(500) as i64], |row| {
                Ok(ModelContextMetricRow {
                    resolved_model: row.get(0)?,
                    runner_id: row.get(1)?,
                    context_window: row.get::<_, i64>(2)? as u32,
                    sample_count: row.get::<_, i64>(3)? as u64,
                    prompt_tokens_total: row.get::<_, i64>(4)? as u64,
                    completion_tokens_total: row.get::<_, i64>(5)? as u64,
                    avg_prompt_tokens_per_sec: row.get(6)?,
                    avg_completion_tokens_per_sec: row.get(7)?,
                    prompt_tps_min: row.get(8)?,
                    prompt_tps_max: row.get(9)?,
                    completion_tps_min: row.get(10)?,
                    completion_tps_max: row.get(11)?,
                    last_updated_at: row.get(12)?,
                })
            })
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

        let mut metrics = Vec::new();
        for row in rows {
            metrics.push(row.map_err(|e| AuditError::DatabaseError(e.to_string()))?);
        }
        Ok(metrics)
    }

    /// Get boot time metrics for a specific runner.
    pub fn get_boot_metrics(&self, runner_id: &str) -> Result<Option<RunnerMetricRow>, AuditError> {
        let metrics = self.get_runner_metrics(runner_id)?;
        Ok(metrics.into_iter().find(|m| m.model_class == "boot"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_test_logger() -> AuditLogger {
        let test_db_path = ":memory:";
        AuditLogger::new(&test_db_path).unwrap()
    }

    #[test]
    fn test_new_creates_tables() {
        let dir = tempfile::tempdir().unwrap();
        let test_db_path = dir.path().join("audit.db").to_str().unwrap().to_string();
        let logger = AuditLogger::new(&test_db_path).unwrap();
        drop(logger);
        assert!(fs::metadata(&test_db_path).is_ok());
    }

    #[test]
    fn test_existing_api_keys_table_gets_roles_migration() {
        let dir = tempfile::tempdir().unwrap();
        let test_db_path = dir.path().join("audit.db").to_str().unwrap().to_string();
        let conn = Connection::open(&test_db_path).unwrap();
        conn.execute(
            "CREATE TABLE api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                plaintext_key TEXT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                revoked INTEGER NOT NULL DEFAULT 0
            )",
            [],
        )
        .unwrap();
        drop(conn);

        let logger = AuditLogger::new(&test_db_path).unwrap();
        let conn = logger.conn.lock().unwrap();
        let roles: String = conn
            .query_row(
                "SELECT dflt_value FROM pragma_table_info('api_keys') WHERE name = 'roles'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(roles, "'[]'");
        drop(conn);
        drop(logger);
    }

    #[test]
    fn test_find_or_create_user_new() {
        let logger = create_test_logger();
        let user = logger
            .find_or_create_user("user123", Some("user@example.com"))
            .unwrap();
        assert_eq!(user.id, "user123");
        assert_eq!(user.email, Some("user@example.com".to_string()));
        assert!(user.is_enabled);
    }

    #[test]
    fn test_find_or_create_user_existing() {
        let logger = create_test_logger();
        let user1 = logger
            .find_or_create_user("user123", Some("user@example.com"))
            .unwrap();
        let user2 = logger
            .find_or_create_user("user123", Some("user2@example.com"))
            .unwrap();
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
    fn test_log_response_records_inference_metrics() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let mut request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
        request.model = Some("class:big".to_string());
        let request_id = logger.log_request(&request).unwrap();

        let mut response = Response::new(request_id, 200);
        response.runner_id = Some("runner-1".to_string());
        response.model_class = Some("big".to_string());
        response.inference_metrics = Some(
            InferenceMetrics {
                resolved_model: Some("qwen3.5:35b".to_string()),
                engine_type: Some("llama_cpp".to_string()),
                context_window: Some(8192),
                prompt_tokens: Some(100),
                completion_tokens: Some(50),
                prompt_eval_ms: Some(1_000),
                completion_eval_ms: Some(2_000),
                total_inference_ms: Some(3_000),
                ..Default::default()
            }
            .with_computed_rates(),
        );

        logger.log_response(&response).unwrap();

        let conn = logger.conn.lock().unwrap();
        let row: (String, String, i64, f64, f64) = conn
            .query_row(
                "SELECT resolved_model, engine_type, context_window, prompt_tokens_per_sec, completion_tokens_per_sec
                 FROM response_inference_metrics WHERE response_id = ?1",
                params![response.id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .unwrap();

        assert_eq!(row.0, "qwen3.5:35b");
        assert_eq!(row.1, "llama_cpp");
        assert_eq!(row.2, 8192);
        assert_eq!(row.3, 100.0);
        assert_eq!(row.4, 25.0);
    }

    #[test]
    fn test_log_response_updates_model_context_metric() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
        let request_id = logger.log_request(&request).unwrap();

        for completion_tokens in [50, 100] {
            let mut response = Response::new(request_id.clone(), 200);
            response.runner_id = Some("runner-1".to_string());
            response.inference_metrics = Some(
                InferenceMetrics {
                    resolved_model: Some("qwen3.5:35b".to_string()),
                    context_window: Some(8192),
                    prompt_tokens: Some(100),
                    completion_tokens: Some(completion_tokens),
                    prompt_eval_ms: Some(1_000),
                    completion_eval_ms: Some(2_000),
                    ..Default::default()
                }
                .with_computed_rates(),
            );
            logger.log_response(&response).unwrap();
        }

        let conn = logger.conn.lock().unwrap();
        let row: (i64, i64, i64, f64, f64) = conn
            .query_row(
                "SELECT sample_count, prompt_tokens_total, completion_tokens_total, prompt_tps_min, completion_tps_max
                 FROM model_context_metrics
                 WHERE resolved_model = 'qwen3.5:35b' AND runner_id = 'runner-1' AND context_window = 8192",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .unwrap();

        assert_eq!(row.0, 2);
        assert_eq!(row.1, 200);
        assert_eq!(row.2, 150);
        assert_eq!(row.3, 100.0);
        assert_eq!(row.4, 50.0);
    }

    #[test]
    fn test_get_model_context_metrics() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
        let request_id = logger.log_request(&request).unwrap();

        let mut response = Response::new(request_id, 200);
        response.runner_id = Some("runner-1".to_string());
        response.inference_metrics = Some(
            InferenceMetrics {
                resolved_model: Some("qwen3.5:35b".to_string()),
                context_window: Some(8192),
                prompt_tokens: Some(100),
                completion_tokens: Some(50),
                prompt_eval_ms: Some(1_000),
                completion_eval_ms: Some(2_000),
                ..Default::default()
            }
            .with_computed_rates(),
        );

        logger.log_response(&response).unwrap();

        let metrics = logger.get_model_context_metrics(10).unwrap();
        assert_eq!(metrics.len(), 1);
        let row = &metrics[0];
        assert_eq!(row.resolved_model, "qwen3.5:35b");
        assert_eq!(row.runner_id, "runner-1");
        assert_eq!(row.context_window, 8192);
        assert_eq!(row.sample_count, 1);
        assert_eq!(row.avg_prompt_tokens_per_sec, Some(100.0));
        assert_eq!(row.avg_completion_tokens_per_sec, Some(25.0));
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
    fn request_attribution_round_trips_without_credentials() {
        let logger = AuditLogger::new(":memory:").unwrap();
        logger.find_or_create_user("user123", None).unwrap();
        let mut request = Request::new("user123".into(), "/v1/chat/completions".into());
        request.auth_method = Some("api_key".into());
        request.api_key_id = Some("key-record-id".into());
        request.api_key_name = Some("qwen".into());
        request.user_agent = Some("QwenCode/test".into());
        request.peer_ip = Some("10.77.0.1".into());
        request.proxy_request_id = Some("edge-request-uuid".into());
        request.client_ip = Some("192.0.2.1".into());
        logger.log_request(&request).unwrap();
        let (requests, _) = logger.get_requests_paginated(None, None, 1, 10).unwrap();
        let json = serde_json::to_value(&requests[0]).unwrap();
        for (field, expected) in [
            ("auth_method", "api_key"),
            ("api_key_id", "key-record-id"),
            ("api_key_name", "qwen"),
            ("user_agent", "QwenCode/test"),
            ("peer_ip", "10.77.0.1"),
            ("proxy_request_id", "edge-request-uuid"),
            ("client_ip", "192.0.2.1"),
        ] {
            assert_eq!(json[field], expected);
        }
        assert!(json.get("key_hash").is_none());
        assert!(json.get("plaintext_key").is_none());
    }

    #[test]
    fn legacy_request_attribution_remains_null_after_repeated_migration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit-attribution.sqlite");
        let conn = Connection::open(&path).unwrap();
        crate::audit::legacy_bootstrap_tests::bootstrap(&conn).unwrap();
        conn.execute(
            "INSERT INTO users(id,created_at,last_seen_at) VALUES('u','now','now')",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO requests(id,timestamp,user_id,request_path,request_body) VALUES('old','now','u','/v1/models','{}')", []).unwrap();
        drop(conn);
        for _ in 0..2 {
            let logger = AuditLogger::new(path.to_str().unwrap()).unwrap();
            let (requests, _) = logger.get_requests_paginated(None, None, 1, 10).unwrap();
            assert_eq!(requests[0].id, "old");
            let json = serde_json::to_value(&requests[0]).unwrap();
            for field in [
                "auth_method",
                "api_key_id",
                "api_key_name",
                "user_agent",
                "peer_ip",
                "proxy_request_id",
            ] {
                assert!(json[field].is_null(), "{field}");
            }
        }
    }

    #[test]
    fn source_app_is_persisted_as_attribution() {
        let logger = AuditLogger::new(":memory:").unwrap();
        logger.find_or_create_user("user123", None).unwrap();
        let mut request = Request::new("user123".into(), "/v1/chat/completions".into());
        request.source_app = Some("org.example.personal".into());
        logger.log_request(&request).unwrap();
        let (requests, _) = logger.get_requests_paginated(None, None, 1, 10).unwrap();
        assert_eq!(
            requests[0].source_app.as_deref(),
            Some("org.example.personal")
        );
    }

    #[test]
    fn test_request_history_filters_snapshot_and_bodies() {
        let logger = AuditLogger::new(":memory:").unwrap();
        let user = logger.find_or_create_user("history-user", None).unwrap();
        let mut request = Request::new(user.id, "/v1/chat/completions".into());
        request.model = Some("history-model".into());
        request.source_app = Some("Example-App".into());
        request.request_body = r#"{"messages":[{"role":"user","content":"hello"}]}"#.into();
        logger.log_request(&request).unwrap();
        let filters = RequestFilters {
            model: Some("history-model"),
            origin: Some("example-app"),
            since: Some("2000-01-01T00:00:00Z"),
            until: Some("2100-01-01T00:00:00Z"),
            ..Default::default()
        };
        let (rows, pages, snapshot) = logger.get_request_history(&filters, 1, 1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(pages, 1);
        let mut newer = request.clone();
        newer.id = "newer".into();
        logger.log_request(&newer).unwrap();
        let (rows, pages, _) = logger
            .get_request_history(
                &RequestFilters {
                    snapshot: Some(snapshot),
                    ..filters
                },
                1,
                1,
            )
            .unwrap();
        assert_eq!(rows[0].id, request.id);
        assert_eq!(pages, 1);
        for filters in [
            RequestFilters {
                origin: Some("missing"),
                ..Default::default()
            },
            RequestFilters {
                until: Some("2000-01-01T00:00:00Z"),
                ..Default::default()
            },
        ] {
            assert!(logger
                .get_request_history(&filters, 1, 10)
                .unwrap()
                .0
                .is_empty());
        }
        let mut response = Response::new(request.id.clone(), 200);
        response.response_body = "answer".into();
        logger.log_response(&response).unwrap();
        let bodies = logger.get_request_bodies(&request.id).unwrap().unwrap();
        assert_eq!(
            bodies.request_body.as_deref(),
            Some(request.request_body.as_str())
        );
        assert_eq!(bodies.response_body.as_deref(), Some("answer"));
        logger
            .save_stream_body(&request.id, "data: streamed answer")
            .unwrap();
        assert_eq!(
            logger
                .get_request_bodies(&request.id)
                .unwrap()
                .unwrap()
                .response_body
                .as_deref(),
            Some("data: streamed answer")
        );
        assert!(logger.get_request_bodies("missing").unwrap().is_none());
        assert!(logger
            .get_request_history(&RequestFilters::default(), 1, 0)
            .is_ok());
    }

    #[test]
    fn activity_summary_counts_window_failures_latency_and_top_usage() {
        let logger = AuditLogger::new(":memory:").unwrap();
        let user = logger
            .find_or_create_user("summary-user", Some("summary@example.com"))
            .unwrap();
        let log = |model: &str, app: Option<&str>, status: u16, latency_ms: u64, age_hours: i64| {
            let mut request = Request::new(user.id.clone(), "/v1/chat/completions".into());
            request.model = Some(model.into());
            request.source_app = app.map(Into::into);
            request.timestamp = Utc::now() - chrono::Duration::hours(age_hours);
            logger.log_request(&request).unwrap();
            let mut response = Response::new(request.id.clone(), status);
            response.latency_ms = latency_ms;
            response.tokens_prompt = Some(10);
            response.tokens_completion = Some(5);
            logger.log_response(&response).unwrap();
            request.id
        };
        for latency in [100, 200, 300, 400] {
            log("busy-model", Some("app-one"), 200, latency, 1);
        }
        let failed = log("quiet-model", None, 503, 9000, 2);
        log("old-model", Some("app-one"), 500, 50, 48);
        let mut pending = Request::new(user.id.clone(), "/v1/embeddings".into());
        pending.api_key_name = Some("ci key".into());
        logger.log_request(&pending).unwrap();

        let since = (Utc::now() - chrono::Duration::hours(24)).to_rfc3339();
        let summary = logger.get_activity_summary(&since, 5).unwrap();
        assert_eq!(
            (summary.requests, summary.completed, summary.failed),
            (6, 5, 1)
        );
        assert_eq!((summary.tokens_prompt, summary.tokens_completion), (50, 25));
        // Failures are excluded from latency percentiles.
        assert_eq!(
            (summary.latency_p50_ms, summary.latency_p95_ms),
            (Some(200), Some(400))
        );
        let models: Vec<_> = summary
            .top_models
            .iter()
            .map(|row| (row.label.as_str(), row.requests, row.failed))
            .collect();
        assert_eq!(
            models,
            [
                ("busy-model", 4, 0),
                ("/v1/embeddings", 1, 0),
                ("quiet-model", 1, 1)
            ]
        );
        let origins: Vec<_> = summary
            .top_origins
            .iter()
            .map(|row| (row.kind.as_str(), row.label.as_str(), row.requests))
            .collect();
        assert_eq!(
            origins,
            [
                ("app", "app-one", 4),
                ("key", "ci key", 1),
                ("user", "summary@example.com", 1)
            ]
        );
        assert_eq!(
            logger
                .get_activity_summary(&since, 1)
                .unwrap()
                .top_models
                .len(),
            1
        );

        let (rows, pages, _) = logger
            .get_request_history(
                &RequestFilters {
                    since: Some(&since),
                    failed_only: true,
                    ..Default::default()
                },
                1,
                10,
            )
            .unwrap();
        assert_eq!(
            rows.iter().map(|row| row.id.as_str()).collect::<Vec<_>>(),
            [failed.as_str()]
        );
        assert_eq!(pages, 1);

        let empty = logger
            .get_activity_summary("2100-01-01T00:00:00Z", 5)
            .unwrap();
        assert_eq!((empty.requests, empty.latency_p50_ms), (0, None));
        assert!(empty.top_models.is_empty());
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
        let (requests, _) = logger
            .get_requests_paginated(Some("user1"), None, 1, 10)
            .unwrap();
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
        let (requests, _) = logger
            .get_requests_paginated(None, Some("model-2"), 1, 10)
            .unwrap();
        assert!(requests
            .iter()
            .all(|r| r.model.as_ref().unwrap().contains("model-2")));
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
            tokens_prompt: 2000,
            tokens_completion: 3000,
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
            user_email: Some("user@example.com".to_string()),
            request_path: "/v1/chat/completions".to_string(),
            model: Some("llama2".to_string()),
            client_ip: Some("192.168.1.100".to_string()),
            source_app: Some("org.example.personal".to_string()),
            auth_method: None,
            api_key_id: None,
            api_key_name: None,
            user_agent: None,
            peer_ip: None,
            proxy_request_id: None,

            status: Some(200),
            latency_ms: Some(150),
            tokens_prompt: Some(10),
            tokens_completion: Some(5),
            runner_id: Some("gpu-server".to_string()),
            wol_sent: true,
        };
        assert_eq!(req_with_resp.status, Some(200));
        assert_eq!(req_with_resp.latency_ms, Some(150));
        assert_eq!(req_with_resp.client_ip, Some("192.168.1.100".to_string()));
        assert_eq!(
            req_with_resp.user_email,
            Some("user@example.com".to_string())
        );
        assert_eq!(req_with_resp.runner_id, Some("gpu-server".to_string()));
        assert!(req_with_resp.wol_sent);
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
        logger
            .upsert_runner(
                "runner-1",
                "Test Runner",
                Some("AA:BB:CC:DD:EE:FF"),
                Some("gpu-server"),
                Some(&models),
            )
            .unwrap();

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
        logger
            .upsert_runner(
                "runner-1",
                "Old Name",
                Some("AA:BB:CC:DD:EE:FF"),
                None,
                Some(&models),
            )
            .unwrap();

        // Update with new name, keeps MAC address and models
        logger
            .upsert_runner("runner-1", "New Name", None, Some("cpu-server"), None)
            .unwrap();

        let runner = logger.get_runner("runner-1").unwrap().unwrap();
        assert_eq!(runner.name, "New Name");
        assert_eq!(runner.mac_address, Some("AA:BB:CC:DD:EE:FF".to_string())); // Preserved
        assert_eq!(runner.machine_type, Some("cpu-server".to_string()));
        assert_eq!(runner.available_models, models); // Preserved
    }

    #[test]
    fn test_upsert_runner_without_mac() {
        let logger = create_test_logger();
        logger
            .upsert_runner("runner-1", "Test Runner", None, None, None)
            .unwrap();

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
        logger
            .upsert_runner(
                "runner-1",
                "Runner 1",
                Some("AA:BB:CC:DD:EE:01"),
                None,
                None,
            )
            .unwrap();
        logger
            .upsert_runner(
                "runner-2",
                "Runner 2",
                Some("AA:BB:CC:DD:EE:02"),
                None,
                None,
            )
            .unwrap();
        logger
            .upsert_runner("runner-3", "Runner 3", None, None, None)
            .unwrap();

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
        logger
            .upsert_runner(
                "runner-big",
                "Big Runner",
                Some("AA:BB:CC:DD:EE:01"),
                None,
                Some(&big_models),
            )
            .unwrap();

        // Runner with fast model
        let fast_models = vec!["llama3:8b".to_string(), "mistral:7b".to_string()];
        logger
            .upsert_runner(
                "runner-fast",
                "Fast Runner",
                Some("AA:BB:CC:DD:EE:02"),
                None,
                Some(&fast_models),
            )
            .unwrap();

        // Query by class with config
        use crate::config::ModelsConfig;
        use crate::gateway::ModelClass;

        let models_config = ModelsConfig {
            big: vec!["llama3:70b".to_string()],
            fast: vec!["llama3:8b".to_string(), "mistral:7b".to_string()],
            ..Default::default()
        };

        let big_runners = logger
            .get_runners_by_model_class(ModelClass::Big, &models_config)
            .unwrap();
        assert_eq!(big_runners.len(), 1);
        assert_eq!(big_runners[0].id, "runner-big");

        let fast_runners = logger
            .get_runners_by_model_class(ModelClass::Fast, &models_config)
            .unwrap();
        assert_eq!(fast_runners.len(), 1);
        assert_eq!(fast_runners[0].id, "runner-fast");
    }

    #[test]
    fn test_get_runners_by_model() {
        let logger = create_test_logger();

        let models1 = vec!["llama3:8b".to_string(), "mistral:7b".to_string()];
        logger
            .upsert_runner("runner-1", "Runner 1", None, None, Some(&models1))
            .unwrap();

        let models2 = vec!["llama3:8b".to_string()];
        logger
            .upsert_runner("runner-2", "Runner 2", None, None, Some(&models2))
            .unwrap();

        // Both have llama3:8b
        let llama_runners = logger.get_runners_by_model("llama3:8b").unwrap();
        assert_eq!(llama_runners.len(), 2);

        // Only runner-1 has mistral
        let mistral_runners = logger.get_runners_by_model("mistral:7b").unwrap();
        assert_eq!(mistral_runners.len(), 1);
        assert_eq!(mistral_runners[0].id, "runner-1");
    }

    // ==================== API Key Tests ====================

    #[test]
    fn test_create_api_key() {
        let logger = create_test_logger();
        let user = logger
            .find_or_create_user("user123", Some("user@example.com"))
            .unwrap();

        let roles = vec!["model:specific".to_string()];
        let (key, secret) = logger.create_api_key(&user.id, "Test Key", &roles).unwrap();

        assert!(!key.id.is_empty());
        assert_eq!(key.user_id, "user123");
        assert_eq!(key.name, "Test Key");
        assert_eq!(key.roles, roles);
        assert_eq!(key.user_email, Some("user@example.com".to_string()));
        assert!(!key.revoked);
        assert!(key.secret_available);
        assert!(key.last_used_at.is_none());

        // Secret should be in sk-<hex> format
        assert!(secret.starts_with("sk-"));
        assert_eq!(secret.len(), 35); // "sk-" + 32 hex chars
        assert_eq!(logger.get_api_key_secret(&key.id).unwrap(), Some(secret));
    }

    #[test]
    fn test_get_api_key_secret_revoked() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();

        let (key, _) = logger.create_api_key(&user.id, "Test Key", &[]).unwrap();
        assert!(logger.get_api_key_secret(&key.id).unwrap().is_some());

        let revoked = logger.revoke_api_key(&key.id).unwrap();
        assert!(revoked);
        assert!(logger.get_api_key_secret(&key.id).unwrap().is_none());
    }

    #[test]
    fn test_validate_api_key_valid() {
        let logger = create_test_logger();
        let user = logger
            .find_or_create_user("user123", Some("user@example.com"))
            .unwrap();

        let roles = vec!["model:specific".to_string()];
        let (_, secret) = logger.create_api_key(&user.id, "Test Key", &roles).unwrap();

        // Validate the key
        let result = logger.validate_api_key(&secret).unwrap();
        assert!(result.is_some());

        let (user_id, email, validated_roles) = result.unwrap();
        assert_eq!(user_id, "user123");
        assert_eq!(email, Some("user@example.com".to_string()));
        assert_eq!(validated_roles, roles);
    }

    #[test]
    fn test_validate_api_key_invalid() {
        let logger = create_test_logger();

        // Try to validate a key that doesn't exist
        let result = logger
            .validate_api_key("sk-invalidkey12345678901234567890")
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_api_key_wrong_prefix() {
        let logger = create_test_logger();

        // Keys not starting with sk- should return None immediately
        let result = logger.validate_api_key("invalid-key").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_api_key_revoked() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();

        let (key, secret) = logger.create_api_key(&user.id, "Test Key", &[]).unwrap();

        // Validate before revocation - should work
        let result = logger.validate_api_key(&secret).unwrap();
        assert!(result.is_some());

        // Revoke the key
        let revoked = logger.revoke_api_key(&key.id).unwrap();
        assert!(revoked);

        // Validate after revocation - should fail
        let result = logger.validate_api_key(&secret).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn validated_api_key_identity_matches_authenticated_record_without_secret() {
        let logger = AuditLogger::new(":memory:").unwrap();
        logger
            .find_or_create_user("u", Some("u@example.com"))
            .unwrap();
        let roles = vec!["model:specific".to_string()];
        let (key, secret) = logger.create_api_key("u", "Qwen Code", &roles).unwrap();
        let identity = logger
            .validate_api_key_with_identity(&secret)
            .unwrap()
            .unwrap();
        assert_eq!(identity.key_id, key.id);
        assert_eq!(identity.key_name, "Qwen Code");
        assert_eq!(identity.user_id, "u");
        assert_eq!(identity.email.as_deref(), Some("u@example.com"));
        assert_eq!(identity.roles, roles);
        let serialized = serde_json::to_string(&identity).unwrap();
        assert!(!serialized.contains(&secret));
        assert!(!serialized.contains(&key.key_hash));
        logger.revoke_api_key(&key.id).unwrap();
        assert!(logger
            .validate_api_key_with_identity(&secret)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_validate_api_key_updates_last_used() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();

        let (_, secret) = logger.create_api_key(&user.id, "Test Key", &[]).unwrap();

        // Validate the key
        logger.validate_api_key(&secret).unwrap();

        // Check that last_used_at was updated
        let keys = logger.list_api_keys().unwrap();
        assert_eq!(keys.len(), 1);
        assert!(keys[0].last_used_at.is_some());
    }

    #[test]
    fn test_update_api_key_roles_changes_authenticated_roles() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let (key, secret) = logger.create_api_key(&user.id, "Test Key", &[]).unwrap();
        let roles = vec!["model:specific".to_string()];

        assert!(logger.update_api_key_roles(&key.id, &roles).unwrap());

        let (_, _, validated_roles) = logger.validate_api_key(&secret).unwrap().unwrap();
        assert_eq!(validated_roles, roles);
        assert_eq!(logger.list_api_keys().unwrap()[0].roles, roles);
    }

    #[test]
    fn test_list_api_keys_empty() {
        let logger = create_test_logger();
        let keys = logger.list_api_keys().unwrap();
        assert!(keys.is_empty());
    }

    #[test]
    fn test_list_api_keys() {
        let logger = create_test_logger();
        let user1 = logger
            .find_or_create_user("user1", Some("user1@example.com"))
            .unwrap();
        let user2 = logger
            .find_or_create_user("user2", Some("user2@example.com"))
            .unwrap();

        logger.create_api_key(&user1.id, "Key 1", &[]).unwrap();
        logger
            .create_api_key(&user1.id, "Key 2", &["model:specific".to_string()])
            .unwrap();
        logger.create_api_key(&user2.id, "Key 3", &[]).unwrap();

        let keys = logger.list_api_keys().unwrap();
        assert_eq!(keys.len(), 3);

        // Should include user emails
        assert!(keys
            .iter()
            .any(|k| k.user_email == Some("user1@example.com".to_string())));
        assert!(keys
            .iter()
            .any(|k| k.user_email == Some("user2@example.com".to_string())));
        assert!(keys
            .iter()
            .any(|k| k.roles == vec!["model:specific".to_string()]));
    }

    #[test]
    fn test_revoke_api_key() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();

        let (key, _) = logger.create_api_key(&user.id, "Test Key", &[]).unwrap();

        // Revoke the key
        let revoked = logger.revoke_api_key(&key.id).unwrap();
        assert!(revoked);

        // Check it's marked as revoked
        let keys = logger.list_api_keys().unwrap();
        assert_eq!(keys.len(), 1);
        assert!(keys[0].revoked);
    }

    #[test]
    fn test_revoke_api_key_not_found() {
        let logger = create_test_logger();

        // Try to revoke a key that doesn't exist
        let revoked = logger.revoke_api_key("nonexistent-id").unwrap();
        assert!(!revoked);
    }

    #[test]
    fn test_api_key_struct() {
        let key = ApiKey {
            id: "key-123".to_string(),
            key_hash: "hash123".to_string(),
            user_id: "user-456".to_string(),
            user_email: Some("user@example.com".to_string()),
            name: "Test Key".to_string(),
            roles: vec!["model:specific".to_string()],
            created_at: "2024-01-01T00:00:00Z".to_string(),
            last_used_at: Some("2024-01-02T00:00:00Z".to_string()),
            revoked: false,
            secret_available: true,
        };
        assert_eq!(key.id, "key-123");
        assert_eq!(key.name, "Test Key");
        assert!(!key.revoked);
        assert!(key.secret_available);
        assert_eq!(key.roles, vec!["model:specific".to_string()]);

        // Verify key_hash is not serialized
        let json = serde_json::to_string(&key).unwrap();
        assert!(!json.contains("hash123"));
        assert!(!json.contains("key_hash"));
    }

    // ==================== Runner Metrics Tests ====================

    #[test]
    fn test_record_metric_new() {
        let logger = create_test_logger();
        logger.record_metric("runner-1", "fast", 100).unwrap();

        let metrics = logger.get_runner_metrics("runner-1").unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].runner_id, "runner-1");
        assert_eq!(metrics[0].model_class, "fast");
        assert_eq!(metrics[0].sample_count, 1);
        assert_eq!(metrics[0].total_ms, 100);
        assert_eq!(metrics[0].min_ms, Some(100));
        assert_eq!(metrics[0].max_ms, Some(100));
    }

    #[test]
    fn test_record_metric_accumulates() {
        let logger = create_test_logger();
        logger.record_metric("runner-1", "fast", 100).unwrap();
        logger.record_metric("runner-1", "fast", 200).unwrap();
        logger.record_metric("runner-1", "fast", 150).unwrap();

        let metrics = logger.get_runner_metrics("runner-1").unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].sample_count, 3);
        assert_eq!(metrics[0].total_ms, 450); // 100 + 200 + 150
        assert_eq!(metrics[0].min_ms, Some(100));
        assert_eq!(metrics[0].max_ms, Some(200));
    }

    #[test]
    fn test_record_metric_multiple_classes() {
        let logger = create_test_logger();
        logger.record_metric("runner-1", "fast", 100).unwrap();
        logger.record_metric("runner-1", "big", 500).unwrap();
        logger.record_metric("runner-1", "boot", 5000).unwrap();

        let metrics = logger.get_runner_metrics("runner-1").unwrap();
        assert_eq!(metrics.len(), 3);

        let fast = metrics.iter().find(|m| m.model_class == "fast").unwrap();
        assert_eq!(fast.total_ms, 100);

        let big = metrics.iter().find(|m| m.model_class == "big").unwrap();
        assert_eq!(big.total_ms, 500);

        let boot = metrics.iter().find(|m| m.model_class == "boot").unwrap();
        assert_eq!(boot.total_ms, 5000);
    }

    #[test]
    fn test_get_all_metrics() {
        let logger = create_test_logger();
        logger.record_metric("runner-1", "fast", 100).unwrap();
        logger.record_metric("runner-2", "fast", 200).unwrap();
        logger.record_metric("runner-1", "big", 500).unwrap();

        let all_metrics = logger.get_all_metrics().unwrap();
        assert_eq!(all_metrics.len(), 3);
    }

    #[test]
    fn test_get_boot_metrics() {
        let logger = create_test_logger();
        logger.record_metric("runner-1", "boot", 5000).unwrap();
        logger.record_metric("runner-1", "fast", 100).unwrap();

        let boot = logger.get_boot_metrics("runner-1").unwrap();
        assert!(boot.is_some());
        assert_eq!(boot.unwrap().total_ms, 5000);
    }

    #[test]
    fn test_get_boot_metrics_not_found() {
        let logger = create_test_logger();
        logger.record_metric("runner-1", "fast", 100).unwrap();

        let boot = logger.get_boot_metrics("runner-1").unwrap();
        assert!(boot.is_none());
    }

    #[test]
    fn test_runner_metric_row_avg_ms() {
        let metric = RunnerMetricRow {
            runner_id: "runner-1".to_string(),
            model_class: "fast".to_string(),
            sample_count: 4,
            total_ms: 400,
            min_ms: Some(50),
            max_ms: Some(150),
            last_updated_at: "2024-01-01T00:00:00Z".to_string(),
        };
        assert_eq!(metric.avg_ms(), Some(100.0));
    }

    #[test]
    fn test_runner_metric_row_avg_ms_zero_samples() {
        let metric = RunnerMetricRow {
            runner_id: "runner-1".to_string(),
            model_class: "fast".to_string(),
            sample_count: 0,
            total_ms: 0,
            min_ms: None,
            max_ms: None,
            last_updated_at: "2024-01-01T00:00:00Z".to_string(),
        };
        assert_eq!(metric.avg_ms(), None);
    }

    #[test]
    fn test_log_response_with_model_class() {
        let logger = create_test_logger();
        let user = logger.find_or_create_user("user123", None).unwrap();
        let request = Request::new(user.id.clone(), "/v1/chat/completions".to_string());
        let request_id = logger.log_request(&request).unwrap();

        let mut response = Response::new(request_id, 200);
        response.latency_ms = 150;
        response.runner_id = Some("runner-1".to_string());
        response.model_class = Some("fast".to_string());
        logger.log_response(&response).unwrap();

        // Verify metrics were recorded
        let metrics = logger.get_runner_metrics("runner-1").unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].model_class, "fast");
        assert_eq!(metrics[0].total_ms, 150);
    }
}
