//! Frozen pre-migration bootstrap, used only by differential tests.
use super::sqlite::AuditError;
use rusqlite::Connection;
pub(super) fn bootstrap(conn: &Connection) -> Result<(), AuditError> {
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
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    // Create requests table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                request_path TEXT NOT NULL,
                request_body TEXT,
                model TEXT,
                client_ip TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )",
        [],
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    let _ = conn.execute("ALTER TABLE requests ADD COLUMN source_app TEXT", []);

    // Migration: add client_ip column if it doesn't exist (for existing databases)
    let _ = conn.execute("ALTER TABLE requests ADD COLUMN client_ip TEXT", []);

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
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

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
    conn.execute(
        "CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                plaintext_key TEXT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                roles TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                revoked INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )",
        [],
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    // Migration: store retrievable API key secrets for keys created after this migration.
    let _ = conn.execute("ALTER TABLE api_keys ADD COLUMN plaintext_key TEXT", []);
    // Migration: existing keys remain class-only until an administrator
    // explicitly grants additional roles.
    let _ = conn.execute(
        "ALTER TABLE api_keys ADD COLUMN roles TEXT NOT NULL DEFAULT '[]'",
        [],
    );

    // Create indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)",
        [],
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id)",
        [],
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_responses_request_id ON responses(request_id)",
        [],
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

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
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    // Migration: add available_models column if it doesn't exist
    let _ = conn.execute("ALTER TABLE runners ADD COLUMN available_models TEXT", []); // Ignore error if column already exists

    // Migration: add model_class column to responses
    let _ = conn.execute("ALTER TABLE responses ADD COLUMN model_class TEXT", []);

    // Create runner_metrics table for tracking boot time and inference latency
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runner_metrics (
                runner_id TEXT NOT NULL,
                model_class TEXT NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                total_ms INTEGER NOT NULL DEFAULT 0,
                min_ms INTEGER,
                max_ms INTEGER,
                last_updated_at TEXT NOT NULL,
                PRIMARY KEY (runner_id, model_class)
            )",
        [],
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS response_inference_metrics (
                response_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                runner_id TEXT,
                model_class TEXT,
                requested_model TEXT,
                resolved_model TEXT,
                engine_type TEXT,
                context_window INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                prompt_eval_ms INTEGER,
                completion_eval_ms INTEGER,
                total_inference_ms INTEGER,
                prompt_tokens_per_sec REAL,
                completion_tokens_per_sec REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (response_id) REFERENCES responses(id),
                FOREIGN KEY (request_id) REFERENCES requests(id)
            )",
        [],
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS model_context_metrics (
                resolved_model TEXT NOT NULL,
                runner_id TEXT NOT NULL,
                context_window INTEGER NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                prompt_tokens_total INTEGER NOT NULL DEFAULT 0,
                completion_tokens_total INTEGER NOT NULL DEFAULT 0,
                prompt_weighted_tps_total REAL NOT NULL DEFAULT 0,
                completion_weighted_tps_total REAL NOT NULL DEFAULT 0,
                prompt_tps_min REAL,
                prompt_tps_max REAL,
                completion_tps_min REAL,
                completion_tps_max REAL,
                last_updated_at TEXT NOT NULL,
                PRIMARY KEY (resolved_model, runner_id, context_window)
            )",
        [],
    )
    .map_err(|e| AuditError::DatabaseError(e.to_string()))?;

    Ok(())
}
