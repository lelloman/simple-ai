//! Audit retention: clear request/response bodies after `BODY_RETENTION_DAYS`
//! and delete audit rows after `METADATA_RETENTION_DAYS`. See
//! `docs/audit-durability.md`.
//!
//! Every batch is one short `IMMEDIATE` transaction on the general connection,
//! so the audit writer and admin queries wait at most one batch, never a run.
use chrono::{DateTime, Duration as Days, Utc};
use rusqlite::{params, TransactionBehavior};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::sqlite::{AuditError, AuditLogger};

pub const BODY_RETENTION_DAYS: i64 = 2;
pub const METADATA_RETENTION_DAYS: i64 = 365;
const BATCH_ROWS: usize = 500;
const BATCH_PAUSE: Duration = Duration::from_millis(20);
const PRUNE_INTERVAL: Duration = Duration::from_secs(3600);

const SELECT_EXPIRED_SQL: &str =
    "SELECT id FROM requests WHERE timestamp < ?1 ORDER BY timestamp LIMIT ?2";
/// Deletes for one batch of expired request ids, children first. `{ids}` is
/// replaced with the id placeholders. Every statement looks rows up by an
/// index: `response_inference_metrics` by its `response_id` key, since its
/// `request_id` has no index.
const DELETE_ROWS_SQL: [&str; 3] = [
    "DELETE FROM response_inference_metrics
     WHERE response_id IN (SELECT id FROM responses WHERE request_id IN ({ids}))",
    "DELETE FROM responses WHERE request_id IN ({ids})",
    "DELETE FROM requests WHERE id IN ({ids})",
];
const SELECT_BODY_ROWS_SQL: &str =
    "SELECT rowid, timestamp FROM requests WHERE rowid > ?1 ORDER BY rowid LIMIT ?2";
const CLEAR_RESPONSE_BODIES_SQL: &str = "UPDATE responses SET response_body = NULL
     WHERE response_body IS NOT NULL
       AND request_id IN (SELECT id FROM requests WHERE rowid > ?1 AND rowid <= ?2)";
const CLEAR_REQUEST_BODIES_SQL: &str = "UPDATE requests SET request_body = NULL, stream_body = NULL
     WHERE rowid > ?1 AND rowid <= ?2
       AND (request_body IS NOT NULL OR stream_body IS NOT NULL)";

#[derive(Debug, Default)]
pub struct PruneReport {
    pub bodies_cleared: usize,
    pub rows_deleted: usize,
    pub batches: usize,
    /// Longest time one batch held the audit connection lock.
    pub longest_batch: Duration,
}

/// Remembers the last request rowid whose bodies were cleared, so each run
/// only walks rows that aged past the body cutoff since the previous run.
#[derive(Debug, Default)]
pub struct PruneCursor {
    body_rowid: i64,
}

impl PruneReport {
    /// Run one batch, record how long it held the lock, then pause so waiting
    /// writers get the lock before the next batch.
    fn time<T>(&mut self, pause: Duration, batch: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let result = batch();
        self.batches += 1;
        self.longest_batch = self.longest_batch.max(started.elapsed());
        if !pause.is_zero() {
            std::thread::sleep(pause);
        }
        result
    }
}

fn db_err(e: rusqlite::Error) -> AuditError {
    AuditError::DatabaseError(e.to_string())
}

impl AuditLogger {
    /// Prune bodies and rows older than the retention ages relative to `now`.
    pub fn prune(
        &self,
        now: DateTime<Utc>,
        cursor: &mut PruneCursor,
        batch_rows: usize,
        pause: Duration,
    ) -> Result<PruneReport, AuditError> {
        let body_cutoff = (now - Days::days(BODY_RETENTION_DAYS)).to_rfc3339();
        let row_cutoff = (now - Days::days(METADATA_RETENTION_DAYS)).to_rfc3339();
        let mut report = PruneReport::default();
        loop {
            let deleted = report.time(pause, || self.delete_rows_batch(&row_cutoff, batch_rows))?;
            report.rows_deleted += deleted;
            if deleted < batch_rows {
                break;
            }
        }
        loop {
            let (cleared, full) = report.time(pause, || {
                self.clear_bodies_batch(cursor, &body_cutoff, batch_rows)
            })?;
            report.bodies_cleared += cleared;
            if !full {
                break;
            }
        }
        Ok(report)
    }

    /// Delete up to `limit` of the oldest requests before `cutoff`, with their
    /// responses and inference metrics. Returns the number of requests deleted.
    fn delete_rows_batch(&self, cutoff: &str, limit: usize) -> Result<usize, AuditError> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_err)?;
        let ids = tx
            .prepare(SELECT_EXPIRED_SQL)
            .map_err(db_err)?
            .query_map(params![cutoff, limit as i64], |row| row.get::<_, String>(0))
            .map_err(db_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(db_err)?;
        if ids.is_empty() {
            return Ok(0);
        }
        let placeholders = vec!["?"; ids.len()].join(",");
        for sql in DELETE_ROWS_SQL {
            tx.execute(
                &sql.replace("{ids}", &placeholders),
                rusqlite::params_from_iter(&ids),
            )
            .map_err(db_err)?;
        }
        tx.commit().map_err(db_err)?;
        Ok(ids.len())
    }

    /// Clear the bodies of up to `limit` requests after the cursor whose
    /// timestamp is before `cutoff`. Walks by rowid (insertion order), stopping
    /// at the first request that is still inside the body retention window.
    /// Returns the number of requests examined and whether the batch was full.
    fn clear_bodies_batch(
        &self,
        cursor: &mut PruneCursor,
        cutoff: &str,
        limit: usize,
    ) -> Result<(usize, bool), AuditError> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|e| AuditError::DatabaseError(e.to_string()))?;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_err)?;
        let rows = tx
            .prepare(SELECT_BODY_ROWS_SQL)
            .map_err(db_err)?
            .query_map(params![cursor.body_rowid, limit as i64], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(db_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(db_err)?;
        let aged = rows
            .iter()
            .take_while(|(_, ts)| ts.as_str() < cutoff)
            .count();
        let Some(&(last, _)) = rows[..aged].last() else {
            return Ok((0, false));
        };
        for sql in [CLEAR_RESPONSE_BODIES_SQL, CLEAR_REQUEST_BODIES_SQL] {
            tx.execute(sql, params![cursor.body_rowid, last])
                .map_err(db_err)?;
        }
        tx.commit().map_err(db_err)?;
        cursor.body_rowid = last;
        Ok((aged, aged == limit))
    }
}

/// Run `prune` once at startup and then every `PRUNE_INTERVAL`.
pub fn spawn_pruner(audit: Arc<AuditLogger>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let cursor = Arc::new(std::sync::Mutex::new(PruneCursor::default()));
        let mut interval = tokio::time::interval(PRUNE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let audit = audit.clone();
            let cursor = cursor.clone();
            let result = tokio::task::spawn_blocking(move || {
                let mut cursor = cursor.lock().unwrap_or_else(|e| e.into_inner());
                audit.prune(Utc::now(), &mut cursor, BATCH_ROWS, BATCH_PAUSE)
            })
            .await;
            match result {
                Ok(Ok(report)) => tracing::info!(
                    bodies_cleared = report.bodies_cleared,
                    rows_deleted = report.rows_deleted,
                    batches = report.batches,
                    longest_batch_ms = report.longest_batch.as_millis() as u64,
                    "Audit retention prune finished"
                ),
                Ok(Err(e)) => tracing::error!("Audit retention prune failed: {}", e),
                Err(e) => tracing::error!("Audit retention prune task panicked: {}", e),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::RequestFilters;
    use crate::models::request::{Request, Response};
    use simple_ai_common::InferenceMetrics;

    /// Log a request and its response `age_days` ago, with bodies and metrics.
    fn log_aged(logger: &AuditLogger, now: DateTime<Utc>, age_days: i64) -> String {
        let mut request = Request::new("user".into(), "/v1/chat/completions".into());
        request.timestamp = now - Days::days(age_days);
        request.request_body = "x".repeat(4096);
        let id = logger.log_request(&request).unwrap();
        logger.save_stream_body(&id, "data: streamed");
        let mut response = Response::new(id.clone(), 200);
        response.response_body = "y".repeat(4096);
        response.inference_metrics = Some(InferenceMetrics {
            prompt_tokens: Some(1),
            ..Default::default()
        });
        logger.log_response(&response).unwrap();
        logger.flush();
        id
    }

    fn count(logger: &AuditLogger, sql: &str) -> i64 {
        let conn = logger.conn.lock().unwrap();
        conn.query_row(sql, [], |row| row.get(0)).unwrap()
    }

    fn logger() -> AuditLogger {
        let logger = AuditLogger::new(":memory:").unwrap();
        logger.find_or_create_user("user", None).unwrap();
        logger
    }

    #[test]
    fn prune_clears_aged_bodies_and_deletes_expired_rows() {
        let logger = logger();
        let now = Utc::now();
        let expired = log_aged(&logger, now, METADATA_RETENTION_DAYS + 1);
        let aged = log_aged(&logger, now, BODY_RETENTION_DAYS + 1);
        let fresh = log_aged(&logger, now, BODY_RETENTION_DAYS - 1);

        let mut cursor = PruneCursor::default();
        let report = logger.prune(now, &mut cursor, 10, Duration::ZERO).unwrap();
        assert_eq!(report.rows_deleted, 1);
        assert_eq!(report.bodies_cleared, 1);

        assert_eq!(count(&logger, "SELECT COUNT(*) FROM requests"), 2);
        assert_eq!(count(&logger, "SELECT COUNT(*) FROM responses"), 2);
        assert_eq!(
            count(&logger, "SELECT COUNT(*) FROM response_inference_metrics"),
            2
        );
        assert!(logger.get_request_bodies(&expired).unwrap().is_none());

        let aged_bodies = logger.get_request_bodies(&aged).unwrap().unwrap();
        assert!(aged_bodies.request_body.is_none());
        assert!(aged_bodies.response_body.is_none());
        let fresh_bodies = logger.get_request_bodies(&fresh).unwrap().unwrap();
        assert_eq!(fresh_bodies.request_body.unwrap().len(), 4096);
        assert_eq!(fresh_bodies.response_body.unwrap(), "data: streamed");

        // History still lists pruned rows with their metadata.
        let (rows, _, _) = logger
            .get_request_history(&RequestFilters::default(), 1, 10)
            .unwrap();
        let ids: Vec<_> = rows.iter().map(|row| row.id.as_str()).collect();
        assert_eq!(ids, [fresh.as_str(), aged.as_str()]);
        assert_eq!(rows[1].status, Some(200));
    }

    /// No prune statement may scan a whole table: each batch runs while the
    /// writers wait, so its cost must follow the batch size, not the table.
    #[test]
    fn prune_statements_never_scan_a_table() {
        let logger = logger();
        let conn = logger.conn.lock().unwrap();
        let deletes = DELETE_ROWS_SQL.map(|sql| sql.replace("{ids}", "?1,?2"));
        let statements = [
            SELECT_EXPIRED_SQL,
            SELECT_BODY_ROWS_SQL,
            CLEAR_RESPONSE_BODIES_SQL,
            CLEAR_REQUEST_BODIES_SQL,
        ]
        .into_iter()
        .chain(deletes.iter().map(String::as_str));
        for sql in statements {
            let plan = conn
                .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
                .unwrap()
                .query_map(params!["a", "b"], |row| row.get::<_, String>(3))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            assert!(!plan.is_empty(), "no plan for {sql}");
            assert!(
                plan.iter().all(|step| !step.starts_with("SCAN ")),
                "{sql} scans a table: {plan:?}"
            );
        }
    }

    #[test]
    fn prune_resumes_from_cursor_as_rows_age() {
        let logger = logger();
        let now = Utc::now();
        let ids: Vec<_> = (0..25)
            .map(|i| log_aged(&logger, now, BODY_RETENTION_DAYS + 30 - i))
            .collect();

        let mut cursor = PruneCursor::default();
        let report = logger.prune(now, &mut cursor, 4, Duration::ZERO).unwrap();
        // Rows aged 32..8 days are past the 2-day cutoff; 25 rows in batches of 4.
        assert_eq!(report.bodies_cleared, 25);
        assert!(report.batches >= 7);

        let later = now + Days::days(5);
        let id = log_aged(&logger, later, 0);
        let report = logger.prune(later, &mut cursor, 4, Duration::ZERO).unwrap();
        assert_eq!(report.bodies_cleared, 0);
        assert!(logger
            .get_request_bodies(&id)
            .unwrap()
            .unwrap()
            .request_body
            .is_some());
        assert!(ids.iter().all(|id| logger
            .get_request_bodies(id)
            .unwrap()
            .unwrap()
            .request_body
            .is_none()));
        assert_eq!(
            count(
                &logger,
                "SELECT COUNT(*) FROM responses WHERE response_body IS NOT NULL"
            ),
            1
        );
    }

    /// While a prune run is in progress, a chat request's queued audit write
    /// is committed after at most one batch, and an inline write on the general
    /// connection waits at most one batch, not the whole run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn request_logging_is_not_stalled_by_a_prune_run() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.db");
        let logger = Arc::new(AuditLogger::new(path.to_str().unwrap()).unwrap());
        logger.find_or_create_user("user", None).unwrap();
        let now = Utc::now();
        for _ in 0..2_000 {
            log_aged(&logger, now, BODY_RETENTION_DAYS + 1);
        }

        let pruner = {
            let logger = logger.clone();
            std::thread::spawn(move || {
                let mut cursor = PruneCursor::default();
                logger
                    .prune(now, &mut cursor, 100, Duration::from_millis(20))
                    .unwrap()
            })
        };
        // Let the run start; with 20 batches and 20 ms pauses it takes > 400 ms.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let started = Instant::now();
        let queued = Request::new("user".into(), "/v1/chat/completions".into());
        logger.log_request_async(&queued).await.unwrap();
        logger.flush();
        let chat_wait = started.elapsed();

        let started = Instant::now();
        let inline = Request::new("user".into(), "/v1/embeddings".into());
        logger.log_request(&inline).unwrap();
        let inline_wait = started.elapsed();
        assert!(!pruner.is_finished(), "prune finished before the writes");

        let report = pruner.join().unwrap();
        assert_eq!(report.bodies_cleared, 2_000);
        assert!(report.batches >= 20);
        for (what, waited) in [
            ("longest prune batch", report.longest_batch),
            ("queued chat write", chat_wait),
            ("inline write", inline_wait),
        ] {
            assert!(
                waited < Duration::from_millis(250),
                "{what} took {waited:?}"
            );
        }
        assert!(logger
            .get_request_bodies(&queued.id)
            .unwrap()
            .unwrap()
            .request_body
            .is_some());
    }
}
