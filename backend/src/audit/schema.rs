//! Shared descriptions of the existing ordinary bootstrap tables.
//! Driver execution, idempotence and legacy ALTER statements remain local.
use simple_server::database::sqlite::schema::*;
/// Add attribution without changing existing request values or storing credentials.
pub(super) fn migrate_request_attribution(
    conn: &rusqlite::Connection,
) -> Result<(), super::sqlite::AuditError> {
    let map_err = |e: rusqlite::Error| super::sqlite::AuditError::DatabaseError(e.to_string());
    let mut statement = conn
        .prepare("PRAGMA table_info(requests)")
        .map_err(map_err)?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(map_err)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(map_err)?;
    for column in [
        "stream_body",
        "auth_method",
        "api_key_id",
        "api_key_name",
        "user_agent",
        "peer_ip",
        "proxy_request_id",
    ] {
        if !columns.iter().any(|existing| existing == column) {
            conn.execute(
                &format!("ALTER TABLE requests ADD COLUMN {column} TEXT"),
                [],
            )
            .map_err(map_err)?;
        }
    }
    Ok(())
}

fn id(s: &'static str) -> Result<Identifier<'static>, DefinitionError> {
    Identifier::new(s)
}
fn describe(
    name: &'static str,
    columns: &[(&'static str, &'static str, bool, Option<&'static str>)],
    primary_key: &[&'static str],
    unique: &[&[&'static str]],
    foreign_keys: &[(&'static str, &'static str, &'static str)],
) -> Result<TableSpec<'static>, DefinitionError> {
    let keys = |names: &[&'static str]| names.iter().map(|s| id(s)).collect::<Result<Vec<_>, _>>();
    Ok(TableSpec {
        name: id(name)?,
        columns: columns
            .iter()
            .map(|&(name, ty, not_null, default)| {
                Ok(ColumnSpec {
                    name: id(name)?,
                    declared_type: ty.into(),
                    not_null,
                    default: default.map(SqlExpression::trusted).transpose()?,
                })
            })
            .collect::<Result<Vec<_>, DefinitionError>>()?
            .into(),
        primary_key: keys(primary_key)?.into(),
        unique_constraints: unique
            .iter()
            .map(|names| keys(names))
            .collect::<Result<Vec<_>, _>>()?
            .into(),
        foreign_keys: foreign_keys
            .iter()
            .map(|&(column, parent, target)| {
                Ok(ForeignKeySpec {
                    columns: vec![id(column)?].into(),
                    parent_table: id(parent)?,
                    parent_columns: vec![id(target)?].into(),
                    on_update: ForeignKeyAction::NoAction,
                    on_delete: ForeignKeyAction::NoAction,
                })
            })
            .collect::<Result<Vec<_>, DefinitionError>>()?
            .into(),
        indexes: vec![].into(),
        unsupported: vec![].into(),
    })
}
fn table(name: &str) -> Result<TableSpec<'static>, DefinitionError> {
    match name {
        "users" => describe(
            "users",
            &[
                ("id", "TEXT", false, None),
                ("email", "TEXT", false, None),
                ("created_at", "TEXT", true, None),
                ("last_seen_at", "TEXT", true, None),
                ("is_enabled", "INTEGER", true, Some("1")),
            ],
            &["id"],
            &[],
            &[],
        ),
        "requests" => describe(
            "requests",
            &[
                ("id", "TEXT", false, None),
                ("timestamp", "TEXT", true, None),
                ("user_id", "TEXT", true, None),
                ("request_path", "TEXT", true, None),
                ("request_body", "TEXT", false, None),
                ("model", "TEXT", false, None),
                ("client_ip", "TEXT", false, None),
            ],
            &["id"],
            &[],
            &[("user_id", "users", "id")],
        ),
        "responses" => describe(
            "responses",
            &[
                ("id", "TEXT", false, None),
                ("request_id", "TEXT", true, None),
                ("timestamp", "TEXT", true, None),
                ("status", "INTEGER", true, None),
                ("response_body", "TEXT", false, None),
                ("latency_ms", "INTEGER", true, None),
                ("tokens_prompt", "INTEGER", false, None),
                ("tokens_completion", "INTEGER", false, None),
            ],
            &["id"],
            &[],
            &[("request_id", "requests", "id")],
        ),
        "api_keys" => describe(
            "api_keys",
            &[
                ("id", "TEXT", false, None),
                ("key_hash", "TEXT", true, None),
                ("plaintext_key", "TEXT", false, None),
                ("user_id", "TEXT", true, None),
                ("name", "TEXT", true, None),
                ("roles", "TEXT", true, Some("'[]'")),
                ("created_at", "TEXT", true, None),
                ("last_used_at", "TEXT", false, None),
                ("revoked", "INTEGER", true, Some("0")),
            ],
            &["id"],
            &[&["key_hash"]],
            &[("user_id", "users", "id")],
        ),
        "runners" => describe(
            "runners",
            &[
                ("id", "TEXT", false, None),
                ("name", "TEXT", true, None),
                ("mac_address", "TEXT", false, None),
                ("machine_type", "TEXT", false, None),
                ("last_seen_at", "TEXT", true, None),
                ("available_models", "TEXT", false, None),
            ],
            &["id"],
            &[],
            &[],
        ),
        "runner_metrics" => describe(
            "runner_metrics",
            &[
                ("runner_id", "TEXT", true, None),
                ("model_class", "TEXT", true, None),
                ("sample_count", "INTEGER", true, Some("0")),
                ("total_ms", "INTEGER", true, Some("0")),
                ("min_ms", "INTEGER", false, None),
                ("max_ms", "INTEGER", false, None),
                ("last_updated_at", "TEXT", true, None),
            ],
            &["runner_id", "model_class"],
            &[],
            &[],
        ),
        "response_inference_metrics" => describe(
            "response_inference_metrics",
            &[
                ("response_id", "TEXT", false, None),
                ("request_id", "TEXT", true, None),
                ("runner_id", "TEXT", false, None),
                ("model_class", "TEXT", false, None),
                ("requested_model", "TEXT", false, None),
                ("resolved_model", "TEXT", false, None),
                ("engine_type", "TEXT", false, None),
                ("context_window", "INTEGER", false, None),
                ("prompt_tokens", "INTEGER", false, None),
                ("completion_tokens", "INTEGER", false, None),
                ("prompt_eval_ms", "INTEGER", false, None),
                ("completion_eval_ms", "INTEGER", false, None),
                ("total_inference_ms", "INTEGER", false, None),
                ("prompt_tokens_per_sec", "REAL", false, None),
                ("completion_tokens_per_sec", "REAL", false, None),
                ("created_at", "TEXT", true, None),
            ],
            &["response_id"],
            &[],
            &[
                ("request_id", "requests", "id"),
                ("response_id", "responses", "id"),
            ],
        ),
        "model_context_metrics" => describe(
            "model_context_metrics",
            &[
                ("resolved_model", "TEXT", true, None),
                ("runner_id", "TEXT", true, None),
                ("context_window", "INTEGER", true, None),
                ("sample_count", "INTEGER", true, Some("0")),
                ("prompt_tokens_total", "INTEGER", true, Some("0")),
                ("completion_tokens_total", "INTEGER", true, Some("0")),
                ("prompt_weighted_tps_total", "REAL", true, Some("0")),
                ("completion_weighted_tps_total", "REAL", true, Some("0")),
                ("prompt_tps_min", "REAL", false, None),
                ("prompt_tps_max", "REAL", false, None),
                ("completion_tps_min", "REAL", false, None),
                ("completion_tps_max", "REAL", false, None),
                ("last_updated_at", "TEXT", true, None),
            ],
            &["resolved_model", "runner_id", "context_window"],
            &[],
            &[],
        ),
        _ => unreachable!("unknown bootstrap table"),
    }
}

fn plan(spec: TableSpec<'static>) -> Result<CreationPlan, DefinitionError> {
    create_plan(&SchemaSnapshot {
        namespace: "simple-ai/audit-bootstrap".into(),
        database: id("main")?,
        version: 0,
        tables: vec![spec].into(),
    })
}
pub(super) fn create_table(
    conn: &rusqlite::Connection,
    name: &str,
) -> Result<(), super::sqlite::AuditError> {
    let commands = plan(table(name).map_err(error)?).map_err(error)?;
    for command in commands.statements {
        conn.execute_batch(&command.replacen("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1))
            .map_err(|e| super::sqlite::AuditError::DatabaseError(e.to_string()))?;
    }
    Ok(())
}
fn error(e: DefinitionError) -> super::sqlite::AuditError {
    super::sqlite::AuditError::DatabaseError(e.to_string())
}
pub(super) fn create_index(
    conn: &rusqlite::Connection,
    table_name: &str,
    name: &'static str,
    column: &'static str,
) -> Result<(), super::sqlite::AuditError> {
    let mut spec = table(table_name).map_err(error)?;
    spec.indexes = vec![IndexSpec {
        name: id(name).map_err(error)?,
        terms: vec![IndexTerm {
            column: id(column).map_err(error)?,
            collation: id("BINARY").map_err(error)?,
            descending: false,
        }]
        .into(),
        unique: false,
        predicate: None,
    }]
    .into();
    for command in plan(spec).map_err(error)?.statements.into_iter().skip(1) {
        let config = rusqlite::config::DbConfig::SQLITE_DBCONFIG_DQS_DDL;
        let previous = conn.db_config(config).map_err(database_error)?;
        conn.set_db_config(config, false).map_err(database_error)?;
        // A missing quoted index column must remain an error, not a DQS string literal.
        let result = conn.execute_batch(&command.replacen(
            "CREATE INDEX ",
            "CREATE INDEX IF NOT EXISTS ",
            1,
        ));
        let restored = conn.set_db_config(config, previous);
        result.map_err(database_error)?;
        restored.map_err(database_error)?;
    }
    Ok(())
}

fn database_error(e: rusqlite::Error) -> super::sqlite::AuditError {
    super::sqlite::AuditError::DatabaseError(e.to_string())
}

#[cfg(test)]
mod tests {
    use crate::audit::sqlite::AuditLogger;
    use rusqlite::Connection;

    fn metadata(conn: &Connection) -> Vec<String> {
        let mut result = Vec::new();
        let names = conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").unwrap().query_map([], |r| r.get::<_,String>(0)).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap();
        for table in names {
            let columns = conn.prepare("SELECT cid,name,type,\"notnull\",dflt_value,pk FROM pragma_table_info(?1) ORDER BY cid").unwrap().query_map([&table], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,bool>(3)?,r.get::<_,Option<String>>(4)?,r.get::<_,i64>(5)?))).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap();
            result.push(format!("{table}:columns:{columns:?}"));
            let fks = conn.prepare("SELECT id,seq,\"table\",\"from\",\"to\",on_update,on_delete,\"match\" FROM pragma_foreign_key_list(?1) ORDER BY \"from\",\"to\"").unwrap().query_map([&table], |r| Ok((r.get::<_,String>(2)?,r.get::<_,String>(3)?,r.get::<_,String>(4)?,r.get::<_,String>(5)?,r.get::<_,String>(6)?,r.get::<_,String>(7)?))).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap();
            result.push(format!("{table}:fks:{fks:?}"));
            let indexes = conn.prepare("SELECT name,\"unique\",origin,partial FROM pragma_index_list(?1) ORDER BY name").unwrap().query_map([&table], |r| Ok((r.get::<_,String>(0)?,r.get::<_,bool>(1)?,r.get::<_,String>(2)?,r.get::<_,bool>(3)?))).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap();
            for index in indexes {
                let terms = conn.prepare("SELECT seqno,cid,name,\"desc\",coll,\"key\" FROM pragma_index_xinfo(?1) ORDER BY seqno").unwrap().query_map([&index.0], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,Option<String>>(2)?,r.get::<_,bool>(3)?,r.get::<_,String>(4)?,r.get::<_,bool>(5)?))).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap();
                result.push(format!("{table}:index:{index:?}:{terms:?}"));
            }
        }
        result
    }

    #[test]
    fn shared_bootstrap_matches_legacy_metadata_and_defaults() {
        let legacy = Connection::open_in_memory().unwrap();
        crate::audit::legacy_bootstrap_tests::bootstrap(&legacy).unwrap();
        super::migrate_request_attribution(&legacy).unwrap();
        let path =
            std::env::temp_dir().join(format!("simple-ai-07e-{}.sqlite", uuid::Uuid::new_v4()));
        let logger = AuditLogger::new(path.to_str().unwrap()).unwrap();
        drop(logger);
        let shared = Connection::open(&path).unwrap();
        assert_eq!(metadata(&shared), metadata(&legacy));
        for conn in [&legacy, &shared] {
            conn.execute(
                "INSERT INTO users(id,created_at,last_seen_at) VALUES('u','now','now')",
                [],
            )
            .unwrap();
            assert_eq!(
                conn.query_row("SELECT is_enabled FROM users WHERE id='u'", [], |r| r
                    .get::<_, i64>(0))
                    .unwrap(),
                1
            );
            conn.execute("INSERT INTO api_keys(id,key_hash,user_id,name,created_at) VALUES('k','hash','u','key','now')",[]).unwrap();
            assert_eq!(
                conn.query_row("SELECT roles,revoked FROM api_keys WHERE id='k'", [], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
                })
                .unwrap(),
                ("[]".into(), 0)
            );
            assert!(conn.execute("INSERT INTO api_keys(id,key_hash,user_id,name,created_at) VALUES('k2','hash','u','key','now')",[]).is_err());
        }
        shared.pragma_update(None, "user_version", 83).unwrap();
        drop(shared);
        let logger = AuditLogger::new(path.to_str().unwrap()).unwrap();
        assert_eq!(logger.find_or_create_user("u", None).unwrap().id, "u");
        drop(logger);
        let conn = Connection::open(&path).unwrap();
        assert_eq!(
            conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0))
                .unwrap(),
            83
        );
        assert_eq!(metadata(&conn), metadata(&legacy));
        drop(conn);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_file_reopens_without_changing_records_with_nullable_attribution() {
        let path = std::env::temp_dir().join(format!(
            "simple-ai-07e-legacy-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let conn = Connection::open(&path).unwrap();
        crate::audit::legacy_bootstrap_tests::bootstrap(&conn).unwrap();
        conn.execute(
            "INSERT INTO users(id,created_at,last_seen_at) VALUES('legacy','now','now')",
            [],
        )
        .unwrap();
        super::migrate_request_attribution(&conn).unwrap();
        let before = metadata(&conn);
        drop(conn);
        let logger = AuditLogger::new(path.to_str().unwrap()).unwrap();
        assert_eq!(
            logger.find_or_create_user("legacy", None).unwrap().id,
            "legacy"
        );
        drop(logger);
        let conn = Connection::open(&path).unwrap();
        assert_eq!(metadata(&conn), before);
        drop(conn);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn malformed_legacy_file_keeps_bootstrap_failure_order() {
        let legacy = Connection::open_in_memory().unwrap();
        legacy
            .execute_batch("CREATE TABLE requests(id TEXT PRIMARY KEY)")
            .unwrap();
        assert!(crate::audit::legacy_bootstrap_tests::bootstrap(&legacy).is_err());
        let path = std::env::temp_dir().join(format!(
            "simple-ai-07e-failed-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let shared = Connection::open(&path).unwrap();
        shared
            .execute_batch("CREATE TABLE requests(id TEXT PRIMARY KEY)")
            .unwrap();
        drop(shared);
        assert!(AuditLogger::new(path.to_str().unwrap()).is_err());
        let shared = Connection::open(&path).unwrap();
        assert_eq!(metadata(&shared), metadata(&legacy));
        assert_eq!(
            shared
                .query_row(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='runners'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
        drop(shared);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn index_execution_restores_dqs_on_error_and_keeps_existing_index_acceptance() {
        let conn = Connection::open_in_memory().unwrap();
        let config = rusqlite::config::DbConfig::SQLITE_DBCONFIG_DQS_DDL;
        conn.set_db_config(config, true).unwrap();
        conn.execute_batch("CREATE TABLE requests(id TEXT PRIMARY KEY)")
            .unwrap();
        assert!(
            super::create_index(&conn, "requests", "idx_requests_timestamp", "timestamp").is_err()
        );
        assert!(conn.db_config(config).unwrap());
        conn.execute_batch("CREATE INDEX idx_requests_timestamp ON requests(id)")
            .unwrap();
        super::create_index(&conn, "requests", "idx_requests_timestamp", "timestamp").unwrap();
        assert!(conn.db_config(config).unwrap());
        conn.set_db_config(config, false).unwrap();
        super::create_index(&conn, "requests", "idx_requests_timestamp", "timestamp").unwrap();
        assert!(!conn.db_config(config).unwrap());
    }
}
