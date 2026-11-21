# Audit database durability

The gateway records every call in the SQLite audit database
(`SIMPLEAI__DATABASE__URL`, in production `~/homelab-data/simple-ai/audit.db`).
All audit reads and writes share one connection behind one lock, so anything
that holds that lock for long stalls request logging.

## Retention

| Data | Kept for | Then |
| --- | --- | --- |
| Request, streamed and response bodies (`requests.request_body`, `requests.stream_body`, `responses.response_body`) | 14 days | set to `NULL` |
| Request, response and inference-metric rows | 365 days | deleted |
| Aggregates (`runner_metrics`, `model_context_metrics`), users, API keys, runners | forever | — |

The same ages apply to every model class, including `class:tagger`.

The ages are the constants `BODY_RETENTION_DAYS` and `METADATA_RETENTION_DAYS`
in `backend/src/audit/retention.rs`. The gateway prunes once at startup and
then every hour. Each batch handles at most 500 requests in one transaction and
then releases the audit lock for 20 ms, so a chat request waits at most one
batch. Every run logs `Audit retention prune finished` with the counts and the
longest batch; failures log `Audit retention prune failed`.

After pruning, the admin request history still lists the request with its
status, tokens, latency and attribution; the detail view shows
"No request body recorded." for the cleared bodies.

## Reclaiming space

Pruning frees pages inside the file but does not shrink it; SQLite reuses the
freed pages for new rows. To give the space back to the disk once (for example
after the first prune of a large database), compact the file in a maintenance
window:

1. Wait for the first `Audit retention prune finished` log line after deploying.
2. Check that the disk has free space of at least the database size.
3. Stop the gateway.
4. `sqlite3 audit.db 'PRAGMA wal_checkpoint(TRUNCATE);'`
5. `sqlite3 audit.db "VACUUM INTO 'audit.compact.db';"`
6. `sqlite3 audit.compact.db 'PRAGMA integrity_check;'` must print `ok`.
7. `mv audit.db audit.db.old`, remove any `audit.db-wal` and `audit.db-shm`,
   then `mv audit.compact.db audit.db`. Keep the file owner and mode the same
   as the old one.
8. Start the gateway and check the admin request history.
9. Delete `audit.db.old` once the gateway has run normally for a day.
