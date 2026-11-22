# Audit database durability

The audit database (`[database] url`, usually `audit.db`) is a SQLite file in
WAL mode with `synchronous=NORMAL`. `AuditLogger::new` switches an existing
rollback-journal file to WAL on first open; the mode persists in the file.

## Request path

`/v1/chat/completions` does no audit disk I/O before the request is enqueued:

- API-key and user lookups are reads on a dedicated connection. They never wait
  for the writer or for admin queries.
- `last_used_at` (API keys) and `last_seen_at` (users) are updated by the writer
  thread, so they trail the request by milliseconds. A first-ever user is still
  inserted inline (one write, once per user).
- Request rows, response rows, runner latency metrics, inference metrics and
  streamed-body capture are queued to one `audit-writer` thread that commits
  batches of up to 512 operations in a single transaction. The response
  observer (pressure tracking) also runs on that thread.
- A batch that cannot commit (lock contention beyond the 5 s busy timeout) is
  retried three times, then dropped with an error log.

Other endpoints (embeddings, classifications, ...) still write inline on the
general connection.

## Guarantees

- Audit rows appear shortly after the response, not before it. Admin views can
  lag by the writer queue.
- A process crash or kill loses the queued tail. Graceful shutdown flushes the
  queue.
- With `synchronous=NORMAL` an OS crash or power loss can lose the most recent
  commits (the file stays consistent). Process crashes alone lose nothing that
  was committed.
- Order is preserved: a request row is always queued before its response row.

## Rollout

No schema change. The first start of a build with this behaviour converts the
file to WAL and creates `audit.db-wal` and `audit.db-shm` next to it (the
container's `/data` volume already holds the file). Stop any `sqlite3` shell
holding the file first; if the switch fails the gateway logs
`Cannot switch audit database to WAL mode` and keeps running in the old mode.
Back up or copy the database with `sqlite3 audit.db ".backup out.db"`, not by
copying the main file alone. To revert, run `PRAGMA journal_mode=DELETE;` with
the gateway stopped.

## Retention

| Data | Kept for | Then |
| --- | --- | --- |
| Request, streamed and response bodies (`requests.request_body`, `requests.stream_body`, `responses.response_body`) | 2 days | set to `NULL` |
| Request, response and inference-metric rows | 365 days | deleted |
| Aggregates (`runner_metrics`, `model_context_metrics`), users, API keys, runners | forever | — |

Owner decision (2026-11-21): bodies are kept 2 days and metadata rows 365 days,
for every model class, including `class:tagger`.

### Per-API-key body retention

A bulk client can fill the database with bodies (about 200k requests an hour
from SanCerchino's tagging is roughly 600 MB an hour). `[audit.body_retention_hours]`
in `config.toml` maps an API key name (the name shown in the admin UI, matched
case-insensitively because the config loader lowercases table keys) to its own
body retention in hours:

```toml
[audit.body_retention_hours]
SanCerchino = 0   # store no bodies
"batch key" = 6   # clear bodies after 6 hours
```

- `0`: `/v1/chat/completions` stores no request body, no successful response
  body and no streamed body for the key. The request row (model, tokens,
  latency, status, error) is still written. Pruning also clears the key's
  bodies that were stored before the setting, on its next run.
- `N > 0`: the hourly pruner clears the key's bodies once they are `N` hours
  old. Values above 48 are capped to the 2-day default; the setting only
  shortens retention.
- Error responses (status 400 and above) keep their response body for the
  normal 2 days, so failures stay debuggable. The request body of a failed
  request is not kept for a `0` key, because the request row is written before
  the outcome is known, and it is cleared at `N` hours for an `N` key. A stream
  that fails after its headers were sent has no body to keep for a `0` key.
- Other keys, JWT users and other endpoints (embeddings, classifications, ...)
  are unchanged.
- The setting costs the request path one map lookup and no database work. The
  pruner starts each key's walk at the default pass's position, so it only
  scans rows between the two cutoffs, in the same bounded batches.
- A change takes effect on restart.

The ages are the constants `BODY_RETENTION_DAYS` and `METADATA_RETENTION_DAYS`
in `backend/src/audit/retention.rs`. The gateway prunes once at startup and
then every hour, on the general connection. Each batch handles at most 500
requests in one `IMMEDIATE` transaction, then pauses 20 ms. While a batch runs,
the `audit-writer` thread waits for it (well inside its 5 s busy timeout) and
admin queries wait on the general connection; request-path reads use their own
connection and are not blocked (WAL). Chat requests only queue their writes,
so they never wait for pruning. Every run logs `Audit retention prune finished`
with the counts and the longest batch; failures log
`Audit retention prune failed`.

A batch costs time in proportion to its size, not to the table, because every
prune statement looks rows up by an index. SQLite enforces the foreign key
from `response_inference_metrics.request_id` to `requests`, so deleting a
request looks for its metrics rows. The gateway therefore creates
`idx_response_inference_metrics_request_id` at startup; the first start after
the upgrade builds it once over the existing metrics rows before serving.
`prune_statements_never_scan_a_table` in `retention.rs` fails if a prune
statement's query plan scans a whole table.

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
