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
