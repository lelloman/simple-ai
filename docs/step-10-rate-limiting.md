# Step 10 rate-limit canary

The backend now uses `simple_server::rate_limit::KeyedLimiter` in its existing
`/v1` middleware. This replaces governor 0.8 and the local map of per-IP limiters.
Reviewed shared revision: `925ea25153a35e1fdb748add4880bc0e7d384ae6`.

Policy is unchanged: configured requests/minute also defines burst size; each
unit refills at `60 seconds / rpm`. Startup still enables the middleware only
when configured. Keys retain first X-Forwarded-For field/first comma-separated
value, X-Real-IP, peer address and finally `unknown` precedence. No new proxy trust
policy is introduced. The existing response remains 429, `Rate limit exceeded`,
and Retry-After rounded down with a one-second minimum. The existing logger,
route scope and public routes are unchanged. Storage explicitly uses unbounded
compatibility mode, matching the previous map; adding a cap is a separate policy
change. The runner has no corresponding inbound limit to migrate.

Work started from `master` at `739cccc` in an isolated worktree. Original README
and semantic-scoring work are unrelated and excluded from this commit.

Verification:

- Original four rate-limit tests passed before edits.
- Two new identity/real-HTTP contract tests passed against the original governor
  implementation, then against the migration (six rate tests total).
- Full `cargo test -p simple-ai-backend --offline`: **319 passed**, one pre-existing
  ignored doctest. Socket tests ran outside the restricted sandbox.
- `cargo clippy -p simple-ai-backend --all-targets --offline` passed with existing
  warnings in untouched common/backend code; no warnings in rate-limit changes.
- Changed Rust file formatted; dependency graph no longer needs direct governor.

This is production adoption of shared budgets/storage through the existing local
HTTP middleware. It does not claim migration to `RateLimitLayer` or completion of
consumer Axum removal. Pezzottify exercises the shared HTTP layer separately.
