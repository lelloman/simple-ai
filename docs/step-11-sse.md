# Owned admin SSE

`GET /admin/runners/events` uses owned `web::sse::{Event, KeepAlive, Sse}`
and shared `IntoResponse` directly. Reviewed shared revision:
`96c542c2935606cbae48573e6d5ee634ed24970c`, pinned in `simple-server.rev`
and the active README checkout instructions. The workspace now enables `web`
rather than `web-compat`; source audit finds no backend/compatibility imports
in the backend, runner or common Rust sources.

JWT query auth and admin-role policy, named event JSON/schema/order, bounded
broadcast subscriptions, lag/serialization-error skipping, audit ownership and
15-second keepalive are unchanged. SSE remains the existing admin compatibility
protocol for clients; no new replay or revocation policy is introduced. The
inference runner's model response streaming is application-owned HTTP body data,
not this event-builder migration.

## Verification

- Baseline backend/common: **374 passed, one existing ignored doctest** against
  old shared pin `ca98a4159e1cb0dd7b9db2faa9a076d198b7973b`.
- Two new production-router SSE contracts pass before and after migration. Real
  HTTP uses fixture RSA-signed JWTs and checks invalid token/audience (401),
  non-admin (403), exact SSE/cache headers, all four named event types with
  Unicode/newline JSON, incremental delivery and shutdown after disconnect.
  Paused-time body checks verify the 15-second heartbeat, reset and bounded
  broadcast lag recovery without synthesizing error events. Public JWKS is
  derived from the repository's test-only RSA key pair.
- Final backend/common: **376 passed, same ignore**. Locked full workspace build
  passes. Non-strict backend/common all-target Clippy passes with 11 existing
  backend and three common warnings; no findings in the new SSE tests. Strict
  Clippy stops at the unchanged common `derivable_impls` findings.
- Workspace formatting retains previously documented unrelated differences;
  new tests pass rustfmt and the change passes `git diff --check`. Production
  import/conversion changes retain surrounding formatting.
- Fresh-checkout full workspace tests fail before SSE edits because an existing
  runner test includes ignored `scripts/configs/rtx.toml`. That deployment file
  was not copied or committed, so no full-workspace test pass is claimed.
  Docker, Android, browser, GPU and deployed OIDC checks were not rerun.

Work was isolated from `master` at `e12969b` in `migration/owned-sse`. Existing
README semantic-scoring edits and untracked files are preserved through local
branch rebase/integration and cleanup. Central trackers record the commit and
remaining exposure. No push or deployment.
