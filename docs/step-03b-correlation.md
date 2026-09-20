# Step 03b: HTTP request correlation assessment

Status: **N/A for the current service behavior** (2026-09-20).

`backend/src/main.rs` composes HTTP routes, authentication, rate limiting, CORS
and `logging::request_logger`. `backend/src/logging.rs` records method, path,
status and duration without HTTP request IDs or header propagation.

`backend/src/models/request.rs::Request::new` generates a UUID for a durable
logged inference request. `backend/src/routes/chat.rs` persists that identity
before inference; queues, telemetry, cancellation and gateway/runner protocol
messages continue to use it beyond an HTTP middleware scope. These are business
keys, not caller-selected HTTP correlation IDs. Their UUID format, persistence,
and ownership remain unchanged.

The optional shared `correlation` contract was reviewed at simple-server
`52e1922bcfff44b4ab55a1b5374b35c6089b6e37`. No current requirement justifies
adding a new externally visible response header or identity policy merely to
enable a Cargo feature. Existing dependency features and source pins remain
unchanged. Reassess if HTTP request correlation is introduced later.

## Verification and integration

Inspected production router construction and middleware, and searched source for
request/correlation IDs and header names. Baseline and final `git diff --check`
pass. This is a documentation-only applicability assessment; no runtime code or
dependencies changed, so application tests/builds were not rerun.

The clean active development branch was `master`, starting at `36c650d3bfe424912664dd43435fb45748038128`.
The assessment was recorded in the dedicated `migration/step03b-correlation`
branch and isolated sibling worktree. Integration rebases `master` onto that
branch, verifies the resulting tree and ancestry, then removes only this
migration worktree and branch. Existing unrelated worktrees are preserved.
No push or deployment is included.
