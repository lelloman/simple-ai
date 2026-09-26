# Step 11: shared routing and HTTP core

Active branch: `master`; baseline `9648e2d`.
Reviewed shared source: `fb6a9e5a91f40529c761a7a7e44c33a442143ed5`, recorded
in `simple-server.rev` for the existing CI, Docker and README checkout workflow.
Original semantic-evaluation files and modified README are preserved and excluded.

Gateway and inference-runner route groups, ordinary handlers, built-in extractors,
responses, streaming bodies, middleware and HTTP test/mock fixtures now use
`simple_server::web`. Both production serving entry points adopt the shared API;
gateway serving retains direct TCP peer metadata through `serve_with_connect_info`.
Multipart image/audio upload handlers use shared borrowed Multipart/Field/errors.
Body limits, auth/user context, quotas, OIDC, audit, inference admission, runner
supervision, response errors and engine-specific streaming remain unchanged.

Shared MethodRouter::with_state binds runner WebSocket route state independently
of the enclosing gateway. Its backend differential contract verifies GET/HEAD,
POST using outer state and unsupported-method behavior. This preserves existing
route assembly and avoids moving application state ownership.

Explicit remaining boundaries are gateway WebSocket socket/message callbacks,
admin SSE event/keepalive production through `web::compat::response`, and the
tracing observer backend response callback through `web::compat::trace_with_observer`.
Multipart is fully shared. Ordinary routing has no backend conversion.

## Verification

- Baseline/final workspace suites: 462 pass each, one existing ignored doctest.
- An existing runner config test includes ignored `scripts/configs/rtx.toml`;
  that original fixture was copied into the worktree before baseline testing,
  used unchanged for both suites and excluded from commits. A fresh checkout
  without this fixture cannot compile that pre-existing test.
- Existing API suites verify gateway Responses/chat, errors, quotas, streaming,
  multipart uploads, health/CORS/headers and safe tracing. Runner real-process
  E2E verifies mock model discovery/chat streaming, SIGTERM/SIGINT drain and
  runner gateway connection behavior without production inference or data.
- Debug workspace build passes. All-target Clippy completes with repository
  warnings; strict Clippy encounters pre-existing common-crate warnings.
  Workspace formatting already fails on untouched baseline. Unrelated formatting
  is preserved and diff whitespace checks pass.
- Shared all-feature suite: 262 pass; strict Clippy and formatting pass.
- Docker/browser/Android flows, live model downloads and GPU inference were not
  run. CI pins are updated, without publishing or deploying new builds.

Work was committed in isolated branches, integrated by rebasing development
branches onto them, and verified by ancestry/tree equality. Original dirty work
is preserved; owned worktrees, branches, fixture copies, build outputs and scratch
files were removed after integration. No push or deployment is included.
