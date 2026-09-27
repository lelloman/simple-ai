# Step 03c: HTTP tracing

Production backend `main` already installs `logging::request_logger` around the complete gateway router. That middleware now delegates span/body observation to simple-server with an application observer preserving INFO header events for every HTTP status. The outer placement observes auth/rate-limit rejections and includes the runner WebSocket HTTP handshake. Domain inference/audit logs are retained; they are not duplicate HTTP middleware events. The inference-runner binary has no equivalent HTTP tracing middleware and is intentionally unchanged.

The production observer uses the owned `simple_server::web::tracing` request,
response metadata, observer, lifecycle and default finish-event contracts. Its
response callback no longer exposes an Axum response type. The admin SSE
producer still uses `simple_server::axum::response::sse` and
`web::compat::response`; that separate streaming migration remains pending.

Validation: baseline `cargo test --locked -p simple-ai-backend` passed 310 tests with one existing ignored doctest; final passed 312 with the same ignore. Two new integration tests verify the production middleware's all-status INFO policy, matched route privacy, unchanged headers/body, lazy SSE body polling, and exactly-once completion/cancellation. Existing smoke/auth/backend tests also pass. All-target Clippy exits successfully with 12 backend and 3 common-library warnings in unchanged code, and no new adapter/test findings. Changed files pass rustfmt and diff whitespace checks. Full inference-runner/Android/GPU/browser/deployed-OIDC checks were not run; repository-wide formatting is not claimed.

Unrelated local semantic-scoring work (README and untracked
scripts/docs/fixtures) is preserved separately from this migration. The README
deployment instructions only add the active reviewed revision beside the
existing `simple-server.rev` checkout command.

The original tracing migration reviewed `adc1640bde4ac8f934ed454c8d6c5e264a6a2790`.
The owned-contract follow-up reviews
`ca98a4159e1cb0dd7b9db2faa9a076d198b7973b`, recorded in
`simple-server.rev`; the CI, Docker and README checkout workflow consumes this
pin. Subscriber configuration remains application-owned.

The event schema intentionally changes to a safe `http.request` span with matched route templates (or `<unmatched>`), `http.response_headers` with header latency, and exactly one `http.finished` body lifecycle event. Raw request paths and query strings are no longer logged by this middleware. Header timing remains response-creation time, while body completion/cancellation is independently observed without buffering. Status, headers, response bodies, metrics and domain events keep their application behavior. Body completion is not evidence of client receipt; cancellation does not prove client disconnect. Upgrades hand off after the HTTP response and do not trace WebSocket session lifetime.

Work was performed in a dedicated branch/worktree based on the established local `master`, with baseline checks completed before editing. Integration rebases `master` onto the tested migration, verifies ancestry/tree, then removes the temporary branch/worktree. See the central simple-server trackers for the final commit and cleanup evidence. No push or deployment is part of this migration.

## Owned tracing follow-up verification (2026-09-27)

This follow-up starts from `master` at `3eedbd6` and reviews simple-server
`ca98a4159e1cb0dd7b9db2faa9a076d198b7973b`. The fresh baseline and final
`cargo test --locked --workspace` runs each pass 464 tests with one ignored
doctest. The focused production middleware contract passes both tests, including
all-status INFO headers, safe route labels, private-data exclusion, unchanged
responses, lazy streaming and completion/cancellation outcomes.

`cargo build --locked --workspace`, non-strict workspace all-target Clippy,
focused rustfmt and `git diff --check` pass. Strict workspace Clippy stops at the
same three pre-existing `derivable_impls` findings in `simple-ai-common` before
checking the rest of the workspace; repository-wide rustfmt retains existing
unrelated formatting differences. Checks use two build jobs and the unchanged
ignored `scripts/configs/rtx.toml` compile fixture. The full suite needs local
ephemeral sockets and fixture-process execution; a sandboxed attempt was denied
those operations, while the unrestricted rerun passed.

No Android, browser, GPU/model-download, Docker or deployed-service checks are
claimed for this import-only migration. Admin SSE remains on the compatibility
response bridge. No push or deployment is included.
