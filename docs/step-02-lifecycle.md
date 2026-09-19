# Step 02: lifecycle / main()

Both the backend and inference runner retain their configuration, logging,
application state, and Tokio runtimes. They now use shared SIGINT/SIGTERM
registration, HTTP bind/serve helpers, and one 30-second shutdown deadline per
process. Backend peer-address extraction remains enabled. The runner's version
command still exits before loading configuration or installing signals.
Unexpected participating-service completion, serving failure, or deadline expiry
returns an error and exits nonzero without waiting indefinitely for Tokio's
blocking-task teardown. Startup work is outside the drain budget.

## Backend ordering

HTTP starts graceful draining when shutdown is requested. Affinity invalidation
and batch dispatch keep running until HTTP has finished, so requests already
waiting in the batch queue can complete. HTTP completion then stops both workers.
The dispatcher finishes its current dispatch and joins tracked dispatched requests,
including requests whose HTTP callers have disconnected. All of this shares the
same deadline; there is no new full budget for each phase.

Upgraded runner/admin WebSocket sessions, speculative wake tasks, scheduler
background work, and detached stream audit finalizers retain their existing
ownership. In particular, keeping runner connections alive during HTTP draining
preserves the registry used by active inference requests. This migration does not
claim to drain every upgraded connection or detached application task. Queued
work with no remaining HTTP consumer is not promised execution during shutdown.

## Inference runner

The runner coordinates HTTP and its optional gateway client. Shutdown cancels
gateway connection/reconnection, status collection, and command handling while
HTTP drains its active responses. Heartbeats now belong to the connection future
instead of a detached task; connection EOF also returns to the reconnect loop.
Stopping the client drops the gateway socket, without a guaranteed WebSocket
close handshake. Gateway commands in flight can be cancelled.

Managed engine processes retain their existing ownership and Drop/kill policies;
external engines such as Ollama are not stopped or unloaded by this change.
There is no new guarantee that every engine subprocess or background extraction
job has been joined. Real GPU engines and fleet deployment were not exercised.

## Building

The workspace uses `../simple-server` with the opt-in `lifecycle` feature.
`simple-server.rev` records reviewed source revision
`c5359079ff4fad0b4b0359e8c88880dbbbc4eeb5`. Run
`bash scripts/checkout-simple-server.sh` to create or verify a clean checkout at
that revision. The helper refuses to overwrite an existing checkout; coordinated
local development can use the path dependency directly. The revision must be
published to the configured remote before fresh remote/CI checkouts can fetch it.
A Cargo lockfile does not pin the contents of a path dependency.

All three CI jobs check out the reviewed dependency before Cargo commands.
The backend Dockerfile uses a named source context, supplied by E2E Compose and
the deployment build script. No deployment or push was performed.

```sh
cargo test --locked --workspace --no-fail-fast
docker build --build-context simple-server-source=../simple-server \
  -f backend/Dockerfile -t simple-ai-backend .
COMPOSE_PROJECT_NAME=simpleai-step02 bash tests/e2e/run-tests.sh
```

`SIMPLE_SERVER_CONTEXT` overrides the sibling source path in Compose and the
server deployment script. E2E Compose grants 35 seconds before forced shutdown.
Host runner builds also require the sibling checkout. The README documents the native runner build; there is no
inference-runner Dockerfile in this repository.

## Verification (2026-09-19)

- Untouched baseline: 440 Rust tests passed, one existing doctest ignored.
- Migrated workspace: 441 Rust tests passed, same ignored doctest. The runner
  process tests send SIGTERM/SIGINT during delayed upstream inference, verify
  the full response, zero exit status, and released HTTP listener. One case
  includes a real WebSocket gateway fixture and checks socket disconnection.
- Existing batch distribution coverage now stops the dispatcher before reading
  response channels and verifies dispatched work and capacity reservations finish.
- Backend release image and all four Docker gateway E2E groups passed:
  authentication (7), model routing (1), workload routing (6 requests), and
  speculative wake priority (3). New release-container SIGTERM/SIGINT checks
  both exit successfully in under half a second. Compose resources were removed.
- Compose validation, shell syntax, changed entry-point formatting, and diff
  checks pass. Workspace formatting retains pre-existing differences.
- Strict workspace Clippy still stops at the same three pre-existing
  `derivable_impls` findings in `simple-ai-common`; this is not a clean full
  workspace Clippy result.

Host checks used `CARGO_TARGET_DIR=/tmp/simple-server-simple-ai-target`,
`CARGO_PROFILE_DEV_DEBUG=0`, and `CARGO_PROFILE_TEST_DEBUG=0`.
