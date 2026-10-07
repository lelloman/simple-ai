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

## Server-triggered runner draining

An administrator can request draining through `POST /admin/runners/{id}/drain`
with a JSON body such as `{"action":"drain"}`. The actions are:

| Action | After accepted work finishes |
| --- | --- |
| `drain` (default) | Stay running with admission closed |
| `stop` | Exit the runner service successfully |
| `shutdown` | Request host power-off using `systemctl --no-ask-password poweroff` |
| `reboot` | Request host reboot using `systemctl --no-ask-password reboot` |

The HTTP response is `202 Accepted` with a `request_id`: it means the command was
queued, not that draining has completed. The existing admin command-completion
event reports the runner's acknowledgment. The runner closes admission before
acknowledging; its status becomes `draining`, then `drained` when no work remains.
Both states are excluded from gateway routing. New `/v1` requests return 503,
as does `/health`; new model load/unload commands are rejected. Requests that
race with the command and were already admitted finish normally.

The work count covers HTTP handlers, streaming response bodies, detached decision
and extraction inference, and detached model preparation. Drain has no forced deadline: a stuck
accepted request keeps the runner draining. The ordinary 30-second process exit
budget starts only after `stop` has finished draining. External SIGINT/SIGTERM
retain their existing behavior and can interrupt a drain.

Repeating the same action is idempotent. A drain-only request can be upgraded to
`stop`, `shutdown`, or `reboot`, including after reaching `drained`. Conflicting
terminal actions are rejected. Restarting the runner reopens admission. Drain
state survives gateway reconnects within the runner process. Terminal actions
wait for the acknowledgment to be sent; if the connection drops first, they wait
for successful gateway registration on reconnect.

Power-off/reboot require the service account to have the corresponding systemd
permissions without interactive authentication. Failed host commands are logged;
the runner stays drained and does not automatically retry. A service configured
with `Restart=always` may restart after `stop`; the supplied runner unit uses
`Restart=on-failure`. Existing automatic Wake-on-LAN policies still apply after a
machine powers off: this API does not establish a persistent maintenance hold.
Deploy the updated backend and runner together to use the new command/statuses.

## Building

The workspace now uses the published `lelloman-simple-server` 0.1.0 package,
aliased as `simple-server`, with the opt-in `lifecycle` feature. `Cargo.lock`
pins the registry package and checksum. CI, native runner builds, E2E Compose,
and Docker builds no longer need the sibling source checkout used during the
original lifecycle migration.

```sh
cargo test --locked --workspace --no-fail-fast
docker build -f backend/Dockerfile -t simple-ai-backend .
COMPOSE_PROJECT_NAME=simpleai-step02 bash tests/e2e/run-tests.sh
```

E2E Compose grants 35 seconds before forced shutdown. The README documents the
native runner build; there is no inference-runner Dockerfile in this repository.

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
