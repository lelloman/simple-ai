# Step 01: centralized Axum

The backend and inference runner use `simple-server` at Git revision
`46a36391ccca522f3ec9aa24206f2b231162dd96`, which pins Axum 0.8.9.
Their manifests no longer declare Axum directly; Rust source and tests use
`simple_server::axum`. The backend enables `ws` and `multipart`; the runner
requires `multipart`.

The upgrade from Axum 0.7.9 changes named route parameters to braces and converts
outgoing Axum WebSocket text/ping payloads to the new types. Optional connection
information uses a fallible `ConnectInfo` extractor, preserving the previous
behavior when a router is exercised without a socket address. The independent
runner WebSocket client remains on tokio-tungstenite 0.24; it is not an Axum
server dependency.

## E2E baseline established first

Commit `1eb5974` adds a test that launches the actual inference-runner binary
with temporary configuration and a local mock Ollama engine. It checks health,
model discovery, chat proxying and usage, and invalid-request rejection. This
passed before the dependency migration. No GPU, production credentials, or
external inference engine is used.

The existing gateway Docker E2E fixture was initially rejected because it
advertised runner protocol version 1 while the server requires version 2.
Updating the fake runner to version 2 restored all four scenario groups on the
pre-migration backend. The E2E launcher also now tears down its Compose project
on failure. These fixture changes were committed before migration.

## Verification (2026-09-19)

- `cargo test --locked --workspace --no-fail-fast`: 440 passed before and after,
  including the new actual-runner E2E; one existing doctest is ignored.
- Docker gateway E2E: all four groups pass before and after using the actual
  backend, mock OIDC provider, and two fake runners: authentication (7 checks),
  model routing (1), concurrent workload routing (1 check covering 6 requests),
  and speculative wake priority (3). These exercise runner WebSocket
  registration and HTTP request forwarding, but do not qualify real GPU
  engines or physical Wake-on-LAN behavior.
- Backend Docker build with Rust 1.91 passes; the resulting image runs the E2E.
- `cargo tree --locked -i axum`: one Axum version, 0.8.9, through simple-server.
- `git diff --check` passes.
- Strict workspace Clippy stops at the same three pre-existing
  `derivable_impls` findings in simple-ai-common before and after migration.
  It is not a clean Clippy result for the whole workspace.
- Workspace formatting already fails on the baseline. Existing broad
  formatting differences were not rewritten as part of this migration.

Commands use `CARGO_TARGET_DIR=/tmp/simple-server-simple-ai-target`,
`CARGO_PROFILE_DEV_DEBUG=0`, and `CARGO_PROFILE_TEST_DEBUG=0` for host builds.
Docker verification used an isolated `COMPOSE_PROJECT_NAME=simpleai-step01`
with `tests/e2e/docker-compose.yml`; its containers and network were removed.
The four Python scenario groups were run separately so an early failure could
not hide results from later groups.

This is a local migration commit, not a deployment or publication.
