# Step 03a: shared logging

Both production entry points (`backend/src/main.rs` and
`inference-runner/src/main.rs`) call their `logging_setup::init` adapter, which
installs `simple_server::logging`. Source pin:
`71755b5e15ada9b22484559146ebaf4d82c91255`.

The backend retains strict `RUST_LOG` parsing with the configured level fallback;
the runner retains the INFO fallback. Text, stdout, targets, nested spans,
NO_COLOR behavior, and the explicit log-facade bridge are preserved. Filter
parsing remains application-owned. Shared initialization errors propagate through
the existing startup failure path. Request tracing and correlation are unchanged.

Implemented in an isolated worktree from master `d9e5bba`. Baseline workspace
tests passed 441 tests, with one ignored doctest. Final workspace tests pass 445,
including fresh-process comparisons exercising the production adapter across
filter/ANSI/format/destination/span/log-bridge combinations and checking the child
actually ran. Workspace builds and changed-file formatting pass. Strict Clippy
stops on three existing derivable_impls findings in simple-ai-common ocr.rs and
speech.rs. No unrelated lint cleanup is included.

Production startup checks used temporary configs: the backend emitted its startup
record before the expected loopback-only OIDC connection refusal; the runner
started with engines disabled and exited cleanly on SIGTERM. No production model,
database, or remote inference service was used. Container/browser matrices were
not rerun.

The runner's existing compile-time test includes ignored scripts/configs/rtx.toml;
a copy of the original ignored file was required in the worktree to run its
configuration-deserialization test. That test only parses the file and compares
configuration values; it does not connect to endpoints or execute configured
commands. The fixture is not included in this commit. Original development work
and ignored fixtures remain untouched. Rebase master onto this migration before
removing the temporary worktree and branch; no push is authorized.
