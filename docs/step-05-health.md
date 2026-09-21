# Step 05: health and readiness

Simple AI has two production contracts. Backend `GET /health` is static process
liveness with status `ok` and package version. Inference-runner `GET /health`
polls every registered engine, retains every detailed result, returns 200 when
any engine is healthy, 503 when all registered engines are unhealthy, and keeps
the existing 200 `starting` empty-engine and 200 `ok` OCR-only semantics.

The backend mounts `Probe::liveness()` through `get_service`. The runner uses a
named typed `Check::run` around its complete aggregate, preserving polling,
payloads, order, status decisions, and existing timeout policy. Reviewed shared
revision: `ed245d2d46e9d29aeee7be5202f3a8b8113c9caf`.

## Verification

The runner health module passes before and after with offline locked dependencies,
two jobs, incremental compilation disabled and debug information disabled. Final
fake-engine tests cover mixed and all-unhealthy engines, every-engine polling,
retained details, the named failure, and empty/OCR-only behavior. Backend tests
cover exact GET JSON, HEAD suppression and POST 405. Changed-file formatting and diff checks pass; full repository formatting
retains pre-existing debt. Real engines, GPU/model inference, external providers, Android/browser E2E
and containers were not rerun. Unrelated semantic-evaluation work remains intact.

Final evidence: baseline runner health module 2 passed; final backend route 1
passed; final runner module 4 passed, including the mixed-engine and
all-unhealthy/empty/OCR regressions through Check::run. The first worktree run
failed to compile a test because the ignored scripts/configs/rtx.toml fixture
was absent. It was copied unchanged from the original checkout, as in earlier
migrations, and the rerun passed. Only the intended health files are formatted;
unrelated formatter changes were removed. Full repository formatting retains
pre-existing debt; the changed files and diff whitespace pass.
