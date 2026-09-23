# Step 06 background work

Starting master: `0523c03`. Shared source:
`06c7531dfcd4149a0947b4b74a79e96bf245639c`.

06a adopts WorkTracker in the backend BatchDispatcher. Named request guards are
reserved before route planning or runner-capacity reservation, moved into spawned
execution, and released after delivering the response (including abandoned
callers). Stop still occurs only after HTTP consumers drain. Admission is now
permanently closed after drain, with explicit caller errors and no runner request
or capacity reservation for late dispatch. Routing, audit and response semantics
for accepted requests are unchanged. Inference-runner protocol/process ownership
remains application-owned.

06b now adopts `BatchReadiness` in the production gateway's per-model dispatch
check (shared revision `8edcf14`). Full batches dispatch immediately; partial
batches require the configured minimum and normal or saturated age threshold.
Model identity, cancellation pruning, routing, live runner reservations and
request ownership stay in the gateway. This is scoped scheduling-primitive
adoption, not replacement of the inference protocol or runner registry.
06c is N/A for independently managed background jobs: batch readiness ages are
not execution/queue-expiry budgets, and outbound inference request errors/timeouts
remain routing/client behavior. No job circuit/pause/retry policy is introduced.

Baseline and final backend suites: 314 passed, one existing ignored doctest.
The real loopback mock-runner test verifies four accepted requests drain before
responses are consumed, both runners' capacities return to zero, then a late
batch returns errors without making further outbound requests or reserving slots.
The full suite also covers process lifecycle and HTTP responses. Backend
all-target Clippy completes with 12 warnings. Changed-file rustfmt and diff checks
pass; workspace-wide formatting reports existing differences in unrelated files,
which were not reformatted. Inference-runner/GPU/browser suites were not rerun.

Used isolated worktree, temporary test databases and loopback listeners with
three Cargo jobs and debug info disabled. Unrelated README/semantic-scoring work
in the original checkout is outside this migration.

06b verification (2026-09-23): baseline 314 backend tests; final 315 passed,
with the same one ignored doctest. Existing loopback mock-runner tests cover
capacity release and drain; readiness tests cover size, age, saturation, minimum,
model isolation and empty queues. Backend all-target Clippy completes with
existing warnings. GPU/inference-runner and browser suites were not rerun.
