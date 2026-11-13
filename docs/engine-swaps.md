# Shared-GPU engine swaps (AI-67)

Engines in the same resource group (for example RTX's `cuda:0`) cannot run
simultaneously. Model acquisition queues by **engine and local model**, after
alias resolution. Once the current batch's inference leases have drained, the
oldest waiting request selects the next model. All requests for that model
already waiting at that instant form a batch. While no competing model is
waiting, new same-model requests join immediately for continuous batching.
Once a competitor queues, admission freezes to a finite cohort. They keep ownership through
startup and their complete responses, including streaming bodies. The next model
cannot unload them between readiness and inference. Arrivals after a competitor queues
wait for a later batch, so a continuous tag stream cannot starve `code:fast`.
There is no configurable residency timer and no background retry/replay loop.

Waiting happens in the HTTP caller's future. Disconnecting removes that request
from the queue; it cannot subsequently trigger a load or unload. The optional
gateway batch dispatcher also skips abandoned requests and cancels an active
proxy when its response receiver disappears. Online chat dispatch goes directly
to HTTP, including cold models, rather than preparing via an independent
WebSocket `LoadModel` command. The HTTP request owns both preparation and
inference. Sleeping-host wake/preparation, intentional speculative wakeups and
explicit administrative load commands retain their existing behavior; these
are not inference requests with a connected HTTP client.

Non-streaming gateway proxy calls retain their 300-second total timeout. The
gateway additionally sends `X-Simple-AI-Deadline` as an absolute Unix timestamp
in milliseconds; the runner checks it before acquisition and bounds the queue
wait/startup wait by the remaining time. An expired waiter returns 504. A direct
runner call or an older gateway may omit this header. Streaming chat has no total
gateway deadline and relies on connection cancellation. No WebSocket message or
required request-body field changes; older runners ignore the new HTTP headers.

An external start/stop **already begun** is allowed to finish after cancellation,
with its resource reservation retained until it finishes, so another engine
cannot start over a half-stopped process. It does not run inference for the gone
caller. Provider inference already executing retains its existing lifecycle
cleanup rules; quiesce waits for that cleanup before releasing GPU memory.
Cancellation does not immediately kill an engine shared with live requests.

Gateway dispatch logs include `proxy_request_id`, runner and model; the same ID
is sent as `X-Request-Id`. Runner load/revalidation and unload logs include this
request ID, target engine/model, and the evicted engine/model when known.
WebSocket preparations use their command request ID. Failed load/unload logs
carry that ID and the engine error. A load/revalidation entry for a healthy
resident model can be a no-op; provider startup logs show actual process starts.

## Verification and rollout

Registry fake-engine tests enqueue alternating tag/code requests, assert that
both tag requests and both code requests run in useful batches with only one
switch each, and verify cancelled queue entries cause no loads. A real local
runner HTTP test disconnects a queued socket and expires a gateway deadline.
Existing tests cover a cancelled startup retaining the gate, failed switches,
dead providers and backend revalidation. Gateway tests cover cold HTTP admission
without load commands and batch cancellation before and during dispatch.

Operator rollout: after review/merge, drain RTX and install/restart the updated
runner first, then deploy the updated backend image. No model, provider script,
configuration or Halo change is required. Restarting the old runner clears its
already detached queue; only the operator may do this. Qualify concurrent real
tag and `code:fast` traffic and cancelled clients after rollout, checking request
IDs and useful completions between swaps. Agents do not access or restart RTX.

## AI-71 concurrency diagnosis and qualification

The old registry batch queue held all later arrivals behind the current cohort,
even for a warm same-model engine with no competitor. A regression holding one
same-model lease while submitting 64 fake 200ms requests timed out at 2 seconds
before the fix; after the fix they complete in about 203ms. Runner HTTP tests
exercise the real registry and managed tag provider: 64 tagger requests reach the
fake provider concurrently (peak 64), finishing in about 326ms, and 32 teacher
requests finish in about 273ms. These CPU measurements establish concurrency,
not live GPU throughput. The gateway dispatcher separately tests 64 overlapping
200ms HTTP requests and capacity discovery after an initial fallback of 1.
The capacity cache is removed: dispatch reads current model-specific runner
status each time, including heartbeat discovery changes.

The tagger already reports `engine_type=tagger`, capacity 256; teacher tag reports
32. Both use the corrected resource queue. Summaries use that queue too but
intentionally retain their separate synchronous provider contract: one HTTP
request containing a batch of cards, capacity 1. This change does not convert
summaries to the async chat provider or admit concurrent summary batches.

At INFO level, follow `Gateway request enqueued`, `Gateway queue dispatch`
(`queue_ms`), `Dispatching runner request` (`proxy_request_id`, inside the
`queued_chat` span carrying the queued request ID), `Runner request received`,
`Runner lease acquired` (`lease_ms`), `Tag provider response complete`
(`provider_ms`, inside `runner_chat` carrying that proxy ID), `Runner response
complete` (`inference_ms`, `total_ms`), and `Gateway runner response complete` /
`Gateway queued response complete`. Log timestamps provide the cross-process
stage times; durations use local monotonic clocks. Requests still travel over
HTTP; WebSocket carries discovery/control, not these inference payloads.
No new protocol fields or configuration are required.

Operator rollout after merge: drain and update the RTX runner first, then deploy
the backend image, retaining AI-70 config/provider/checkpoint unchanged. The
registry fix is shared runner code; include it in subsequent Halo binary updates
(no Halo configuration or provider change). Existing runners remain compatible.
Re-run RTX `~/tagger-ft-4cc1f8a/tools/tagger-ft/bench_gw.py` with SanCerchino's key,
the same inputs and 64–128 concurrency. At least 40 domains/s and the stage logs
must be checked live; agents cannot establish this gate or perform rollout.

## RTX summary/tagger co-residency (AI-72)

The optional runner table `co_resident_models` maps engine names to **exact
local model IDs**, forming one residency cohort on one device. Configuration
fails if a member has no resource mapping or members map to different devices.
All unlisted engines and all other models in a listed engine retain exclusive
swap semantics. Omit the table to retain the existing behavior.

The cohort shares the resource queue key and read leases, including cold startup
of the second provider while the first is serving. Revalidation/load still runs
inside each engine's lifecycle lock. On transition into the cohort, peers are
retained only if every loaded model matches their configured cohort model.
A waiting nonmember freezes admission for the whole cohort and waits for all
member responses to drain before eviction. Startup/stop cancellation retains
the existing detached-operation reservation. Each provider remains responsible
for cleaning up its own failed process; the resource gate never assumes that a
shared owner key proves provider health.

The gateway already sends different models concurrently over HTTP. Its capacity
checks now subtract tracked requests for other loaded models; unknown/cold work
still counts against runner capacity. Thus tagger's concurrency does not consume
the one summary slot. Existing loaded-model status and aliases suffice; there
are no new wire fields, and older runners keep their exclusive queue policy.

### Operator rollout and exact changes

Do not deploy from an agent session. After merge:
1. Drain RTX and install the new runner binary first. Its restart requires the
   owner's script. Keep the old configuration initially.
2. Deploy the new backend image.
3. Copy `scripts/simple_ai_summary_provider.py` to
   `/home/lelloman/simple-ai-summary-provider.py` on RTX, then merge the exact
   field changes in `deploy/co-residency/rtx.runner-overlay.toml` into
   `~/config.toml` and restart via the owner's script after draining.
   Preserve the existing summary/tagger model paths and every other
   `engine_resources` entry, including legacy `tag` and `vllm`.
   No backend configuration, Halo update, new dependency or checkpoint is needed.

The overlay sets summary utilization to **0.55**, tagger to **0.30**, and tagger
sequences to **128** while retaining its 4608-token context. Summary context
1024, sequences 512 and the synchronous batched API remain unchanged. The
provider accepts `--gpu-memory-utilization` via its existing configured command;
omitting it preserves 0.92. Fractions must be in (0, 1].

These are conservative starting allocations from the ticket, **not measured
KV-cache sizing**. The combined reservation is 85% of 24 GiB (20.4 GiB), leaving
15% outside the configured vLLM budgets. Real KV capacity, startup in both orders,
preemption, memory headroom and combined throughput must be qualified on RTX;
max sequences is a scheduling ceiling, not a guarantee of 128 full contexts.

Run SanCerchino's live summary phase together with
`~/tagger-ft-4cc1f8a/tools/tagger-ft/bench_gw.py` at 64–128 tagger concurrency.
Record summary >=25 sites/s and tagger >=35 sites/s concurrently, GPU memory
and KV/preemption diagnostics, and no unload journal entries between these
models. Also qualify tagger-first and summary-first cold starts and a subsequent
legacy `class:tag` request that drains/evicts both. These live gates have not
been run in the agent environment.

Rollback after draining: remove `co_resident_models`, restore summary command
without the utilization flag, and restore tagger utilization 0.9 / sequences
256, then use the owner's restart script. The new provider defaults and protocol
remain backward compatible.
