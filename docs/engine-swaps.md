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
