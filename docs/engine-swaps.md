# Shared-GPU engine swaps (AI-67)

Engines in the same resource group (for example RTX's `cuda:0`) cannot run
simultaneously. Model acquisition queues by **engine and local model**, after
alias resolution. Once the current batch's inference leases have drained, the
oldest waiting request selects the next model. All requests for that model
already waiting at that instant form a finite batch. They keep ownership through
startup and their complete responses, including streaming bodies. The next model
cannot unload them between readiness and inference. Arrivals after the snapshot
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
