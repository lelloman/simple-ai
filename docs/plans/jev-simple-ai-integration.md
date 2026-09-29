# JEV-9B integration into Simple-AI

Status: implemented and deployed, 2026-09-29. Live gateway decision, chat switching,
and powered-off-host wake checks passed; see [validation evidence](../evals/jev-rtx3090/integration.json).
Scope includes all three decision types. See [API usage](../decisions.md).
## Intended outcome

A client calls the ordinary authenticated Simple-AI URL with
`model: "class:semantic_decisions"`. Simple-AI locates or wakes the RTX runner,
waits for active GPU work, loads JEV, evaluates the request, and returns typed
answers. A subsequent chat request switches the GPU back to chat automatically.
Users do not SSH, stop the runner, start Docker, or manage keepalives.

Definition of done: this complete flow works through the deployed gateway,
including chat → JEV → chat, while `simple-ai-runner` stays running throughout.
Installing the model or demonstrating the loopback provider alone is insufficient.

## Existing implementation and reusable pieces

- `scripts/simple_ai_semantic.py`: pinned JEV tokenizer, trained decision adapter,
  bare-v1 prompt, head bias/temperature, bounded branch execution, input validation.
- `deploy/semantic-rtx3090/`: validated BF16 vLLM profile and the version-checked
  LM-head LoRA registration fix. Existing JEV volume is reused; no duplicate weights.
- `inference-runner/src/engine/registry.rs`: model leases and exclusive resource
  groups. Engines sharing `cuda:0` cannot safely run conflicting models together.
- `inference-runner/src/engine/vllm.rs`: Compose lifecycle and adoption precedent.
  It currently assumes one chat-oriented base URL/Compose directory, so adding
  JEV as a normal chat model there would conflate two different services.
- `inference-runner/src/engine/extraction.rs`: managed capability and protection
  of in-flight provider requests, including client cancellation.
- `backend/src/routes/extractions.rs`, gateway scheduler/router and `wol.rs`:
  authentication, class permissions, audit events, model preparation and wake flow.
- `docs/evals/jev-rtx3090/`: real GPU/HTTP evidence; 24/24 labelled synthetic
  examples, successful 16K inputs, approximately 22 GiB runtime VRAM. Initial
  500-token latency is 259 ms versus the retired Qwen scorer's 154 ms.

The existing NLI `/v1/classifications` contract returns entailment/neutral/
contradiction per hypothesis. JEV does not implement that contract. Retain it
and add a separate first-class decision capability.

## Public API

Add `POST /v1/decisions` on gateway and runner. Public class:
`class:semantic_decisions`; concrete model: `autotrust/JEV-9B`.
Use ordinary API-key/OIDC/LAN authentication policy and existing model-class
permissions. A concrete model ID requires the existing `model:specific` role.

One request contains one shared state and an ordered list of questions:

```json
{
  "model": "class:semantic_decisions",
  "state": {
    "message": "I was charged twice. Please refund the duplicate.",
    "policy": "Duplicate charges are refundable."
  },
  "questions": [
    {
      "id": "refund_requested",
      "type": "boolean",
      "instruction": "Does the customer request a refund?"
    },
    {
      "id": "department",
      "type": "choice",
      "instruction": "Which department should handle the request?",
      "options": [
        {"id": "billing", "description": "Payments and refunds"},
        {"id": "support", "description": "Technical problems"},
        {"id": "unknown", "description": "Insufficient information"}
      ]
    },
    {
      "id": "urgency",
      "type": "rating",
      "instruction": "How urgent is the customer's request?",
      "levels": [
        "No action required", "Routine", "Needs attention soon",
        "Urgent", "Critical", "Immediate emergency"
      ]
    }
  ]
}
```

Contract decisions:

- `state`: nonempty text, object, or array; preserve structured data, reject
  unsupported scalars and nonfinite numbers. No remote URLs or document fetching.
- `questions`: 1–128, unique IDs; results preserve order and IDs. IDs are bookkeeping,
  not instructions. IDs: 1–64 ASCII letters/digits/underscore/hyphen.
- Boolean maps to JEV's `noul`, trained option order `false`, `true`.
- Choice accepts 2–16 unique option IDs, preserves caller order, and evaluates
  descriptions. Return the stable ID, not a generated or reformatted label.
- Rating maps to JEV's `score`: exactly six described levels corresponding to
  integers 0–5. Deterministically append the numbered rubric to the instruction;
  keep the model's trained options `0` through `5`. Test this rendering on the
  real model. No arbitrary-length scales or promise of full TypeSafe compatibility.
- Use tagged request/response enums and shared Rust validation. Reject unknown
  fields and combinations invalid for their question type at gateway and runner.
- Bound JSON body to 2 MiB, each instruction/option/level description to 8,192
  characters, and total submitted branch tokens to 4,194,304. Tokenizer checks
  each complete prompt against 18,432 tokens including the readout; never truncate.
  Validate every branch before running any branch. Gateway handles structural
  validation before scheduling; exact token limits require runner tokenization.

Response envelope: request ID, concrete resolved model, pinned revision, ordered
`answers`, usage, timing, `calibrated: false`, and
`upstream_temperature_applied: true`.

Answer shapes:

- Boolean: `id`, `type`, `value` (boolean), `probabilities` with `false`/`true` keys.
- Choice: `id`, `type`, `selected` (option ID), `probabilities` keyed by option ID.
- Rating: `id`, `type`, `value = sum(i * p_i)` in [0,5], six probabilities and
  the level legend. `value` is an expected rating, not probability of correctness.

Deterministic ties choose the first trained/caller option. Do not add a loosely
specified `confidence` field or advertise application-level calibration. Usage
separates question count, submitted prompt tokens, readout tokens and observed
cached tokens (null if unavailable). Timing distinguishes scheduling/loading from
inference where measured. No raw logits or internal backend details by default.

A failed branch fails the request; do not return a success envelope with silently
missing answers. Caller IDs identify results, not an idempotency promise.
Existing standalone `/v1/semantic/*` paths can remain internal compatibility
wrappers over the same implementation; the documented gateway API is `/v1/decisions`.

## Managed runner engine and GPU ownership

Create `DecisionEngine` (`engine_type = "decisions"`) implementing the normal
`InferenceEngine` lifecycle and a new `decide` trait method. New configuration
`[engines.decisions]` defaults to disabled. It owns the JEV Compose stack as
one unit: vLLM plus Python provider. No shell commands supplied by API callers.

Configure `[engine_resources] decisions = "cuda:0"` alongside vLLM chat,
llama.cpp and other existing CUDA engines. Keep `batch_size() = 1` initially:
that means admitted HTTP evaluations, not the provider's two internal branches.
Do not advertise 128-request capacity just because a request accepts 128 questions.

Lifecycle requirements:

1. Availability advertises the prepared JEV model while unloaded. Preflight checks
   require the pinned image, complete checkpoint, tokenizer and head metadata;
   missing artifacts yield a clear preparation error, not request-time downloads.
2. Loading is serialized/idempotent. Acquire the existing resource lease, let the
   registry drain/unload conflicting engines, then start both JEV services.
3. Readiness requires healthy vLLM, the exact model/adapter revision, and a healthy
   provider reporting the expected model/revision/protocol. HTTP listener readiness
   alone is insufficient. Verify the adapter is selected for every decision.
4. Restart adoption reconciles both containers and their identities. Adopt a valid
   running stack; repair or stop a partial stack. Do not report stale loaded state
   after a crash, or mistake the existing chat vLLM service for JEV.
5. Unload blocks until active provider work finishes; stop the provider then vLLM,
   verify GPU-owning processes exited, and only then release ownership. Keep the
   checkpoint volume. Startup failure/cancellation must clean up partial containers.
6. The admitted request runs in an owned task retaining the **ModelLease** and the
   engine in-flight guard until actual inference has ended, even if HTTP clients
   disconnect. Dropping an outer handler must not make the GPU appear idle.
7. On inference timeout, stop/drain the JEV stack before dropping its lease if
   backend cancellation cannot be positively confirmed. On failure to stop, keep
   the resource unavailable and report the error; never start a second GPU model.
8. Fix/test stale-owner recovery where needed: the registry's shared-owner fast path
   skips `load_model`, so a crashed JEV backend must be revalidated/recovered under
   its retained ownership. Failed health checks must not imply a free GPU.

Use a bounded runner admission queue (32 waiters, 120-second admission deadline)
and one active evaluation initially. Acquire an admission permit before holding
an inference GPU lease, so queued JEV work does not unnecessarily prevent chat
handoff. The existing gateway chat batch queue is not generic decision admission;
do not reuse it without adapting its typed request/cancellation handling.

Keep automatic model switching demand-driven. After JEV finishes, it may remain
loaded under existing idle policy; the next chat request unloads it and loads chat.
No automatic chat reload after every decision, no manual runner shutdown, and no
permanent keepalive or always-on GPU reservation.

Suggested defaults: startup 600 s, inference 180 s, shutdown/drain 60 s. Align
scheduler `model_prepare_timeout_secs` with readiness plus drain overhead. Most
cold loading currently occurs in scheduler `prepare_for_request`, before HTTP
proxying; do not assume raising the proxy timeout alone solves cold starts.
The current generic proxy client has a 300 s timeout. Support a decision-specific
budget up to 960 s to cover a raced reload plus admission/inference; preserve
other routes' timeout behavior. Document the total client budget including the
configured wake timeout and scheduler preparation, and return 504 with the failed
stage rather than an unexplained disconnect.

## Gateway, discovery and observability

- Add `Capability::SemanticDecisions` and `ModelClass::SemanticDecisions`, including
  enum iteration/display/parser coverage and model config classification.
- Add `[models].semantic_decisions = ["autotrust/JEV-9B"]` and RTX-specific class
  preferences/speculative wake targets matching the runner's actual ID/tags.
  Initial rollout is RTX only; do not advertise unqualified Halo support.
- Register gateway/runner routes and `decide` scheduler/router methods. Enforce
  the decision class and supported concrete models before causing a model switch.
- Route to the prepared runner/resolved model. Avoid preparing one runner then
  independently reselecting another in the proxy method. Replan explicitly if
  the selected runner disappears.
- Apply normal request authentication and model permissions; runner communication
  follows the existing trusted gateway transport. Provider/backend remain private
  (loopback or internal Docker network), with no public vLLM admin endpoints.
- Use existing wake/load events and activity leases throughout wake, admission,
  inference and cancellation cleanup; no ad-hoc background keepalive loop.
- Advertise supported types, context, max questions/options, revision, loaded/error
  state and concurrency in capability metadata. Do not advertise JEV as chat just
  because its implementation uses vLLM. Check model-list and admin rendering.
- Audit one request/response with resolved model, revision, runner, status, question
  count, token usage, elapsed time and wake/load information. Use scoped guards for
  active-request accounting so cancellation/errors cannot leak counts. Do not put
  state/question content in ordinary logs.
- Error contract: existing auth statuses; 400 invalid request/model capability;
  413 body too large; 429 admission saturation; 503 unavailable/preparation failure;
  504 timeout; 502 invalid upstream response. Preserve actionable, sanitized errors.

## Implementation sequence

### 1. Contract and provider

Add `simple-ai-common/src/decision.rs` and exports. Implement validation and typed
responses. Extend the Python provider to handle mixed typed questions with one
common execution path and exact JEV slot mappings. Retain the validated checkpoint
and runtime profile; add ordinal rating coverage and internal readiness metadata.

### 2. Runner and lifecycle

Add `inference-runner/src/engine/decisions.rs`, API route, configuration, main
registration and capability discovery. Implement composite lifecycle, restart
adoption, admission and cancellation-safe leases. Make narrowly scoped registry
changes required by failure/recovery tests. Extract a small Compose helper only
if it avoids real duplication; do not refactor chat request normalization.

### 3. Gateway and deployment

Add backend route, scheduler/router, class/config/wake/telemetry wiring and admin
metadata coverage. Extend `scripts/deploy-fleet.sh` to deploy provider plus pinned
runtime files as a coherent version. Update `scripts/configs/rtx.toml` and actual
gateway configuration; other runners retain disabled defaults.

Retain/reuse `jev-rtx3090_models`; the retired Qwen scorer remains removed. Keep
runtime base-image identity and the vLLM 0.27.1 head-registration patch pinned.
Do not rebuild the old model or introduce another checkpoint cache. Normal
runtime disables vLLM development routes; benchmark mode remains explicit.

### 4. Full-path validation and rollout

Run targeted tests below, then workspace checks appropriate to the shared enum
and routing changes. Deploy gateway support **before** a runner advertises the
new capability, since older gateways may reject the new enum. Prepare artifacts,
deploy the RTX runner, and validate through the authenticated external gateway.
Update user documentation with one gateway curl example; operational Docker
instructions belong in maintenance docs only.

Rollback: disable decision advertisement/config on the RTX runner first, stop its
managed stack safely, restore previous runner/config if needed, and only then
roll back gateway capability support. Preserve JEV weights and existing chat config.

## Acceptance tests — integration is not complete without these

1. Shared schema tests: mixed kinds, IDs/order, six-level rating expectation,
   16-option boundary, unknown fields, invalid JSON shapes and size/token limits.
2. Provider tests: exact template, head/bias/temperature, forced adapter selection,
   malformed upstream results, order, failures and cancellation draining.
3. Runner with controllable fake services: first load, repeated load, concurrent
   load coalescing, partial startup failure, crash/reload, runner restart adoption,
   unload failure and health reconciliation; no second GPU owner on any failure.
4. Resource tests: active chat stream blocks JEV load; active/cancelled JEV blocks
   chat until quiescent; queued work does not hold unnecessary leases; timeouts
   release admission and activity accounting without premature GPU release.
5. Gateway integration harness (pattern: `scripts/check-extraction-local.py`):
   auth and permission failures, class versus concrete ID, capability discovery,
   resolved-model attribution, unavailable runner, load deadline and error mapping.
6. Real RTX gateway flow with the runner continuously active:
   chat → mixed boolean/choice/rating decision request → chat. Repeat with JEV
   cold and warm, runner restart, concurrent requests and client disconnect.
7. Sleeping-host request wakes the RTX via existing scheduling and returns an
   answer without user intervention. Activity tracking allows normal idle policy
   to resume after completion.
8. Re-run 24 labelled/four ambiguous examples, add rating rubric examples, and
   exercise 16K input. Record full-path cold-start and warm latency, queue time,
   switching time, throughput and VRAM. Keep the slower initial JEV baseline
   visible; speed optimization is not a substitute for integration correctness.
9. A user with an ordinary class-level API key can use the documented gateway
   request without SSH, a local tunnel, Docker commands or stopping chat manually.

## Scope boundaries

First release: one shared state/request, all three typed decisions, RTX-only,
authenticated synchronous gateway API, managed loading/unloading and existing
observability. No chat generation through JEV, image/audio input, streaming,
new Android Binder API, arbitrary rating scales, TypeSafe SDK compatibility,
quantization work or confidence-based automated actions. These can be evaluated
separately once the gateway path works reliably.
