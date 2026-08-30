# Cache-Aware Machine Affinity

## Status and Working Method

This document is the design specification and implementation record. The design was implemented across the shared protocol, gateway, runner, Android client, admin UI, and test suite in August 2026.

Rust workspace verification passes with `cargo test --workspace`. Android protocol and call-site changes are complete; running the Android unit suite additionally requires a configured Android SDK (`ANDROID_HOME` or `android/local.properties`).

Resolved decision records remain in the document to preserve the reasoning and contracts that implementation must follow.

## Summary

Add soft, in-memory machine affinity for chat requests carrying `prompt_cache_key`. Related requests will prefer the runner that handled the key previously, but move when that runner is invalid or estimated to add more than 2 seconds of wait.

This stage improves cache-hit likelihood without claiming a confirmed engine-level hit. Slot-level cache control is deferred. It also removes the current scheduler/router double-selection problem.

## Settled Requirements

### Request Interfaces and Configuration

- Add optional `prompt_cache_key` to Chat Completions and Responses request types. This matches the official Responses API concept documented by [OpenAI](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).
- Normalize by trimming, reject empty values or values over 64 UTF-8 bytes with HTTP 400, and leave requests without a key unchanged.
- Change Android's existing `cloudChat` AIDL signature to accept nullable `promptCacheKey`; pass it in the HTTP body. Make Android service protocol version 2 the only supported version, with no compatibility overload.
- Add routing defaults:
  - `prompt_cache_affinity_enabled = true`
  - `prompt_cache_affinity_ttl_secs = 600`
  - `prompt_cache_affinity_max_entries = 10000`
  - `prompt_cache_affinity_max_extra_wait_ms = 2000`
- Add runner protocol version 2 hooks:
  - Optional `PromptCacheCapabilities` on engine status, with `scope: engine | model_process | slot`, `accepts_cache_key`, and `reports_cached_tokens`.
  - Optional `cached_prompt_tokens` in internal inference metrics.
  - Current engines advertise no capability and report no cached-token count until engine-specific support is implemented.

### Affinity Behavior

- Introduce an `AffinityStore` shared by the router:
  - Key: HMAC-SHA256 of a versioned domain tag plus length-delimited authenticated user ID, requested model selector, and normalized cache key, using a random per-process secret.
  - Value: runner ID, resolved model, and last-used time.
  - Use sliding TTL, lazy expiration, LRU capacity eviction, and compare-and-bind semantics so concurrent first requests converge on one binding.
  - Never persist, log, emit, or forward the caller's raw key. Remove it from audited request bodies and replace it with the scoped digest before proxying to a runner when forwarding is enabled.
- Validate bindings on every lookup. Reuse only when the runner is operational, its circuit is closed, and the resolved model remains loaded and valid for the requested class.
- Invalidate on disconnect, model unload, circuit failure, expiry, capacity eviction, or failed dispatch.
- Preserve the bound resolved model for class requests. Different users, cache keys, or model selectors receive independent bindings.
- Apply soft affinity after normal eligibility and model-loaded filtering:
  - Compute the normal routing winner using existing machine, queue, latency, and scarcity logic.
  - Estimate wait as `floor(active_requests / engine_batch_size) × average_request_latency`, using 1000 ms when no history exists.
  - Reuse the affinity runner when its additional estimated wait versus the normal winner is at most the configured budget, inclusive.
  - Otherwise spill only that request to the normal winner and keep the existing affinity binding. Temporary overload never moves the binding; a later request returns when the affine runner has capacity again.

### Routing and Dispatch

- Refactor dispatch around one routing authority:
  - Non-batched streaming and non-streaming requests execute the exact `RoutePlan` prepared by the scheduler instead of selecting again.
  - Batch preparation resolves the model and ensures capacity without fixing a final runner.
  - Queued requests carry resolved model, class hint, and affinity context.
  - At drain time, the batch dispatcher asks the shared router for the final affinity-aware plan, reserves active capacity before selecting the next item, and uses router-owned proxy/circuit-breaker accounting.
  - Preserve stream-lifetime request accounting and release all reservations on success, failure, cancellation, or dropped receivers.

### Observability and Privacy

- Add sanitized admin routing events for `new`, `reuse`, `spillover_overloaded`, and `rebind_invalid`.
- Add admin snapshot counters and a compact affinity summary in the routing panel.
- Add Prometheus metrics:
  - `simpleai_cache_affinity_decisions_total{outcome=...}`
  - `simpleai_cache_affinity_evictions_total{reason=...}`
  - `simpleai_cache_affinity_bindings`
- Use only fixed labels; never attach users, models, runners, raw keys, or scoped digests as metric labels.
- Raw keys must not appear in audit records, tracing, error messages, admin events, metrics, or runner-bound request bodies.

## Design Decisions (Resolved)

### D1. Identity and Affinity Context Plumbing

- [x] Use persisted `user.id` as the digest identity. Both OIDC and API-key authentication resolve to this canonical SimpleAI user ID before affinity derivation. Do not use email, API-key ID, bearer token, or client IP.
- [x] Pass only an opaque affinity key below the route layer:

  ```rust
  #[derive(Clone, Debug)]
  struct AffinityContext {
      key: AffinityKey,
  }

  #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
  struct AffinityKey([u8; 32]);
  ```

- [x] Keep the requested selector outside `AffinityContext` as existing routing data. Queued requests group it with the resolved routing state:

  ```rust
  struct QueuedRouteContext {
      requested_selector: String,
      resolved_model: String,
      class_hint: Option<ModelClass>,
      affinity: Option<AffinityContext>,
  }
  ```

- [x] Authenticate, resolve and authorize the effective requested selector, then take, trim, validate, and hash the raw key in the HTTP route before audit serialization. Put shared parsing and derivation in the affinity module, for example `ValidatedPromptCacheKey::parse(...)` and `AffinityContext::derive(user_id, requested_selector, validated_key)`. Map parse failures to HTTP 400.
- [x] Discard the raw value at the route boundary. Only `Option<AffinityContext>` may travel through the scheduler, router, queue, and dispatcher. D2 separately defines the sanitized audit/request representation and any scoped value forwarded to capable engines.
- [x] Extend both scheduler chat entry points and router planning with affinity explicitly:

  ```rust
  RequestScheduler::chat_completion(
      request_id,
      model,
      model_request,
      affinity: Option<&AffinityContext>,
      request,
      use_batching,
  )

  RequestScheduler::chat_completion_stream(
      request_id,
      model,
      model_request,
      affinity: Option<&AffinityContext>,
      request,
  )

  InferenceRouter::plan_request(
      model_request,
      affinity: Option<&AffinityContext>,
  )
  ```

  Exact Rust argument order may follow local style, but the affinity input and single-selection invariant are required.
- [x] Keyed request flow: deserialize → authenticate to `user.id` → resolve and authorize selector → consume/validate/hash raw key → audit sanitized request → scheduler → router or queue → dispatcher → proxy. Unkeyed requests carry `None` through the same path.

Decision: resolved. The identity, context types, route boundary, processing order, and carrying interfaces are fixed above.

### D2. Request Sanitization and Runner Forwarding

- [x] Add `prompt_cache_key` to the public request with `#[serde(default, skip_serializing_if = "Option::is_none")]`. Do not introduce a parallel internal request type. In the authenticated route, consume the field with `take()` and reuse the now-sanitized request for auditing and internal dispatch.
- [x] Preserve the Responses key during Responses-to-chat conversion, then consume it from the resulting chat request through the same shared helper as Chat Completions.
- [x] Omit `prompt_cache_key` entirely from audit JSON after extraction. Do not store a redaction marker or scoped value; keyed and unkeyed audited request shapes remain identical.
- [x] Use fail-closed runner forwarding:

  | Request/capability state | Runner-bound `prompt_cache_key` |
  |---|---|
  | Keyed and `accepts_cache_key = true` | Inject the 64-character lowercase hexadecimal scoped HMAC |
  | Keyed and `accepts_cache_key = false` | Omit |
  | Keyed and capability absent | Omit |
  | Unkeyed | Omit |

  Current engines advertise no support, so this phase provides gateway routing affinity without forwarding a key to them.
- [x] In direct Ollama mode, accept and validate the public field for API compatibility, remove it before audit/proxy, perform no affinity operation, and forward no raw or scoped value. Direct-engine cache-key integration requires a later explicit capability contract.
- [x] Derive the affinity key with HMAC-SHA256 using a cryptographically random per-process secret created at gateway startup. Affinity state already disappears on restart, so cross-restart digest stability is neither required nor desired. Add the `hmac` crate to the backend; existing `sha2`, `hex`, and `rand` dependencies supply the remaining primitives.
- [x] Use the following unambiguous, domain-separated HMAC message:

  ```text
  UTF-8("simpleai-prompt-cache-affinity-v1")
  || u32_be(UTF-8 byte length of user_id) || UTF-8(user_id)
  || u32_be(UTF-8 byte length of requested_selector) || UTF-8(requested_selector)
  || u32_be(UTF-8 byte length of normalized_key) || UTF-8(normalized_key)
  ```

  Reject any component whose byte length cannot fit in `u32`, though the public key already has a 64-byte limit. Store the 32-byte output as `AffinityKey`; expose lowercase hex only to the capable-runner request builder. Never log either representation.
- [x] Add privacy tests using a distinctive raw value. Assert that neither the raw key nor scoped hexadecimal value appears in audit JSON, errors, traces/admin events, metrics, incapable-runner requests, or direct Ollama requests. In a capable-runner test only, assert the raw key is absent and the expected scoped value appears in the captured outbound body.

Decision: resolved. Audit JSON, internal context, digest derivation, and runner/direct-engine behavior are fixed above.

### D3. Unified Routing and Execution APIs

- [x] Represent original routing intent as one coherent value:

  ```rust
  #[derive(Clone, Debug)]
  struct RouteRequest {
      requested_selector: String,
      model_request: ModelRequest,
      affinity: Option<AffinityContext>,
  }
  ```

  `requested_selector` is the effective public selector after applying any default, `model_request` is its parsed form, and any affinity key was derived using exactly that selector.
- [x] Represent the final runner decision as an authoritative, immutable plan:

  ```rust
  #[derive(Clone, Debug)]
  struct RoutePlan {
      request: RouteRequest,
      runner: ConnectedRunner,
      resolved_model: String,
      requires_model_load: bool,
      affinity_decision: AffinityDecision,
  }

  enum AffinityDecision {
      Unkeyed,
      Disabled,
      New,
      Reuse,
      RebindOverloaded,
      RebindInvalid,
  }
  ```

  D5 may add an expected binding/version token for conditional store mutation. Capacity reservation state is deliberately not part of `RoutePlan`; D4 defines a separate lifecycle guard.
- [x] Use `QueuedRouteContext` from D1 for runner-independent queued state. Batch preparation may choose a provisional model-preparation target to resolve a class and ensure at least one loaded copy, but it must not store that runner as the dispatch target or create/rebind affinity. Drain-time planning makes the final runner decision from current state.
- [x] Implement the following conceptual router API; exact argument borrowing may follow local Rust style:

  ```rust
  plan_request(&RouteRequest) -> Result<RoutePlan, RouterError>
  prepare_queued_route(&RouteRequest) -> Result<QueuedRouteContext, RouterError>
  plan_queued_request(&QueuedRouteContext) -> Result<RoutePlan, RouterError>
  execute_chat_plan(RoutePlan, &ChatCompletionRequest)
      -> Result<RoutedResponse<ChatCompletionResponse>, RouterError>
  execute_chat_stream_plan(RoutePlan, &ChatCompletionRequest)
      -> Result<RoutedResponse<reqwest::Response>, RouterError>
  ```

  The non-streaming execution path is shared by immediate and batched requests.
- [x] Non-batched flow: build `RouteRequest` → `plan_request` → scheduler prepares the selected model if required → validate/recover → execute that exact plan. Streaming uses the same flow with stream execution.
- [x] Batched flow: build `RouteRequest` → `prepare_queued_route` → scheduler ensures resolved-model capacity → enqueue context and sanitized request → dispatcher calls `plan_queued_request` → validate/reserve → execute that exact plan.
- [x] Replace chat `chat_completion` with `execute_chat_plan`, replace `chat_completion_raw` with `execute_chat_stream_plan`, and remove `chat_completion_batched`. Remove the batch dispatcher's `select_runner` and `send_request_with_client`; the router owns chat proxying and circuit-breaker accounting. Existing non-chat endpoint routing helpers remain in scope unchanged.
- [x] Validate a final plan immediately before reservation/proxy: runner operational, circuit available, and resolved model still eligible and loaded. On failure, replan once from the original routing context before any request bytes are sent. Never retry after the request may have reached a runner, to avoid duplicate completions.
- [x] A non-batched replacement plan may go through scheduler model preparation. A queued replacement may use another currently loaded eligible runner; if none remains loaded, fail cleanly instead of entering a dequeue/load/requeue loop.
- [x] Final selection invariant: every request has exactly one current final dispatch plan, and proxying uses its runner without selecting again. Class resolution or cold-model preparation may use a provisional target. A stale final plan may be replaced at most once before dispatch.

Decision: resolved. Each request mode now has one final selection authority and one router-owned chat proxy/accounting path.

### D4. Capacity Reservation Ownership

- [x] Define `active_requests` as work reserved for imminent or active HTTP inference. Queue waiting, routing, Wake-on-LAN, and model loading do not count as active work.
- [x] Reserve capacity only after final plan validation and immediately before dispatch. In a drained batch, reserve each selected runner before selecting the next item so subsequent selection sees the updated load.
- [x] Use a non-cloneable RAII `CapacityReservation` over the selected runner's `Arc<AtomicUsize>`, carried with the plan as `ReservedRoute`. Creation increments exactly once and `Drop` decrements exactly once without async registry access. Detect and report underflow instead of wrapping the counter.
- [x] Only the router/registry reservation API may create chat reservations. The scheduler and batch dispatcher do not increment or decrement chat counters directly. Legacy non-chat paths may retain their existing accounting until separately refactored.
- [x] Non-streaming execution owns the reservation until the complete response or error. Dropping or aborting the execution future releases it automatically.
- [x] Streaming execution transfers the reservation with the runner response into the gateway response-body state. It releases on EOF, stream error, client body drop, or task abort. Remove the existing manual chat-stream decrement.
- [x] A batch caller cancelled before drain consumes no reservation. If the caller disappears after dispatch while the runner is still working, the execution task retains the reservation until runner work completes or its transport is aborted; a closed response channel does not release capacity early.
- [x] Planning failure, model-load failure, and stale-plan rejection occur before reservation. Serialization failure, connection failure, runner error, decoding failure, normal completion, cancellation, body drop, task abort, and unwinding after reservation all release through the same guard.
- [x] Tests must prove counts return to zero for every termination path, a reservation cannot be cloned or released twice, release still works after disconnect/reconnect, underflow is detected, and sequential batch selection observes each prior reservation.

Decision: resolved. A machine is counted busy only while work is reserved or running, and automatic ownership-based cleanup prevents leaked or double-released capacity.

### D5. Affinity Store Atomic Semantics

- [x] Create the first binding when the first request reaches final validation/reservation immediately before dispatch. Planning alone does not bind, so an abandoned or cancelled plan leaves no affinity state; waiting until completion is too late for concurrent requests.
- [x] Use first-dispatcher-wins behavior. A plan records the binding revision it observed. At reservation time, it may commit only if that observation is still current. A concurrent loser treats its plan as stale and uses the single pre-dispatch replan allowed by D3.
- [x] Route requests to the valid affine runner while its additional estimated wait remains within the configured budget. When that budget is exceeded, spill only the current request to the normal winner and retain the original binding. Temporary overload never creates, replaces, or refreshes a binding for the spillover runner.
- [x] Replace a binding only when its runner/model becomes invalid, not when it is merely busy. Conditional replacement must match the observed revision so an old invalidation decision cannot overwrite newer state.
- [x] A failed dispatch conditionally removes only the exact binding/revision used by that dispatch. It cannot erase a binding created or replaced by a newer request.
- [x] Refresh sliding TTL and LRU position only when a still-current, valid binding is actually selected for reuse. Merely inspecting an invalid binding or spilling elsewhere does not refresh it.
- [x] Use a monotonically increasing binding revision/token for conditional commit, touch, replace, and invalidate operations.
- [x] Inject or abstract a monotonic clock so expiry tests advance time without sleeping.
- [x] Expire entries lazily when encountered and evict the least recently used entry before inserting beyond capacity. Configuration validity, runtime reload scope, and zero TTL/capacity behavior are decided by D10; the store never operates with invalid values.

A candidate atomic API to accept or revise:

```rust
lookup(key, now) -> Option<Binding>
bind_if_absent(key, candidate, now) -> Binding
replace_if(key, expected, replacement, now) -> bool
invalidate_if(key, expected) -> bool
invalidate_runner(runner_id) -> usize
```

The implementation may revise these method names, but it must preserve revision-checked commit, touch, invalid replacement, and invalidation.

Decision: resolved. The first request reaching dispatch binds the key; valid affinity remains sticky through temporary overload, which causes one-request spillover rather than rebinding.

### D6. Eligibility and Wait Estimation Inputs

- [x] Define virtual queue wait as `floor(active_requests / engine_batch_size) × average_request_latency`. Each complete group of active requests represents one batch ahead of the new request.
- [x] Compare the affine runner with the normal routing winner using `additional_wait = max(0, affinity_wait - normal_wait)`. Reuse affinity when additional wait is less than or equal to the configured allowance; above it, spill only the current request as defined in D5.
- [x] Use the audit-derived historical average for the runner and resolved model's effective configured class. This applies equally to explicit model requests that belong to a configured class. When no matching history exists or the model has no configured class, use the existing 1000 ms default.
- [x] Determine batch size from the engine on the selected runner that reports the resolved model, accounting for canonical/local aliases. Prefer an engine reporting the model as loaded. If multiple matching engines remain ambiguous, use the smallest reported size conservatively and emit a warning; never use a fleet-wide or unrelated-engine maximum.
- [x] Treat a reported batch size of zero as one for routing safety and emit a warning. Runner-protocol validation may additionally prevent zero at its source.
- [x] Snapshot the normal and affine runners' active atomic counts immediately before comparison and before reserving the new request. The estimate excludes the not-yet-created reservation. A global lock is not required; routing remains robust to normal concurrent changes.
- [x] Calculate in milliseconds with saturating integer arithmetic. Floor division applies only to `active_requests / engine_batch_size`; use `max(0, ...)` semantics for subtraction and compare the configured threshold inclusively.
- [x] Exclude model-load time: an affinity binding is reusable only when its resolved model is currently loaded. Cold-model preparation occurs outside the wait comparison.
- [x] For a class selector, re-run current class classification for the bound resolved model. If it no longer belongs to the requested class, treat the binding as invalid and rebind under D5/D7 rather than treating the condition as overload spillover.

Decision: resolved. A runner is virtually full when its batch-wave estimate exceeds the normal winner's estimate by more than the configured extra-wait allowance.

### D7. Lifecycle and Failure Invalidation

- [x] Combine eager cleanup for known lifecycle changes with validation on every binding lookup as a safety net for delayed, lagged, or missed events.
- [x] Give each runner connection an internal generation/session identifier. Store both runner ID and connection generation in each binding. A restarted or reconnected runner using the same public ID cannot inherit affinity for cache state from its previous process/connection.
- [x] Subscribe one gateway lifecycle task to registry events. Explicit disconnect and runner timeout remove all bindings for that runner generation; unhealthy status does the same; model unload removes only bindings for models no longer loaded on that runner generation.
- [x] Add a circuit-breaker open event and subscribe the same lifecycle task (or a clearly owned companion task). Circuit opening removes bindings for that runner generation. Lookup-time circuit validation remains authoritative if event delivery lags.
- [x] On every lookup, verify the same runner generation is connected and operational, its circuit is available, the bound model is loaded, and any requested class still contains that model. Failure conditionally removes the observed binding and produces a new binding through normal planning.
- [x] A chat dispatch failure immediately performs D5's revision-checked conditional invalidation. It cannot remove a binding that a newer request has already changed.
- [x] With the configured maximum of 10,000 entries, scan the store for runner/model lifecycle cleanup. Do not add a reverse index unless profiling later demonstrates a need; avoiding duplicated index state is preferred at this scale.
- [x] Use these fixed removal reasons: `expired`, `capacity`, `runner_disconnected`, `runner_unhealthy`, `model_unloaded`, `circuit_open`, `dispatch_failed`, and `binding_invalid`. Runner timeout uses `runner_disconnected`; class mismatch and generation mismatch use `binding_invalid`.

Decision: resolved. Known failures clean up eagerly, every reuse is independently validated, and connection generations prevent cache ownership from surviving a runner restart.

### D8. Metrics and Admin UI Infrastructure

- [x] Use a dedicated Prometheus registry owned by application state rather than global static counters. Add a suitable Rust Prometheus client dependency during implementation. Each application/test state receives a fresh registry so tests do not share counters.
- [x] Expose Prometheus/OpenMetrics output at `GET /admin/metrics` under the existing admin-role authentication middleware. A scraper may use an admin API key. Do not add a public `/metrics` endpoint.
- [x] Count keyed-request decisions once for the final dispatched outcome using only `new`, `reuse`, `spillover_overloaded`, `rebind_invalid`, and `disabled`. Do not count unkeyed requests, discarded stale plans, users, selectors, models, runners, keys, or digests.
- [x] Count removals using only D7's fixed reasons: `expired`, `capacity`, `runner_disconnected`, `runner_unhealthy`, `model_unloaded`, `circuit_open`, `dispatch_failed`, and `binding_invalid`.
- [x] Keep decision and removal counters as process-lifetime totals and expose current active bindings as a gauge. They reset naturally with the process-local store on gateway restart.
- [x] Extend `RouterStateSnapshot` with this aggregate shape:

  ```json
  {
    "affinity": {
      "enabled": true,
      "bindings": 247,
      "max_entries": 10000,
      "ttl_secs": 600,
      "max_extra_wait_ms": 2000,
      "decisions": {
        "new": 120,
        "reuse": 615,
        "spillover_overloaded": 31,
        "rebind_invalid": 4,
        "disabled": 0
      },
      "evictions": {
        "expired": 80,
        "capacity": 2,
        "runner_disconnected": 5,
        "runner_unhealthy": 1,
        "model_unloaded": 3,
        "circuit_open": 1,
        "dispatch_failed": 3,
        "binding_invalid": 1
      }
    }
  }
  ```

- [x] Show a compact routing-panel summary: enabled/disabled, bindings versus capacity, reuse rate, temporary spillovers, invalid rebindings, and total expired/evicted bindings. Never send per-binding identities, selectors, model names, runner IDs, raw keys, or digests to the panel.
- [x] Test admin authentication, metrics content type, fixed label sets, counter/gauge behavior, fresh-registry isolation, snapshot serialization, and absence of raw/scoped keys.

Decision: resolved. Operators receive authenticated aggregate process-level visibility without exposing affinity identities or infrastructure mappings.

### D9. Protocol Versioning and Compatibility

- [x] Make a clean Android version-2 cutover: set `SERVICE_VERSION = 2`, `MIN_PROTOCOL_VERSION = 2`, and `MAX_PROTOCOL_VERSION = 2`; add nullable `promptCacheKey` to `cloudChat`; update every AIDL call site, example, document, and test; retain no protocol-1 overload.
- [x] Separately bump the gateway-to-runner WebSocket `PROTOCOL_VERSION` from 1 to 2. The existing exact-version validation rejects protocol-1 runners, and gateway plus runners must be deployed together.
- [x] Retain optional `PromptCacheCapabilities` because an updated engine may legitimately declare no prompt-cache support. `None` means no declared support; `Some(...)` declares its scope and reporting/forwarding capabilities.
- [x] Retain optional `cached_prompt_tokens` because even an updated engine may not support or produce the measurement for every request.
- [x] Do not implement old-runner compatibility, mixed-fleet behavior, compatibility overloads, or mixed-version tests. Update all fixtures and protocol tests directly to version 2.

Decision: resolved. Android and runner protocols both require version 2; optional fields represent runtime capability, not backward compatibility.

### D10. Rollout, Configuration Validation, and Documentation

- [x] Validate configuration at gateway startup. TTL and maximum entries must be greater than zero; extra wait may be zero, meaning affinity receives no additional-delay allowance. Reject invalid values with a clear startup error rather than silently clamping them. Use checked conversions where platform-sized values are required.
- [x] Treat affinity configuration as startup-only; runtime reload is out of scope. Log enabled state, TTL, maximum entries, and extra-wait allowance at startup without logging secrets or binding data.
- [x] Add all fields and defaults to `RoutingConfig::default`, `backend/config.example.toml`, backend/operator documentation, and relevant deployment examples:

  ```toml
  [routing]
  prompt_cache_affinity_enabled = true
  prompt_cache_affinity_ttl_secs = 600
  prompt_cache_affinity_max_entries = 10000
  prompt_cache_affinity_max_extra_wait_ms = 2000
  ```

- [x] When affinity is disabled, continue accepting and validating public cache keys, route exactly like unkeyed requests, perform no store lookup/mutation, and count the keyed decision as `disabled`. A capable selected engine may still receive the scoped HMAC because this setting disables machine affinity rather than the public cache-key interface.
- [x] Keep the unified routing and reservation/accounting refactor active when affinity is disabled. Restarting with `prompt_cache_affinity_enabled = false` clears process-local bindings and restores ordinary machine selection without a database change.
- [x] Roll out protocol-2 gateway and runners together. Initially set affinity false, verify ordinary routing/accounting, Android protocol 2, and admin metrics, then enable affinity and restart. Observe reuse, spillover, invalidation, and dispatch-failure counters.
- [x] A feature rollback changes the setting to false and restarts the gateway. A binary rollback must roll back gateway and runners together because D9 intentionally retains no runner protocol-1 compatibility.
- [x] Update Android README/examples, runner protocol documentation, backend/operator documentation, startup guidance, and rollback guidance.
- [x] Use these exact final verification commands from the repository root:

  ```bash
  cargo test --workspace
  (cd android && ./gradlew test)
  tests/e2e/run-tests.sh
  ```

  Extend the E2E environment with the two-runner affinity cases in this plan before treating the final command as complete coverage.

Decision: resolved. Configuration fails clearly when invalid, is startup-only, and supports a documented disable/restart rollback without undoing the routing safety refactor.

## Implementation Phases

### Phase 1: Request Schema, Validation, and Protocols

Blocked by: D1, D2, D9.

Work:

- Add request fields and shared normalization/validation.
- Preserve the field through Responses-to-chat conversion without auditing the raw value.
- Add runner capability and cached-token protocol hooks.
- Update Android AIDL, service, cloud client, protocol constants, and tests.

Acceptance:

- Both endpoints accept valid keys and reject empty/oversized keys with HTTP 400.
- Omitted keys serialize and behave exactly as before.
- No raw key appears in audit output or runner-bound requests.
- Android protocol 2 handles nullable keys and rejects protocol 1.

### Phase 2: Affinity Store

Blocked by: D5 and the digest decision in D2.

Work:

- Add the store, clock abstraction, binding/key types, configuration, and store-level metrics hooks.
- Implement sliding TTL, LRU capacity, compare-and-bind, conditional replace, and conditional invalidation.

Acceptance:

- Deterministic unit tests cover user/selector/key scoping, expiry, refresh, eviction, concurrent first binds, stale replacement, and stale invalidation.

### Phase 3: Single Routing Authority and Reservations

Blocked by: D3, D4.

Work:

- Separate planning from execution.
- Make prepared plans authoritative for streaming and non-streaming requests.
- Centralize proxy, circuit-breaker, and active-capacity accounting in the router.
- Preserve stream-lifetime accounting using the selected reservation design.

Acceptance:

- Tests prove the planned runner is the dispatched runner.
- Every success, error, cancellation, body-drop, and channel-drop path returns active counts to zero.
- Unkeyed selection results remain compatible with existing smart routing.

### Phase 4: Non-Batched Affinity Routing

Blocked by: D1, D5, D6, D7 and Phases 2-3.

Work:

- Add affinity-aware planning after normal eligibility/model-loaded filtering.
- Implement valid reuse, invalid rebind, overload spillover without rebinding, class-model preservation, and dispatch-failure invalidation.

Acceptance:

- Sequential same-key requests reuse one runner.
- Different users, keys, and selectors do not share bindings.
- Class requests retain the valid bound resolved model.
- Additional wait at or below the configured threshold reuses; above it spills to the normal winner without changing the binding.

### Phase 5: Batch Queue and Dispatcher Integration

Blocked by: D3, D4, D6 and Phase 4.

Work:

- Extend `QueuedRequest` with resolved model, class hint, and affinity context.
- Move final selection and reservation to drain time.
- Replace dispatcher-owned selection, HTTP proxying, and accounting with router-owned operations.
- Preserve balanced distribution for unkeyed requests.

Acceptance:

- Keyed queued requests follow the same selector as non-batched requests.
- Unkeyed queued requests remain balanced.
- Cancellation before and after dispatch does not consume permanent capacity.
- Stale queued context safely replans or fails according to D3.

### Phase 6: Lifecycle Invalidation

Blocked by: D7 and Phases 2, 4-5.

Work:

- Wire registry, status, circuit, and dispatch-failure events to conditional store invalidation.
- Add fixed eviction/invalidation outcomes.

Acceptance:

- Disconnect, runner timeout, unload, invalid class mapping, circuit opening, and failed dispatch cannot leave a reusable stale binding.
- A stale failure cannot erase a newer binding.

### Phase 7: Observability and Admin UI

Blocked by: D8 and stable outcomes from D5/D7.

Work:

- Add decision and eviction metrics, binding gauge, authenticated `/admin/metrics`, admin snapshot fields, routing events, and UI summary.
- Audit all serialization/logging paths for raw keys and digests.

Acceptance:

- Metrics expose only fixed labels and correct values.
- Admin state reports aggregate affinity health without binding identities.
- Automated tests prove a distinctive raw test key never appears in captured audit, trace/event, metrics, or outbound payloads.

### Phase 8: E2E, Documentation, and Rollout

Blocked by: D10 and all earlier phases.

Work:

- Add two-runner E2E coverage for reuse, invalidation, overload spillover, and return to the original runner when capacity recovers.
- Update example configuration and operator documentation.
- Verify enabled, disabled, keyed, and unkeyed modes.

Acceptance:

- `cargo test --workspace` passes.
- `(cd android && ./gradlew test)` passes from the repository root.
- `tests/e2e/run-tests.sh` passes after adding this plan's affinity scenarios.
- Disabling affinity preserves the unified routing path and existing routing behavior.

## File Impact Map

This map is expected to change as decisions are resolved, but each listed area must be reviewed.

| Area | Primary files | Expected change |
|---|---|---|
| Public request schemas | `simple-ai-common/src/chat.rs`, `simple-ai-common/src/responses.rs` | Add key and cached-token fields; preserve Responses conversion |
| Runner protocol | `simple-ai-common/src/protocol.rs`, `inference-runner/src/gateway/status.rs` | Bump to version 2 and add optional runtime capabilities |
| HTTP routes and audit | `backend/src/routes/chat.rs`, `backend/src/routes/responses.rs`, `backend/src/routes/auth_helpers.rs` | Validate, derive identity, sanitize audit, build affinity context |
| Configuration | `backend/src/config.rs`, `backend/config.example.toml` | Add defaults and validation |
| Affinity state | new `backend/src/gateway/affinity.rs` or decision-selected equivalent | Store, clock, atomic operations, tests |
| Planning and execution | `backend/src/gateway/router.rs`, `backend/src/gateway/scheduler.rs` | One selection authority, wait comparison, prepared execution |
| Capacity accounting | `backend/src/gateway/registry.rs`, router/stream wrappers | Reservation ownership and exact release |
| Batch path | `backend/src/gateway/batch_queue.rs`, `backend/src/gateway/batch_dispatcher.rs` | Queue context and router-owned drain-time dispatch |
| Lifecycle | `backend/src/gateway/registry.rs`, `backend/src/gateway/ws.rs`, `backend/src/circuit_breaker.rs` | Invalidation producers and subscriptions |
| Application wiring | `backend/src/main.rs`, `backend/src/lib.rs`, `backend/src/gateway/mod.rs` | Construct/share store, metrics, lifecycle task |
| Admin and metrics | `backend/src/gateway/telemetry.rs`, `backend/src/routes/admin.rs`, `backend/static/admin.html` | Aggregate snapshot/events, metrics endpoint and UI |
| Android | AIDL, `SimpleAIService.kt`, `CloudLLMClient.kt`, Gradle constants and tests | Nullable argument, protocol 2, request serialization |
| Integration tests | backend tests, `tests/e2e` | Two-runner reuse, overload, failure and privacy coverage |

## Test Matrix

- Schema and route tests for both endpoints: key accepted, Responses-to-chat conversion preserves it, omitted keys retain existing behavior, and invalid keys return 400.
- Affinity-store tests for user/model scoping, sliding TTL, LRU eviction, concurrent binding, conditional replacement/invalidation, and deterministic time.
- Privacy tests covering audit records, tracing/admin events, metrics, errors, and outbound runner/direct-engine bodies.
- Router tests covering:
  - Sequential same-key requests reuse one runner.
  - Class requests retain the same resolved model.
  - Different users or selectors do not share affinity.
  - Disconnect, unload, circuit opening, and dispatch failure invalidate affinity.
  - Additional wait at or below 2000 ms reuses; above it spills without rebinding, and later capacity recovery returns to the affine runner.
  - Disabled affinity and unkeyed requests preserve existing smart-routing behavior.
- Batch tests proving keyed requests use the same selector, unkeyed distribution remains balanced, stale plans recover as specified, and active counts return to zero on every termination path.
- Streaming and non-streaming tests proving the prepared runner is the dispatched runner and reservations live exactly as long as execution/streaming.
- Android tests verifying protocol 2, nullable key handling, all call-site changes, and request-body serialization.
- Two-runner E2E coverage for sequential reuse, invalidation, overload spillover, and post-overload return to the affine runner.

## Assumptions and Deferred Work

- Affinity state is intentionally process-local and disappears on gateway restart; no database migration is needed.
- Only `/v1/chat/completions` and `/v1/responses` participate. Other inference endpoints remain unchanged.
- "Affinity reuse" means the likely cache-owning machine was selected, not that the engine confirmed a KV-cache hit.
- A later phase will use the new capability and metrics hooks to implement llama.cpp/Ollama-specific slot selection and confirmed cached-token reporting.
- Distributed affinity across multiple gateway processes is out of scope.
