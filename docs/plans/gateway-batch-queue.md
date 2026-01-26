# Gateway Buffer Queue with Batching Support

## Overview

Add a buffer queue to the gateway that collects incoming requests and batches them based on runner-reported batch sizes. This improves throughput by allowing runners to process multiple requests in parallel.

## Implementation Phases

### Phase 1: Protocol & Runner Config Changes

**Add `batch_size` to protocol structs** (`simple-ai-common/src/protocol.rs`):

1. Add to `EngineStatus` (line ~154):
   ```rust
   /// Maximum batch size for this engine (default: 1 = no batching)
   #[serde(default = "default_batch_size")]
   pub batch_size: u32,
   ```

2. Add helper function:
   ```rust
   fn default_batch_size() -> u32 { 1 }
   ```

3. Update `RunnerStatus::starting()` if needed

**Add `batch_size` to runner engine configs** (`inference-runner/src/config.rs`):

1. Add to `LlamaCppEngineConfig` (after line ~147):
   ```rust
   /// Maximum batch size for concurrent inference (default: 1)
   #[serde(default = "default_batch_size")]
   pub batch_size: u32,
   ```

2. Add to `OllamaEngineConfig` (after line ~93):
   ```rust
   /// Maximum batch size (default: 1)
   #[serde(default = "default_batch_size")]
   pub batch_size: u32,
   ```

3. Add default function:
   ```rust
   fn default_batch_size() -> u32 { 1 }
   ```

**Update status collection** (`inference-runner/src/gateway/status.rs`):
- Include `batch_size` from engine config when building `EngineStatus`

### Phase 2: Gateway Configuration

**Add batching config to `GatewayConfig`** (`backend/src/config.rs`, line ~97):

```rust
/// Enable request batching
#[serde(default)]
pub batching_enabled: bool,
/// Maximum time to wait for batch to fill (milliseconds)
#[serde(default = "default_batch_timeout_ms")]
pub batch_timeout_ms: u64,
/// Minimum batch size before sending (if timeout not reached)
#[serde(default = "default_min_batch_size")]
pub min_batch_size: u32,
```

Add defaults:
```rust
fn default_batch_timeout_ms() -> u64 { 50 }
fn default_min_batch_size() -> u32 { 1 }
```

Update `Default` impl and config loader defaults.

### Phase 3: Batch Queue Module

**Create `backend/src/gateway/batch_queue.rs`**:

Key structures:
- `QueuedRequest` - Holds request + oneshot response channel + timestamp
- `BatchQueueConfig` - batch_timeout, min_batch_size
- `ModelQueue` - Per-model queue with requests and first_request_at timestamp
- `BatchQueue` - Main manager with `RwLock<HashMap<String, ModelQueue>>`

Key methods:
- `enqueue(model, request)` -> `oneshot::Receiver<Result<Response, Error>>`
- `should_dispatch(model, runner_batch_size)` -> bool (check size OR timeout)
- `take_batch(model, max_size)` -> `Option<RequestBatch>`
- `pending_models()` -> `Vec<String>`

### Phase 4: Batch Dispatcher

**Create `backend/src/gateway/batch_dispatcher.rs`**:

- `BatchDispatcher` struct with `BatchQueue`, `RunnerRegistry`, `InferenceRouter`
- `run()` method - async loop that:
  - Listens for queue notifications
  - Periodically checks for timeout-based dispatch (every 10ms)
  - Calls `try_dispatch(model)` which checks batch readiness and dispatches
- `dispatch_batch()` - Routes batch to runner, sends responses via oneshot channels
- `get_runner_batch_size(model)` - Query registry for max batch_size among runners with model

### Phase 5: Router & Registry Integration

**Update `ConnectedRunner`** (`backend/src/gateway/registry.rs`, line ~37):
```rust
/// Maximum batch size this runner supports (from engine status)
pub batch_size: u32,
```

Update `register()` to extract max batch_size from `status.engines`.

**Add method to router** (`backend/src/gateway/router.rs`):
```rust
pub async fn chat_completion_batched(...)
```
This method enqueues to BatchQueue and awaits the oneshot response.

### Phase 6: Chat Handler Integration

**Update `chat_completions` handler** (`backend/src/routes/chat.rs`):

1. Check if batching should be used:
   ```rust
   let use_batching = state.config.gateway.batching_enabled
       && !request.stream.unwrap_or(false);
   ```

2. If batching enabled for this request, use `chat_completion_batched()`
3. Otherwise use existing `chat_completion()` path

**Streaming requests bypass batching** - they need immediate per-token response.

### Phase 7: Module Registration

**Update `backend/src/gateway/mod.rs`**:
```rust
pub mod batch_queue;
pub mod batch_dispatcher;

pub use batch_queue::{BatchQueue, BatchQueueConfig};
pub use batch_dispatcher::BatchDispatcher;
```

**Initialize in main/app setup**:
- Create `BatchQueue` with config
- Spawn `BatchDispatcher::run()` task
- Pass queue reference to router/handlers

## Files to Modify

| File | Changes |
|------|---------|
| `simple-ai-common/src/protocol.rs` | Add `batch_size` to `EngineStatus` |
| `inference-runner/src/config.rs` | Add `batch_size` to engine configs |
| `inference-runner/src/gateway/status.rs` | Include batch_size in status |
| `backend/src/config.rs` | Add batching config to `GatewayConfig` |
| `backend/src/gateway/mod.rs` | Export new modules |
| `backend/src/gateway/registry.rs` | Add `batch_size` to `ConnectedRunner` |
| `backend/src/gateway/router.rs` | Add `chat_completion_batched` method |
| `backend/src/routes/chat.rs` | Use batched path when appropriate |

## New Files

| File | Purpose |
|------|---------|
| `backend/src/gateway/batch_queue.rs` | Request queue and batching logic |
| `backend/src/gateway/batch_dispatcher.rs` | Batch dispatch loop |

## Configuration Examples

**Runner config** (`inference-runner/config.toml`):
```toml
[engines.llama_cpp]
enabled = true
batch_size = 8
```

**Gateway config** (`backend/config.toml`):
```toml
[gateway]
enabled = true
batching_enabled = true
batch_timeout_ms = 50
min_batch_size = 1
```

## Verification

1. **Unit tests**: Test BatchQueue enqueue/dispatch logic with mocked time
2. **Integration test**: Start runner with batch_size=4, send 4 concurrent requests, verify they batch
3. **Streaming test**: Verify streaming requests bypass batching
4. **Timeout test**: Verify requests dispatch after timeout even with batch_size not reached
5. **Manual test**: Run gateway + runner, check logs for batch dispatch messages
