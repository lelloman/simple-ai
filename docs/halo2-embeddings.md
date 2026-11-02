# Halo2 query embeddings (AI-65)

Status: branch ready for operator staging, **not qualified for rollout**. No parity or live headroom result
has been collected by the agent. Do not enable the new backend routing until
the operator records the checks below.

## Choice and compatibility

Use the existing pinned Qwen/Qwen3-Embedding-0.6B checkpoint
(97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3), tokenizer, raw text,
512-token truncation, left padding, last-token pooling and float32 L2
normalisation, producing 1024 dimensions. Halo2 runs CPU float32 SDPA,
RTX retains CUDA fp16 SDPA. The CPU choice avoids a ROCm environment and GPU
evictions; numerical compatibility is conditional on the parity gate below.
Four CPU threads and microbatches of eight bound contention and working memory.

The existing halo2 snapshot has halogen, decisions and llama_cpp sharing
gpu:0, while extraction and audio are CPU providers. The new
cpu:text_embeddings group is independent: registry loading only evicts peers
with the same group. It does not reserve physical RAM. Weights need about
2.4 GB at float32, plus tokenizer, runtime and attention/activation memory.
Live free memory under the largest supported Halogen workload has **not**
been established; the operator must measure it, including CPU providers.
Do not infer headroom from nominal host RAM or the small checkpoint alone.

## Generic embedding routing

This policy applies to **all gateway embedding requests**, whether selected by
concrete model, alias or embedding class; it has no checkpoint, host-ID or
application-specific rule. Class/alias resolution still uses existing mappings.

- With `allow_engine_swap` omitted or `false`, use a runner with the resolved
  embedding model already loaded and spare advertised capacity. Prefer the
  existing `gpu-server` machine type when multiple warm runners qualify.
- If no warm runner has capacity, allow a cold load only when the advertised
  resource group has no conflicting loaded engine or other model. For this
  deployment that normally means halo2's independent CPU group, while RTX tags
  or summarizes. A cold runner missing resource metadata is conservatively
  ineligible. Busy runners are skipped using the dispatcher's existing
  runner-wide active count and model batch capacity; this may reject while an
  independent CPU engine could theoretically take more work.
- If neither is possible, return **HTTP 503**, with a message explaining that
  no runner can serve without an engine swap. Do not wake, prepare or expand
  capacity in the background. Halo down + RTX cold with another engine loaded
  therefore fails immediately. RTX warm with capacity can still serve during
  a Halo outage.
- Bulk callers explicitly send **`"allow_engine_swap": true`** to prefer an
  online GPU runner even if its embedder is cold or another engine must be
  swapped. The normal resource queue drains other engines' in-flight work
  before loading. If no compatible runner is online, this opt-in retains the
  existing configured wake/preparation path. Without an online GPU, another
  compatible online runner may serve, as before; opt-in does not require a
  particular host. No speculative/capacity expansion is started by embeddings.

Example request bodies for `POST /v1/embeddings`:

```json
{"model":"Qwen/Qwen3-Embedding-0.6B","input":"interactive query"}
```

```json
{"model":"Qwen/Qwen3-Embedding-0.6B","input":["bulk document"],"allow_engine_swap":true}
```

The boolean defaults to false and is forwarded to the runner. Upgraded runners
recheck live engine state under the existing resource gate, keeping the lease
through inference. If another engine wins after gateway selection, the default
request is rejected with 503 rather than retried on a swapping route. Default
acquisition does not call peer unload/quiesce, queue behind a conflicting engine,
or switch models inside the same engine. Ungrouped engines require opt-in at
the runner because independence cannot be established. Direct legacy Ollama mode
(`gateway.enabled=false`) similarly requires opt-in: it has no registry state.

There is no new gateway configuration, class, permission or WebSocket protocol
field. The optional HTTP field is backward compatible for opt-in requests, but
old runners cannot enforce the new no-swap policy after a stale selection.
**Upgrade both embedding runners before the backend.** Other engine APIs keep
their existing acquisition behavior. Both runners must use this checkpoint's
qualified tokenizer/pooling/normalisation/dimension before mixed routing starts.

## Exact configuration changes

Merge [halo2.runner-overlay.toml](../deploy/pytorch-embeddings/halo2.runner-overlay.toml)
into halo2's current runner config. Merge entries into existing tables; do not
duplicate table headers or replace existing GPU members. Keep all existing
engines, aliases, routes and credentials.

Backend config.toml diff: **empty**. Routing policy is in the backend, with
no new preference settings or class mappings. RTX config diff: **empty**;
keep its engine and cuda:0 group for GPU bulk work and direct parity measurements.

## Operator installation and gate

Agents do not perform these steps. Back up halo2's runner binary/config,
provider files, RTX runner binary and backend image reference first. Pause
embedding callers until qualification and backend rollout finish: the previous
backend strips the opt-in field, so upgraded runners may safely reject its
embedding requests during this maintenance window.

1. On halo2, create an isolated CPU environment (do not run the CUDA-only
   setup-embedding.sh there). Copy embedding-requirements.txt and
   simple_ai_embedding_provider.py from scripts/; install the latter at
   /home/lelloman/simple-ai-embedding-provider.py. Run:

   ```sh
   python3 -m venv /home/lelloman/.simple-ai-embedding/venv
   /home/lelloman/.simple-ai-embedding/venv/bin/pip install 'torch==2.13.0' --index-url https://download.pytorch.org/whl/cpu
   /home/lelloman/.simple-ai-embedding/venv/bin/pip install -r embedding-requirements.txt
   /home/lelloman/.simple-ai-embedding/venv/bin/pip freeze > /home/lelloman/.simple-ai-embedding/requirements.lock
   ```

   Copy RTX's pinned model directory to
   /home/lelloman/.simple-ai-embedding/model on halo2 (preserve all checkpoint
   and tokenizer files). Compare file checksums with RTX before starting.
   Do not change RTX's environment or model.
2. Build the runner using
   `CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 cargo build --locked --release -p inference-runner`.
   Drain halo2 work, install the binary and overlay, then restart its runner
   using the existing service procedure. Also drain RTX work and install/restart
   the same updated runner binary on RTX, preserving its provider/environment,
   engine config and cuda:0 group. This runner update supplies the live no-swap
   admission check; it does not change RTX's embedding engine or numeric settings.
   Keep the previous backend running throughout staging/qualification.
3. In an operator maintenance window, pause RTX batch callers. Export exactly
   200 distinct real multilingual production texts, including multilingual
   queries and documents, to /tmp/ai65-texts.json as
   `{"input": ["text", "..."]}`. Keep the text file private. Call direct
   runner addresses so gateway load balancing cannot compare a runner to
   itself (both runner APIs use port 8080 in the checked-in examples; verify
   actual ports locally):

   ```sh
   python3 scripts/check-halo-embedding-parity.py \
     --texts /tmp/ai65-texts.json \
     --halo-url http://HALO2_ADDRESS:8080 \
     --rtx-url http://RTX_ADDRESS:8080 \
     --output /tmp/ai65-parity.json
   ```

   The harness explicitly sends `allow_engine_swap=true` to both direct runners.
   These calls load the embedding engines: RTX may evict its active engine.
   Run only after batch work drains. Attach input hash, both dependency locks,
   model checksums, and report (200 pair cosines, minimum and mean).
   Accept only minimum >= 0.999. If lower, block and consider matching CPU
   dtype/kernel settings or a separately qualified ROCm fp16 provider;
   do not switch traffic or re-embed stored vectors under this ticket.
4. With Halogen at its largest supported load and existing CPU engines loaded,
   record baseline MemAvailable, provider RSS and peak memory while embedding
   batches of eight and maximum-length inputs. Verify no swap/OOM or engine
   eviction, and record interactive latency under concurrent chat/extraction.
   Confirm all previously loaded engines stay loaded via runner status and logs.
   If headroom is inadequate, stop and report measurements.
5. Only after parity and headroom pass **and both runners are upgraded**, deploy
   the new backend image. Backend `config.toml` diff is empty. Resume callers.
   Qualify these cases using response `X-Runner-Id`, audit records, loaded-engine
   status and correlated runner logs:
   - RTX embedder warm/idle: default requests use RTX.
   - RTX tagging with embedder cold: repeated default requests use halo2; tag
     residency and successful tag completions continue without unload/swap.
   - Both embedders warm/idle: default requests prefer RTX.
   - Drained bulk maintenance window: opt-in requests load on RTX and permit
     a swap after other in-flight work drains.
   - Halo unavailable + RTX cold/another engine resident: default requests
     return 503 promptly and leave RTX untouched. Opt-in retains ordinary
     queue/load/wake behavior. Restore Halo before resuming interactive use.
   - RTX embedder saturated: default requests use safe halo2 capacity, or 503
     if no safe capacity is available. A stale selection may also return 503;
     retry defaults later rather than automatically enabling swaps.

## Rollback

Pause embedding callers, then restore the previous backend image first.
An older gateway does not forward the opt-in: an upgraded runner safely defaults
it to false and may reject requests. Restoring the old RTX runner afterward
restores its prior loading behavior, including possible RTX evictions; keep
interactive embeddings paused until the operator deliberately restores service.
Drain halo2, remove only the overlay's engine, resource and route entries, restore
its previous runner binary/config and restart. Restore the prior RTX runner
binary after draining, without changing its config/provider/environment. Existing
RTX and halo2 engines and resource groups remain intact; no vectors or databases are migrated.
