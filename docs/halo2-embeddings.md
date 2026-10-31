# Halo2 query embeddings (AI-65)

Status: staged, **not qualified for rollout**. No parity or live headroom result
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

Routing is deliberately fixed to runner ID strix-halo-02 for this explicit
checkpoint. There is **no RTX fallback**, including when Halo is cold, offline,
draining, disabled, or circuit-broken. Selection fails with the existing
no-runner error when Halo is unavailable. This protects RTX batches through
outages as well as normal operation. Automatic wake, speculative preparation and capacity expansion are skipped for
this model; halo2 must already be connected. Other embedding model IDs/classes keep
their routes. No gateway API or wire protocol changes are needed; older RTX
runners continue to work. The new halo2 config requires the updated runner.

## Exact configuration changes

Merge [halo2.runner-overlay.toml](../deploy/pytorch-embeddings/halo2.runner-overlay.toml)
into halo2's current runner config. Merge entries into existing tables; do not
duplicate table headers or replace existing GPU members. Keep all existing
engines, aliases, routes and credentials.

Backend config.toml diff: **empty**. Routing policy is in the backend, with
no new preference settings or class mappings. RTX config diff: **empty**;
keep its engine and cuda:0 group for rollback and direct parity measurements.

## Operator installation and gate

Agents do not perform these steps. Back up halo2's runner binary/config,
provider files and backend image reference first.

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
   using the existing service procedure. Keep the previous backend running.
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
5. Only after parity and headroom pass, deploy the new backend image. No config
   changes are needed. Resume callers. Issue repeated query embeddings while RTX
   tags; gateway audit records must show only strix-halo-02 and RTX's tag engine
   must remain resident with no tag failures. Test Halo unavailable in maintenance:
   embeddings must fail instead of reaching RTX.

## Rollback

Restore the previous backend image first to restore old routing (embedding
traffic may again evict RTX engines; pause interactive embeddings during rollback).
Drain halo2, remove only the overlay's engine, resource and route entries, restore
its previous runner binary/config and restart. Existing RTX and halo2 engines
and resource groups remain intact; no vectors or databases are migrated.
