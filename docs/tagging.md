# Batched site tagging (AI-63, AI-64)

`POST /v1/chat/completions` accepts `model: "class:tag"`. Keep the existing
SanCerchino tags-v2 messages and `response_format`; send 16 independent requests
concurrently. There is no client-side batch envelope. Example:

```json
{"model":"class:tag","messages":[{"role":"user","content":"Classify this website: ..."}],"max_tokens":100,"response_format":{"type":"json_schema","json_schema":{"name":"tags","strict":true,"schema":{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}}},"required":["tags"],"additionalProperties":false}}}}
```

The schema above illustrates the transport, not SanCerchino's taxonomy. Use its
real tags-v2 schema unchanged. The shared request retains the entire
`response_format`, including schema/name/strict, through gateway and runner.
vLLM's OpenAI server performs guided decoding; this is not a prompt-only JSON
instruction. Non-vLLM chat backends return HTTP 400 with
`response_format requires a vLLM-backed chat model`, including when the backend
is an older runner (the gateway checks its existing engine/model status).
The gateway's direct Ollama mode also rejects it. Legacy requests omit the new
optional field and keep their behavior. Use `response_format`, not the obsolete
`guided_json` vLLM extension.

The selected RTX profile is **Qwen3.6-35B-A3B AWQ**, public/served model
`qwen3.6-35b-a3b-tag-awq`, using the existing AI-60 venv (vLLM 0.29.0).
The checkpoint is `engines.tag.model_path`. The operator measured 5.07 sites/s
at 16 concurrent requests, zero errors on 500 sites, and 85% agreement with
tags-v2 versus 54% for the 4B. Live gateway qualification remains required.

`engines.tag.vllm_args` is an optional list of argv strings that **replaces**
all provider tuning defaults, rather than appending to them. Identity, loopback
host and port are supplied by the runner; do not override those in the list.
When absent, the original 4B defaults remain: bf16, context 2048, 16384 batched
tokens, 32 sequences, GPU utilization .92, prefix caching and distributed
executor `uni`. An empty list delegates tuning to vLLM. The 35B profile below
intentionally omits `--dtype` to use AWQ's default; the 4B settings OOM during
gated-delta-rule prefill on the 3090.

The runner forces `chat_template_kwargs.enable_thinking=false`, discards
reasoning controls for this class, and defaults to greedy 100-token generation.
Explicit temperature/max_tokens remain supported. Requests beyond the context
limit return vLLM's 400; source facts are never truncated. Tagging currently
supports non-streaming chat only; `stream:true` returns a clear 400.

## Model ownership and failure recovery

AI-60's synchronous offline provider does not share its loaded engine with the
async server. Tagging therefore loads a separate engine **on demand**, using the
configured checkpoint on disk. `tag`, `summaries` and every existing RTX GPU engine
must remain in the same `cuda:0` resource group. They evict/reload each other;
they are not resident simultaneously at the configured GPU utilization. This avoids changing
the summary API/provider or depending on room for two engines in 24 GB VRAM.
Do not run summary traffic during the throughput probe: alternating engines
incurs cold loads. No Halo configuration changes are needed.

Each active tag request retains the provider lifecycle read lock through its
response, even when its caller disconnects. Unload waits for them. Transport,
body parsing and HTTP 5xx failures kill the provider process group (including
vLLM's engine-core child); the next request reloads it. Ordinary 400/429 errors
retain the healthy engine. Quiesce cleans up the cached process group even when
the provider parent has already exited, before another GPU engine can load.
Startup cancellation/failure also kills descendants.
There is a 600-second startup and 180-second inference timeout. A fatal engine
failure can fail concurrent tag requests; callers may retry after reload.

## Operator rollout (RTX only)

Agents do not deploy, restart, reconfigure or access RTX. After review and merge:

1. Reuse the serving venv at `/home/lelloman/.simple-ai-summary/venv`.
   The selected checkpoint is
   `/home/lelloman/simple-ai-models/Qwen3.6-35B-A3B-AWQ`.
   This AWQ build needs `quantization_config.modules_to_not_convert` rebuilt
   from the safetensors dtypes. The operator has **already patched** that copy;
   preserve the patch when copying/restoring it. The summary provider's
   `check_checkpoint` only accepts compressed-tensors W8A8 and rejects AWQ.
   The tag provider skips that validator and does not import the summary
   provider; vLLM validates/loads its configured checkpoint. It has no
   dependency on the deleted `~/.simple-ai-summary/model`.
2. Copy `scripts/simple_ai_tag_provider.py` to
   `/home/lelloman/simple-ai-tag-provider.py`. No summary Python module needs
   to be copied alongside it. Ensure checkpoint parents are readable.
3. Drain `gpu-server`, wait for active work to finish, install the updated RTX
   runner, and merge `deploy/tag/rtx.runner-overlay.toml` into its existing
   config, retaining every existing `cuda:0` member. The runner ID stays
   `gpu-server`. On the runner's existing systemd unit ensure this venv's bin
   directory is prepended to its existing PATH, preserving the remainder:
   `Environment="PATH=/home/lelloman/.simple-ai-summary/venv/bin:<existing PATH>"`.
   No new service or secret is needed. Keep the existing unit's
   `KillMode=control-group` (systemd default); never use `KillMode=process`.
   No `VLLM_ENABLE_V1_MULTIPROCESSING=0` override is needed for the async engine.
   Run daemon-reload only if the unit changed, restart that runner, re-enable it,
   and confirm it advertises the selected tag model. The runner still admits up to
   32 concurrent HTTP requests; vLLM schedules at most 16 sequences, queueing
   excess calls. Use 16 concurrent calls for qualification.
4. Deploy the merged backend image and merge
   `deploy/tag/backend-overlay.toml` into existing tables. This adds `tag` to
   model classes, class preferences and wake targets for `gpu-server`.
   Deploying runner before enabling the gateway class prevents requests from
   reaching an unprepared provider. Existing Halo runners remain compatible.
   Confirm discovery, `X-Runner-Id: gpu-server`, the response
   `model: qwen3.6-35b-a3b-tag-awq`, `X-Model-Revision` and audit placement through
   the gateway. Existing class-request permissions cover `class:tag`.
5. Change SanCerchino's model setting to `class:tag`; keep its request/schema
   and use 16 concurrent calls. Do not start the 38k-site run before qualification.

The checked-in `deploy/tag/rtx.runner-overlay.toml` contains this profile
(merge into existing tables, preserving all other CUDA resource members):

```toml
[engines.tag]
enabled = true
command = ["/home/lelloman/.simple-ai-summary/venv/bin/python", "/home/lelloman/simple-ai-tag-provider.py"]
model = "qwen3.6-35b-a3b-tag-awq"
model_path = "/home/lelloman/simple-ai-models/Qwen3.6-35B-A3B-AWQ"
vllm_args = ["--max-model-len", "2048", "--max-num-seqs", "16", "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--language-model-only"]

[model_routes."qwen3.6-35b-a3b-tag-awq"]
engine = "tag"
engine_model = "qwen3.6-35b-a3b-tag-awq"
```

In the backend, replace the existing class list with
`[models] tag = ["qwen3.6-35b-a3b-tag-awq"]` (TOML table and assignment on
separate lines), retaining the `gpu-server` class preferences/wake targets in
`deploy/tag/backend-overlay.toml`. Roll out provider + runner + runner config
first, then backend config/image; no Halo update is needed.

Rollback: restore SanCerchino's prior model, drain the RTX runner, disable only
`engines.tag`, remove the tag route/class mappings, and restart/re-enable RTX.
Leave summaries and all other RTX resource memberships unchanged.

## Operator qualification still required

No agent evidence establishes RTX speed, real vLLM startup or tagging quality.
On homelab use `~/sancerchino-eval/never-tagged-500.txt` and the 500 existing
tags-v2 results to prepare exactly aligned real requests. Export JSONL rows as
`{"id":"site-id","request":{...actual tags-v2 messages/response_format...}}`.
Do not substitute the illustrative taxonomy above. Privately export the existing
SanCerchino `SIMPLE_AI_API_KEY`, then run:

```bash
python3 scripts/check-tags-live.py --input tags-v2-500-requests.jsonl --output ai63-live.json
```

The harness warms one request, then times 16 concurrent requests, retains
ID-aligned outputs/usage/runner IDs and input SHA256, checks JSON parsing, and
fails below **4.6 sites/s**. Verify outputs against the actual JSON schema as well;
JSON parsing alone is insufficient. Compare per-site tags with the 500 prior
tags-v2 outputs, inspect disagreements/unsupported tags and record the quality
comparison on AI-63 before the bulk run. The owner selected the 35B profile based on the operator comparison above. Record actual mean input
and output tokens (target shape ~870/~100), cold load time, VRAM, and summary ↔
tag reload behavior. Keep test assets/results outside the repository.

Local gate: `CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0`, then
`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace -- --test-threads=4`, and
`python3 -m unittest discover -s tests -p test_tag_provider.py`.
Rust tests cover class routing, schema forwarding/rejection, concurrent requests,
cancelled callers, ordinary errors, fatal failure/reload and descendant cleanup.
The Python test checks launch arguments without importing vLLM. No schemas,
contracts or lockfile generators are introduced. This repository has no PR CI;
only merging to master triggers backend image publishing.
