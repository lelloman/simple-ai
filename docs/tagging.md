# Batched site tagging (AI-63)

`POST /v1/chat/completions` accepts `model: "class:tag"`. Keep the existing
SanCerchino tags-v2 messages and `response_format`; send 32 independent requests
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

The new `tag` engine uses the existing AI-60 venv, vLLM 0.29.0,
compressed-tensors 0.17.0 and local Qwen3-4B W8A8 checkpoint. The configured
public/served model defaults to `qwen3-4b-tag-w8a8`; the checkpoint is
`engines.tag.model_path`. Both are config values. A separate public ID avoids
ambiguity with the summary engine's concrete model. Changing the ID also
requires changing the model route and backend class list.

The standard async OpenAI server provides continuous batching (32 sequences,
16384 batched tokens, bf16, prefix caching, 2048-token context, GPU utilization
.92). The runner forces `chat_template_kwargs.enable_thinking=false`, discards
reasoning controls for this class, and defaults to greedy 100-token generation.
Explicit temperature/max_tokens remain supported. Requests beyond the context
limit return vLLM's 400; source facts are never truncated. Tagging currently
supports non-streaming chat only; `stream:true` returns a clear 400.

## Model ownership and failure recovery

AI-60's synchronous offline provider does not share its loaded engine with the
async server. Tagging therefore loads a separate engine **on demand**, using the
same checkpoint on disk. `tag`, `summaries` and every existing RTX GPU engine
must remain in the same `cuda:0` resource group. They evict/reload each other;
they are not resident simultaneously at .92 utilization. This avoids changing
the summary API/provider or depending on room for two engines in 24 GB VRAM.
Do not run summary traffic during the throughput probe: alternating engines
incurs cold loads. No Halo configuration changes are needed.

Each active tag request retains the provider lifecycle read lock through its
response, even when its caller disconnects. Unload waits for them. Transport,
body parsing and HTTP 5xx failures kill the provider process group (including
vLLM's engine-core child); the next request reloads it. Ordinary 400/429 errors
retain the healthy engine. Startup cancellation/failure also kills descendants.
There is a 600-second startup and 180-second inference timeout. A fatal engine
failure can fail concurrent tag requests; callers may retry after reload.

## Operator rollout (RTX only)

Agents do not deploy, restart, reconfigure or access RTX. After review and merge:

1. Prepare/reuse the existing AI-60 environment. If it is absent, use the
   [summary setup recipe](summaries.md#quantization-and-setup-operator-rtx-only),
   with the qualified checkpoint at
   `/home/lelloman/emb-test/llm/Qwen3-4B-W8A8`. No quantization or new download is
   needed. Confirm the serving venv still has vLLM 0.29.0 and
   compressed-tensors 0.17.0; do not install the quantization stack into it.
2. Copy `scripts/simple_ai_tag_provider.py` to
   `/home/lelloman/simple-ai-tag-provider.py` and copy
   `scripts/simple_ai_summary_provider.py` alongside it as
   `/home/lelloman/simple_ai_summary_provider.py` (underscore name required for
   the checkpoint-validator import). This does not replace the running summary
   provider at its hyphenated filename. Ensure checkpoint parents are readable.
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
   and confirm it advertises the tag model with 32 slots.
4. Deploy the merged backend image and merge
   `deploy/tag/backend-overlay.toml` into existing tables. This adds `tag` to
   model classes, class preferences and wake targets for `gpu-server`.
   Deploying runner before enabling the gateway class prevents requests from
   reaching an unprepared provider. Existing Halo runners remain compatible.
   Confirm discovery, `X-Runner-Id: gpu-server`, the response
   `model: qwen3-4b-tag-w8a8`, `X-Model-Revision` and audit placement through
   the gateway. Existing class-request permissions cover `class:tag`.
5. Change SanCerchino's model setting to `class:tag`; keep its request/schema
   and use 32 concurrent calls. Do not start the 38k-site run before qualification.

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

The harness warms one request, then times 32 concurrent requests, retains
ID-aligned outputs/usage/runner IDs and input SHA256, checks JSON parsing, and
fails below **10 sites/s**. Verify outputs against the actual JSON schema as well;
JSON parsing alone is insufficient. Compare per-site tags with the 500 prior
tags-v2 outputs, inspect disagreements/unsupported tags and record the quality
comparison on AI-63 before the bulk run. No quality threshold was specified;
the owner selects whether the change is acceptable. Record actual mean input
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
