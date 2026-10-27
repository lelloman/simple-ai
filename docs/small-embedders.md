# Small embedding candidates (AI-57)

This is an operator rollout and qualification plan, not a production result.
The existing Ollama engine already supports batched dense embeddings. No
backend or protocol change is needed. Upgrade the runner binary from this PR:
its Ollama lifecycle handling falls back to empty-input `/api/embed` when an
embedding-only model rejects `/api/generate`, for both loading and unloading.
The fallback is limited to Ollama’s HTTP 400 unsupported-generate error; other
errors still fail acquisition. Failed warmups are not cached. See the upstream
[empty-input embedding load handler](https://github.com/ollama/ollama/blob/v0.12.0/server/routes.go#L544-L579).
`class:embed_large` and its current Qwen3 8B mapping stay unchanged.

## Placement and artifacts

Use **Halo 2 (`strix-halo-02`)**, provisionally: the
[ownership runbook](runner-host-states.md) identifies it as the SimpleAI host,
while Halo 1 is assigned to agents. AI-56 also selects Halo 2 for MADLAD.
This avoids taking an agent host or adding work to RTX's existing GPU services.
It is not live headroom evidence: before rollout the operator must confirm
ownership, available unified/GPU memory and the existing service inventory.
The candidates' weight files total about 1.8 GB, in addition to runtime buffers,
activations, MADLAD and existing models. Do not infer runtime residency from
weight size alone.

Use the existing GPU-enabled Ollama service on loopback, or have the operator
prepare a qualified Strix Halo GPU-enabled Ollama installation if absent.
Do not point at a remote or CPU-only Ollama instance. These are the upstream
Ollama distributions of the requested models, rather than new Python providers:

| Public model id | Ollama tag | Weights | Manifest SHA256 (2026-10-27) |
| --- | --- | --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` | `qwen3-embedding:0.6b` | Q8_0, 639,150,592 bytes | `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d` |
| `BAAI/bge-m3` | `bge-m3:567m` | F16, 1,157,671,200 bytes | `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` |

Manifests and config blobs were inspected at
`https://registry.ollama.ai/v2/library/<name>/manifests/<tag>`.
Record actual installed digests and Ollama version in the live report; reject
a changed manifest until its identity is checked. The Qwen candidate uses Q8_0
rather than original BF16 weights to reuse the existing runtime with little
weight compression; the comparison must identify that quantization. BGE uses
F16. Neither representation has been qualified on production by this change.

Operator steps (do not run these from an agent session):

1. Capture the existing embedding/classification latency baseline below before
   loading either candidate. Record loaded models, memory, runner versions,
   class mapping and host ownership from the admin UI/API.
2. On Halo 2, pull `qwen3-embedding:0.6b` and `bge-m3:567m` with `ollama pull`.
   Confirm `ollama list` IDs match the manifest prefixes above. Merge
   [the runner overlay](../deploy/small-embedders/halo2.runner-overlay.toml)
   into existing tables, preserving all current settings. Keep Ollama in the
   same resource group as **all** other GPU engines, including AI-56's
   translation engine. Use the actual existing group name if it differs from
   `gpu:0`. Do not add candidates to gateway embed class lists or change the
   `embedding:large` alias. Existing explicit-model permissions apply:
   use a client key with `model:specific` for this qualification.
3. Have the operator install this PR’s runner binary, then restart the runner
   with the merged config. No backend or other runner rollout is required for AI-57. AI-56 has its own backend-first
   rollout and has landed separately; qualify coexistence again after
   its operator-managed rollout.
4. Verify both public ids in gateway `/v1/models`; call each via
   `/v1/embeddings` with one text and a batch, checking 1024 finite nonzero
   values per text. Check `ollama ps` during inference for GPU residency and
   inspect runner/provider logs and memory. CPU fallback fails the GPU scope.
5. Run the loaded qualification below. Attach its JSON and a report of memory,
   residency and routing. GPU resource groups serialize different engines and
   may unload/reload models; measure those costs, including translation.
   Shared grouping protects memory, but does **not** guarantee simultaneous
   residency or no regression. If qualification shows degraded existing
   service latency, do not accept the rollout; the operator must resolve
   capacity/placement before ticket closure.

## Input semantics

Both models produce 1024-dimensional dense vectors natively. The gateway
forwards the provided strings; it does not insert prompts or reduce dimensions.
BGE's Ollama `/api/embed` result is dense only: no sparse or ColBERT output.

For Qwen retrieval queries, construct
`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <query>`
with an actual newline. Use the same English task description for all query
languages. Documents/domain profiles are raw text, with no instruction prefix.
BGE-M3 needs no query or document instruction. The throughput test uses raw
700-character domain-profile text for every model, including the 8B baseline.
Do not mix prompted queries and unprompted documents in the same timing result.

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[BGE model card](https://huggingface.co/BAAI/bge-m3),
[Qwen Ollama distribution](https://ollama.com/library/qwen3-embedding:0.6b),
[BGE Ollama distribution](https://ollama.com/library/bge-m3:567m).

## Reproducible gateway qualification

The standard-library harness changes no configuration. Provide the same
production probe payloads before and after rollout, targeting existing
embedding/classification models. For example, save this as a local file
`/tmp/ai57-probes.json` (use the actual existing classifier id):

```json
{
  "/v1/embeddings": {"model": "class:embed_large", "input": ["Un'impresa italiana offre assistenza e servizi alle aziende."]},
  "/v1/classifications": {"model": "class:text_classification", "input": ["Un'impresa italiana offre assistenza alle aziende."], "labels": [{"label": "business", "hypothesis": "This text is about business."}, {"label": "sport", "hypothesis": "This text is about sport."}]}
}
```

Check the classification payload against the deployed API; if necessary copy
an existing successful request. Neither probe may select a candidate. Keep
input and client key identical between phases. Export `SIMPLE_AI_API_KEY` in
the operator shell without putting it in a command line or committing it.

Before rollout/loading:

```bash
python3 scripts/check-small-embedders-live.py \
  --probes /tmp/ai57-probes.json --output /tmp/ai57-before.json
```

After rollout:

```bash
python3 scripts/check-small-embedders-live.py \
  --probes /tmp/ai57-probes.json --baseline /tmp/ai57-before.json \
  --output /tmp/ai57-loaded.json
```

The loaded phase requires candidate discovery and validates embedding count,
order, dimensions and finite nonzero vectors. It measures all three models
at batch 32, concurrency 1/4/8, with the same 512 distinct 700-character texts
per cell. Each cell has a separately reported warmup. Throughput is completed
texts divided by wall time; HTTP/validation failures abort the run, persist an
incomplete report and never count as successful work. Request p50/p95 latency
is also recorded. Timeouts are 600 seconds per HTTP request, not successful
throughput samples.

The matrix has no competing probes. Then each candidate is measured at
concurrency 4 with continuous existing embedding/classification probes; these
**contention** results are separate from the pure throughput matrix. Twelve
probe pairs are collected before the matrix and after contention, in addition
to the twelve before rollout. Compare probe p50/p95 against the before-rollout
baseline and inspect individual warmup costs. There is no invented pass/fail
threshold: attach the measurements and have the operator assess whether the
acceptance condition of no regression is met.

Gateway latency alone cannot prove which runner served a probe or whether
weights remained resident. Correlate the run time window with gateway audit
runner ids, `ollama ps`, GPU/system memory and runner logs. Keep AI-56 MADLAD
loaded/active for a second qualification run once available, recording its
request timings as well. Translation qualification remains a separate operator
check; the harness does not drive that endpoint. Synthetic profile text is for
throughput only;
the operator's SanCerchino multilingual retrieval evaluation remains separate.

Rollback: the operator removes only these two aliases/routes, restores the
previous Ollama enablement/resource-group settings if this rollout changed
them, and restarts the runner after active work drains. Preserve all existing
model/class mappings and AI-56 settings. No model downloads are deleted.
