# Qwen3 PyTorch embeddings (AI-59)

Serve **Qwen/Qwen3-Embedding-0.6B** on **gpu-server-rtx** through
`POST /v1/embeddings`. It is discovered through `GET /v1/models` once the
updated RTX runner registers. Selecting this explicit id requires the existing
`model:specific` permission. `class:embed_large` and `embedding:large` retain
their existing Qwen3 8B mapping.

The AI-57 Halo 2 Ollama candidates (Qwen 0.6B and BGE-M3) were never deployed.
Their rollout overlay is removed; do not install those routes. BGE-M3 is not
part of this rollout. The old AI-57 benchmark is retained as historical tooling;
it expects both former candidates and is not this ticket's acceptance command.
The owner's RTX measurements found llama.cpp/Ollama serving the small Qwen
about **3× slower** than fp16 PyTorch (~54 versus 150–157 texts/s).

## Input and model semantics

`input` is one string or an ordered array of strings. The provider uses CUDA
fp16, transformers SDPA, left padding, last-token pooling, and float32 L2
normalisation. Output is 1024 finite dimensions with unit norm; no dimension
reduction. Each input is truncated to **512 tokens**. No instruction is inserted:
send raw documents/domain profiles. Queries may supply their own prefix, e.g.
`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <query>`
with an actual newline.

Requests allow 1–256 nonblank texts, at most 32,768 Unicode characters per text
and 1 MiB of UTF-8 text total; excess is HTTP 400. These conservative bounds cap
tokenisation and response memory independently of the GPU batch. Inputs process
in order in microbatches of **64** by default. HTTP inference is serialized,
with at most eight admitted requests (excess provider requests return 429).
The runner advertises one active inference slot. Timeout is 180 seconds;
startup allows 600 seconds. Model unload waits for in-flight requests even
after callers disconnect. A crashed process can be loaded again.

Pinned checkpoint:
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
of `Qwen/Qwen3-Embedding-0.6B` (Hugging Face revision resolved 2026-10-27).
Model loading is offline and uses no remote code.

## Operator rollout (user 2)

Agents must not deploy, restart or reconfigure services.

1. On RTX, copy `scripts/setup-embedding.sh` and
   `scripts/embedding-requirements.txt` together, then run
   `bash setup-embedding.sh /home/lelloman/.simple-ai-embedding`.
   This creates a **dedicated** venv with torch 2.13.0 CUDA 13.0,
   transformers 4.57.6 and the pinned checkpoint under `model/`.
   It leaves extraction/classification/other environments alone and writes the
   installed dependency lock to the install directory. Use that exact model
   directory in the overlay; a different local directory must contain the same
   pinned artifact.
2. Copy `scripts/simple_ai_embedding_provider.py` to
   `/home/lelloman/simple-ai-embedding-provider.py`. The fleet deploy scripts
   are unchanged; this provider needs this explicit copy.
3. Install the updated runner binary, merge
   [the overlay](../deploy/pytorch-embeddings/rtx.runner-overlay.toml)
   into RTX's existing config, and restart the runner after draining work.
   Put `text_embeddings` in **cuda:0**, the same resource group as all other
   RTX GPU engines. Preserve those members and all current routes. This uses
   existing GPU exclusivity, so another engine can cause an unload/reload.
   Do not add embedding class lists or aliases. No gateway/protocol upgrade is
   required for serving valid requests; deploy the updated backend as well
   before acceptance to get early HTTP 400 validation at the gateway.
4. Confirm model discovery, CUDA residency in logs/`nvidia-smi`, and run the
   live qualification below. Roll back by disabling `text_embeddings` and
   removing only its route after work drains; preserve existing resource groups.

## Verification

Local checks:

```bash
export CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads=4
python3 -m unittest discover -s tests -p test_embedding_provider.py
```

Python tensor tests require CPU or CUDA torch; they download no model. They
verify last-token pooling, float32 normalisation, output dimension, raw text,
token truncation settings, microbatch order and recovery after inference errors.
Rust checks cover provider lifecycle/caller cancellation/crash recovery, invalid
provider outputs, input bounds, explicit gateway routing and model discovery.

Live acceptance is **pending the operator**, not established by unit tests.
User 2 has the 2026-10-27 RTX reference vectors. Export `SIMPLE_AI_API_KEY`
without recording the secret. Save the exact reference texts and vectors as
`{"input": ["raw text"], "embeddings": [[1024 floats]]}` and valid requests for
every other RTX engine as an array of `{"path": "/v1/...", "body": {...}}`.
Use concrete models on RTX, not requests that can route to a Halo.

```bash
python3 scripts/check-pytorch-embeddings-live.py \
  --reference /tmp/ai59-reference.json --probes /tmp/ai59-rtx-probes.json \
  --output /tmp/ai59-live.json
```

The read-only harness calls **https://ai.lelloman.com**, checks discovery,
1024-dimensional unit vectors, minimum reference cosine 0.99 and >=120 texts/s
for 700-character inputs at batches 32/64 and concurrency 1/4. Each cell warms
separately, then times 2048 completed texts. Any failed request aborts and saves
an incomplete report. It then alternates each supplied RTX probe with embeddings
three times, recording reload latency.

Before rollout, record those same other-engine probes and latency on the
existing deployment. After rollout compare outputs and latency against that
baseline, including warm reloads; inspect gateway audit runner IDs and GPU
logs to prove the probes hit RTX and resources released. The harness's HTTP
success alone does not establish no regression: attach that comparison and
operator assessment. Do not close live acceptance without it.

After qualification the operator separately switches SanCerchino's
`SIMPLE_AI_MODEL_EMBED` and re-embeds profiles; that is outside AI-59.
