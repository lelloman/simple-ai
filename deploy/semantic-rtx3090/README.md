# JEV-9B semantic decisions on RTX 3090

This replaces the retired Qwen3.6-35B-A3B AWQ / SGLang experiment. The public
loopback semantic API stays on port 18040; vLLM is on loopback port 30000.
Historical Qwen measurements remain in `docs/evals/semantic-rtx3090/`.

The checkpoint is `autotrust/JEV-9B`, pinned to
`4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35`. Serving uses the upstream
`adapter_vllm` decision adapter, including its trained output head, with the
published bias and per-kind temperature applied by our provider. No answer text
is parsed. This is decision-only use of JEV; it is itself based on Qwen3.5-9B.

## Runtime

The deployment reuses the RTX host's existing `qwen38-27b-3090:latest` base image,
identity `sha256:4c0aa3f009b6ae7795bbdaf2b696bfcee7a6cea1e17f1f90b8ab72dea7aa55d8`:
vLLM 0.27.1, PyTorch 2.13.0+cu130, Transformers 5.15.0. `prepare.sh` checks this
identity before building a small `simple-ai-jev:20260929` layer. It registers
Qwen3.5's `lm_head` with vLLM's existing embedding-LoRA implementation; without
this registration the upstream adapter is rejected. The version-checked patch
is in `enable-head-lora.py`. Other deployments using the base image are unaffected.

Configuration: BF16, 18,432-token context, maximum four backend sequences,
2,048-token prefill chunks, 90% GPU allocation, eager execution, prefix caching
with `mamba-cache-mode=align`. The provider submits two concurrent branches.
Cache usage telemetry is enabled. Development/admin routes are disabled by
default; set `JEV_BENCHMARK_MODE=1` only during an exclusive cache-reset benchmark.
A full state plus question/options must fit the context; no silent truncation.

## Prepare and run

On the RTX host, from this directory:

```bash
./prepare.sh
docker compose up -d
docker compose logs -f vllm provider
curl -f http://127.0.0.1:18040/health
```

The model download is stored once in `jev-rtx3090_models`; both containers share
that volume. Downloads are about 18 GB. Offline serving is enabled after preparation.
Both published ports bind to 127.0.0.1; use SSH forwarding for remote access.

The runtime is managed by simple-ai's `decisions` engine. Use the gateway's
`POST /v1/decisions`; see [API usage](../../docs/decisions.md). The runner owns
startup, shutdown, and GPU switching. Manual compose commands above are for
initial preparation and maintenance only, with other GPU workloads stopped.

The homelab idle manager may power the host off during preparation. Wake it via
`POST http://192.168.1.101:8090/nodes/gpu-server-rtx/wake`, then send
`POST .../keepalive` every minute while working. Stop keepalives when finished.

```bash
docker compose down
```

The JEV model volume is retained. The retired semantic Qwen cache, container,
and unused SGLang image were removed during migration; unrelated chat weights
and historical result artifacts were retained.

See [API and validation](../../docs/semantic-scoring.md) and the
[upstream model card](https://huggingface.co/autotrust/JEV-9B/tree/4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35).
