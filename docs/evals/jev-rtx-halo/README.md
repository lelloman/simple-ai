# JEV-9B: RTX 3090 versus Strix Halo

Measured on 2026-09-29. Results compare the working serving stacks, not hardware
in isolation: RTX uses vLLM 0.27.1/CUDA; Halo uses AMD's gfx1151 vLLM 0.19.1/ROCm
image with the standalone text-model compatibility backport included here.
See `manifest.json` for exact versions, image identities, and source hashes.

## Results

Median milliseconds per request; higher decisions/sec is better.

| State tokens | Questions | RTX ms | Halo ms | RTX decisions/s | Halo decisions/s | RTX speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 93.2 | 169.8 | 10.73 | 5.89 | 1.82× |
| 32 | 8 | 684.1 | 1,328.7 | 11.69 | 6.02 | 1.94× |
| 100 | 1 | 92.5 | 205.0 | 10.81 | 4.88 | 2.22× |
| 100 | 8 | 671.4 | 1,496.5 | 11.92 | 5.35 | 2.23× |
| 250 | 1 | 120.3 | 268.7 | 8.32 | 3.72 | 2.23× |
| 250 | 8 | 957.3 | 1,958.8 | 8.36 | 4.08 | 2.05× |
| 500 | 1 | 252.4 | 467.1 | 3.96 | 2.14 | 1.85× |
| 500 | 8 | 1,664.9 | 3,485.4 | 4.81 | 2.30 | 2.09× |
| 2,000 | 1 | 590.0 | 1,538.6 | 1.69 | 0.65 | 2.61× |
| 2,000 | 8 | 1,688.8 | 3,927.7 | 4.74 | 2.04 | 2.33× |
| 5,000 | 1 | 1,434.3 | 4,097.7 | 0.70 | 0.24 | 2.86× |
| 5,000 | 8 | 2,304.1 | 6,196.6 | 3.47 | 1.29 | 2.69× |
| 16,000 | 1 | 4,781.4 | 15,921.1 | 0.21 | 0.06 | 3.33× |
| 16,000 | 8 | 5,709.1 | 18,493.7 | 1.40 | 0.43 | 3.24× |

RTX is faster at every tested size. At 32–250 state tokens, single decisions
take about 93–120 ms on RTX and 170–269 ms on Halo. For 16,000 tokens,
RTX is about 3.3× faster. Very short requests have a latency floor; fewer
tokens do not necessarily yield proportionally lower latency. These are
boolean questions over synthetic states, not a universal latency guarantee.

Both stacks classified all 24 labelled fixtures correctly and agreed on all
28 predictions. Neither assigned ≥90% confidence to the four ambiguous cases.
Maximum labelled probability difference in the final quality reports is
0.002894 (0.29 percentage points). This small fixture is a compatibility check,
not a broad evaluation of decision quality.

The initial 500/2,000/16,000 run and the requested short-state follow-up were
separate server sessions with the same inference settings. The short-state
run adds 32, 100 and 250 tokens. A further matched run adds 5,000-token states
(`--lengths 5000`, all other benchmark settings unchanged). Five samples per
point; all raw samples and cache telemetry are retained in the six
`*-benchmark.json` files. Halo’s
older API omits cached-token telemetry when there are no hits, represented
as null. Successful engine cache resets are recorded in `halo-server.log`.

## Method

The pinned JEV checkpoint and its trained LM-head LoRA adapter are identical.
Both use BF16 weights, float32 recurrent state, eager execution, 18,432-token
context, 2,048-token chunked prefill, four maximum sequences, prefix caching
with Mamba `align`, and two concurrent question branches. Halo's KV cache is
explicitly limited to about 3.95 GiB, matching RTX's reported cache budget.

Synthetic build-log states contain exactly 500, 2,000 or 16,000 tokenizer tokens.
Each request asks one or eight distinct boolean questions. Five timed repetitions
per point follow shape-specific warmups. Prefix caches are cleared before each
request, outside the timed interval. Questions within a request may share state
cache. Throughput is questions divided by median request time; it is not a
saturated multi-client load test.

Timing includes local Python scorer/tokenizer work and HTTP calls to vLLM.
It excludes gateway routing, queue admission, host wake-up, model loading and
kernel compilation. RTX's client runs in the provider container; Halo's client
runs inside its engine container. Memory sampling is unavailable in these client
containers and is deliberately left null in the raw reports.

```sh
python /work/scripts/benchmark-semantic.py \
  --exclusive-backend --lengths 500 2000 16000 --batches 1 8 \
  --modes shared --repetitions 5 --context 18432 --concurrency 2 \
  --output /results/benchmark.json
```

On RTX use `/app/venv/bin/python` and `--backend http://vllm:30000` in the
provider container. Set `JEV_BENCHMARK_MODE=1` when starting its compose stack.
On Halo, `start-halo.sh` records the exact isolated launch. Stop the production
runner and any GPU workloads before either experiment; restore them afterward.
The benchmark resets the engine's cache and must not share it with live traffic.

## Halo compatibility

The preinstalled Halo vLLM 0.14 image cannot load Qwen3.5. The separate image used
here is documented in [AMD's gfx1151 vLLM guide](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/vllm.html).
Its Qwen3.5 text implementation exists but was only used under the multimodal
wrapper. `enable-halo-head-lora.py` registers that existing text class, reuses
its wrapper's hybrid-cache metadata methods, backports the text configuration
hook from the working vLLM 0.27.1 installation, and registers the LM-head LoRA.
The configuration hook selects the checkpoint's float32 recurrent state and
removes inherited multimodal RoPE fields in memory. Checkpoint files and
inference kernels are unchanged. The benchmark-only cache-reset endpoint also returns the engine’s existing
boolean success result, instead of silently returning empty HTTP 200.
The patch only affects a disposable container;
Halo's production image and runner configuration are unchanged.

An initial RTX trial was discarded after the quality check overlapped its final
point and cache reset correctly failed. The saved RTX benchmark is the complete
replacement run. Five samples are useful for a practical comparison, not robust
tail-latency estimates; the raw `p95_ms` is effectively the maximum of five.

## Restoration

The Halo test container is stopped and its production runner is active. RTX’s
benchmark compose stack is removed and its production runner is restored.
The temporary keepalive is stopped after restoration so normal idle shutdown
can resume. No Halo JEV routing or production model configuration was added.

Halo retains the isolated 17 GiB benchmark directory (including weights) and
the downloaded AMD runtime image (36.3 GB logical image size, shared layers may
reduce incremental disk usage), so the experiment can be repeated without
another download. These are test artifacts, not running services.

The semantic provider’s 17 CPU tests pass. The benchmark gained `--modes` so
a comparison can run the shared-state path alone; default behavior remains
both shared and independent modes.
