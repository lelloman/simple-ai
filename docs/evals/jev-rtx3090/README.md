# JEV-9B replacement validation — 2026-09-29

The standalone semantic scorer was replaced with `autotrust/JEV-9B` revision
`4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35`. GPU and HTTP validation passed.
The old semantic Qwen checkpoint/cache, container and unused SGLang image were
removed. Free disk on the RTX host increased from approximately 16 GiB to 60 GiB
**after** downloading JEV: roughly 44 GiB reclaimed overall. The unrelated chat
model/deployment and historical Qwen results were retained.

JEV was stopped after validation and the ordinary chat container and
`simple-ai-runner` user service were restored. JEV remains installed and ready
for exclusive-GPU standalone use via [Compose](../../../deploy/semantic-rtx3090/README.md).
It is not a gateway-routed capability and does not share the GPU allocator.

## Configuration

RTX 3090, 24,576 MiB; BF16 weights, vLLM 0.27.1, PyTorch 2.13.0+cu130,
Transformers 5.15.0. 18,432-token context, four maximum sequences,
2,048-token prefill chunks, 90% GPU allocation, eager execution, prefix caching
with `mamba-cache-mode=align`; two client branches in flight.

The small derived runtime image registers `lm_head` as an embedding-LoRA
module in Qwen3.5. This makes vLLM load JEV's trained decision adapter rather than
rejecting it. We use the exact upstream template, slot bias and temperature.
Image identities and source hashes: [manifest](run-manifest.json).

## Validation

- 14 CPU protocol/API tests passed, including real HTTP test servers.
- Real tokenizer matched all 24 trained verbalizer IDs; six template roundtrips passed.
- Live HTTP score, score-many, classification and 16-option requests passed.
  17 options and over-context input returned HTTP 400. [HTTP evidence](http-smoke.json)
- All 24 labelled synthetic quality examples were correct. None of the four
  deliberately ambiguous examples received maximum probability >= 0.9.
  This small fixture does not establish accuracy or calibration on application data.
  [Full quality report](quality.json)
- 16,000-token states worked for one and eight decisions without truncation.

## Initial performance

Three repetitions per point, warm kernels and cold KV at each trial; synthetic
build-log states. These are smoke measurements, not reliable p95 estimates or
maximum-throughput measurements. Timings include tokenization/native backend
HTTP and exclude outer API HTTP, model startup and initial kernel compilation.

| State tokens | Single-decision median | Eight decisions, shared-state median | Shared decisions/s |
| ---: | ---: | ---: | ---: |
| 500 | 259.4 ms | 1,690.4 ms | 4.73 |
| 2,000 | 589.5 ms | 1,674.5 ms | 4.78 |
| 16,000 | 4,719.7 ms | 5,724.2 ms | 1.40 |

Peak sampled device VRAM was 22,110 MiB (21.6 GiB). No prefix-cache hits were
reported for the 500-token workload; 2K and 16K batches reused 1,584 and 15,840
prefix tokens on later branches. Against sequential cache-cleared decisions,
8-question batching reduced elapsed time by approximately 1.24x, 2.81x and
6.77x respectively, excluding cache-reset overhead.

The largest shared/independent True-score delta in the benchmark comparisons
was 0.017. Scores are not claimed to be numerically invariant or calibrated.

The [retired Qwen configuration](../semantic-rtx3090/README.md) measured
153.5 / 398.2 / 2,760.5 ms for single decisions at the same state lengths.
**This initial JEV configuration is slower**, despite using fewer total model
parameters. Both configurations have eager execution and constrained concurrency;
this migration did not tune CUDA graphs, quantization or maximum throughput.

[Raw benchmark](benchmark.json). Reproduce on an otherwise idle GPU:

```bash
JEV_BENCHMARK_MODE=1 docker compose -f deploy/semantic-rtx3090/compose.yaml up -d
docker compose -f deploy/semantic-rtx3090/compose.yaml exec vllm \
  /app/venv/bin/python /work/scripts/benchmark-semantic.py --exclusive-backend \
  --lengths 500 2000 16000 --batches 1 8 --repetitions 3 --output /tmp/jev-benchmark.json
```

Copy results before stopping/recreating the container. Disable benchmark mode
for normal serving. JEV's classification limit is 16 options (previously 64).
`label_probability_mass` is no longer returned, since the JEV readout is
restricted to its trained answer slots; see [API semantics](../../semantic-scoring.md).

## Gateway integration validation (2026-09-29)

The deployed gateway now routes `/v1/decisions` to the managed RTX engine.
[Integration evidence](integration.json) records real gateway responses:

| Scenario | End-to-end time |
| --- | ---: |
| Chat → JEV, three mixed questions | 116.60 s |
| Warm repeat, three mixed questions | 0.47 s |
| JEV → chat, reply `READY` | 84.85 s |
| Powered-off RTX → decision via gateway wake | 121.09 s |
| Cold-starting chat → JEV after startup-race fix | 70.25 s |

These are smoke-test observations, not latency percentiles. Boolean, choice,
and six-level rating responses passed checks. Authentication, invalid input,
wrong class, and oversized body checks returned 401/400/413 as expected.
The wake response ID matched its database audit row, including wake flag and
182 prompt / 3 completion tokens. Existing NLI classification remains separate.

Validation found a boot race in the existing managed chat engine: Docker can
restart it before it advertises a loaded model. Resource switching now stops
its configured services regardless of HTTP readiness. Failed switches clear
ownership so subsequent requests cannot skip cleanup. Regression coverage
also checks cancellation during startup and same-owner backend recovery.

The final source has 473 passing Rust tests across the workspace (full suite
plus the updated runner suite) and 17 passing CPU provider tests. The seven
resource-registry tests cover the lifecycle cases above. GPU memory ownership
is retained through detached inference and timeout cleanup.
