# RTX 3090 semantic scoring measurements — 2026-09-20

Historical: this deployment was replaced by JEV-9B on 2026-09-29. The old model cache and SGLang image were removed; these measurements are preserved. See [current service](../../semantic-scoring.md).

Qwen3.6-35B-A3B AWQ runs on one RTX 3090 with 16K states and real shared-prefix
reuse. Long-state batches benefit substantially; short-state batches benefit
little. This is an experimental service, not a validated confidence gate.

**Stopped at the user’s request.** Saved 25 of 32 batch-matrix points.
The single-decision runs at all four lengths are complete (20 repetitions each).
The remaining batch points, targeted numerical-repeatability check, and real-GPU
outer HTTP smoke test were not completed. The experimental backend and temporary
idle-manager keepalive were stopped; model downloads and the container are retained.

## Configuration and method

- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB, driver 580.173.02, 350 W power limit.
- Model: `mattbucci/Qwen3.6-35B-A3B-AWQ`, revision `7525d0f423bd615da6cc3cf3ae6fdae42941ea05`.
- Runtime: SGLang v0.5.19, image digest `sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`.
- AWQ/Marlin, BF16 activations, Triton attention, TP=1, CUDA graphs disabled.
- Context 18,432; KV token capacity 24,576; chunked prefill 2,048; static memory fraction 0.90.
- Client concurrency eight; **effective backend concurrency two**, clamped by the ten-slot recurrent-state cache. See [startup evidence](startup.log).
- Synthetic repeated DNS build logs; exact state lengths verified by retokenization. Each predicate has a distinct numbered suffix.
- Kernels warmed at each shape; measured trials begin with cold KV. Multi-predicate shared calls warm the prefix once; single decisions use one direct readout.
- Independent evaluation is sequential, with cache cleared between decisions. Both raw wall time and time excluding reset overhead are retained.
- Latency includes tokenizer preparation and native backend HTTP; excludes the optional outer HTTP service and model startup.
- Device-wide VRAM sampled about every 50 ms. The GPU reached 87°C during the sustained run; clocks and thermals were not controlled.
- **20 repetitions** for single-decision latency. **One measured pass** for each batch-matrix point: throughput figures are exploratory, and their JSON p95 fields are not useful tail estimates.

Run commands, from the repository root on the GPU host:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate-semantic.py --output results/quality.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark-semantic.py --exclusive-backend --batches 1 --repetitions 20 --output results/single-decision.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark-semantic.py --exclusive-backend --repetitions 1 --output results/batch-matrix.json
```

## Single-decision latency

Cold state, one predicate, one selected-token readout. Twenty samples per length.

| State tokens | Median ms | p95 ms | Peak VRAM MiB |
| ---: | ---: | ---: | ---: |
| 500 | 153.5 | 153.9 | 21,668 |
| 2,000 | 398.2 | 400.2 | 21,942 |
| 8,000 | 1236.4 | 1241.0 | 21,938 |
| 16,000 | 2760.5 | 2770.3 | 21,940 |

Raw trials: [single-decision.json](single-decision.json).

## Shared-state throughput

Each row is one state evaluated against N predicates. Speedup removes cache-reset
overhead from the independent baseline; both total wall times include all normal
preparation and inference work. N=1 takes the same direct path in both modes.

| State tokens | N | Shared s | Independent s | Shared decisions/s | Shared ms/decision | Speedup excluding reset | Peak shared MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 1 | 0.153 | 0.154 | 6.52 | 153.4 | 1.00× | 21,668 |
| 500 | 2 | 0.383 | 0.312 | 5.22 | 191.4 | 0.80× | 21,712 |
| 500 | 4 | 0.672 | 0.627 | 5.95 | 167.9 | 0.91× | 21,712 |
| 500 | 8 | 1.252 | 1.258 | 6.39 | 156.5 | 0.98× | 21,712 |
| 500 | 16 | 2.412 | 2.524 | 6.63 | 150.7 | 1.02× | 21,712 |
| 500 | 32 | 4.740 | 5.020 | 6.75 | 148.1 | 1.03× | 21,712 |
| 500 | 64 | 9.363 | 10.086 | 6.84 | 146.3 | 1.05× | 21,712 |
| 500 | 128 | 18.666 | 20.182 | 6.86 | 145.8 | 1.05× | 21,712 |
| 2,000 | 1 | 0.399 | 0.398 | 2.51 | 398.9 | 1.00× | 21,942 |
| 2,000 | 2 | 0.556 | 0.811 | 3.60 | 278.1 | 1.44× | 21,944 |
| 2,000 | 4 | 0.837 | 1.631 | 4.78 | 209.3 | 1.91× | 21,944 |
| 2,000 | 8 | 1.400 | 3.270 | 5.71 | 175.0 | 2.28× | 21,944 |
| 2,000 | 16 | 2.526 | 6.561 | 6.33 | 157.9 | 2.53× | 21,944 |
| 2,000 | 32 | 4.771 | 13.136 | 6.71 | 149.1 | 2.68× | 21,944 |
| 2,000 | 64 | 9.263 | 26.294 | 6.91 | 144.7 | 2.76× | 21,944 |
| 2,000 | 128 | 18.247 | 52.594 | 7.01 | 142.6 | 2.80× | 21,944 |
| 8,000 | 1 | 1.243 | 1.251 | 0.80 | 1243.4 | 1.01× | 21,938 |
| 8,000 | 2 | 1.465 | 2.502 | 1.37 | 732.6 | 1.70× | 21,942 |
| 8,000 | 4 | 1.756 | 5.001 | 2.28 | 439.1 | 2.83× | 21,942 |
| 8,000 | 8 | 2.333 | 10.030 | 3.43 | 291.6 | 4.27× | 21,942 |
| 8,000 | 16 | 3.490 | 20.010 | 4.58 | 218.1 | 5.69× | 21,942 |
| 8,000 | 32 | 5.802 | 39.945 | 5.52 | 181.3 | 6.83× | 21,942 |
| 8,000 | 64 | 10.456 | 80.090 | 6.12 | 163.4 | 7.59× | 21,944 |
| 8,000 | 128 | 19.785 | 160.211 | 6.47 | 154.6 | 8.03× | 21,944 |
| 16,000 | 1 | 2.765 | 2.766 | 0.36 | 2764.6 | 1.00× | 21,940 |

Raw trials, cache telemetry, scores and timing breakdowns: [batch-matrix.json](batch-matrix.json).

## Cache reuse and numerical agreement

Observed cached tokens per branch, for multi-predicate requests:

| State tokens | Cached tokens observed |
| ---: | --- |
| 500 | 512 |
| 2,000 | 1984, 2048 |
| 8,000 | 8000 |

The independent trials report zero cached tokens. Alignment/recurrent-state
checkpoints can leave a small tail to recompute; the full requested prefix is
not necessarily reusable. These are measured radix-cache hits, not a pinned fork.

The maximum absolute True-score difference between shared and independent paths
was **0.1384**. Across 766 paired decisions, **1 selected-label disagreements** occurred:

| State tokens | Batch N | Decision | Shared True score | Independent True score |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 128 | 44 | 0.377541 | 0.500000 |

A tie selects the first label (`True`) by the API’s deterministic tie rule.
Even identical cold single-call comparisons show some variation, so the shared
path is not the only source of numerical variability. The artifacts preserve
raw selected logprobs. Do not treat a threshold near an ambiguous score as stable
or claim numerical equivalence from performance results.

## Quality and calibration

The synthetic fixture has 24 labelled cases and 4 deliberately ambiguous cases. The labelled cases
cover binary predicates, routing, multiclass, relevance, rule matching, and
confidence-sensitive decisions. Ambiguous cases are excluded from accuracy and
calibration metrics.

| Metric | Measured value |
| --- | ---: |
| accuracy | 1.000000 |
| multiclass_brier | 0.001254 |
| binary_brier | 0.000886 |
| nll | 0.011582 |
| ece_10_equal_width_bins | 0.011242 |

All 24 labelled examples received confidence above 0.9; 2 of the four ambiguous examples
also received confidence at least 0.9. This small, easy fixture cannot establish
calibration or production accuracy. Scores remain explicitly `calibrated: false`.
No calibration temperature was fitted.

Full predictions, category metrics, reliability bins and selective coverage:
[quality.json](quality.json). Tokenizer boundaries and 64 single-token answer
labels: [tokenizer-check.json](tokenizer-check.json).

## Operational limits and next experiment

The first successful launch took about nine minutes, most of it CUDA kernel
compilation. FP16 activations failed a recurrent-state dtype assignment; this
profile therefore uses BF16. The container retains compiled kernels until it
is recreated. The checkpoint cache is a separate persistent volume.

The shared-prefix speedup is real for this long-state workload, but bounded
concurrency and eager execution still impose substantial per-decision overhead.
Before using this as an agent confidence gate, investigate numerical stability
and validate on a larger representative dataset, including abstention and label-order
perturbations. Then evaluate larger recurrent-state capacity and graph execution
with the same memory, cache, and score-agreement checks.

Earlier [500-token smoke](smoke-benchmark.json) and [16K smoke](long-smoke-benchmark.json)
reports used a redundant prefix warmup even for a single decision. They are kept
as historical evidence; use the final single-decision and batch reports above
for current performance.

Package versions and source hashes: [run-manifest.json](run-manifest.json).
