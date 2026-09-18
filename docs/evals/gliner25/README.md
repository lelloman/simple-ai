# GLiNER2.5 CPU/CUDA comparison — 2026-09-18

Measured sequentially on the RTX host with the same pinned model, schemas, input texts and PyTorch 2.13.0+cu130. CPU uses FP32 and 8 threads; CUDA uses FP16 on RTX 3090. Thus this compares practical deployment configurations, including precision differences. Each workload has two warmups and ten timed repeats. Timing is provider model inference, excluding process load, gateway/network latency and queueing.

| Workload | CPU median | CUDA median | Speedup |
| --- | ---: | ---: | ---: |
| Short Italian text, four entity labels | 56.47 ms | 19.81 ms | 2.85× |
| Longer text, chunked | 607.84 ms | 21.14 ms | 28.76× |
| 16 short inputs, inference batch size 4 | 437.05 ms | 79.56 ms | 5.49× |

Peak PyTorch CUDA allocation over these workloads: 1,463.56 MiB. This excludes driver/context reservations and is not a capacity guarantee for larger schemas or batches. SDPA was unavailable for this DeBERTa stack, so the loader selected eager attention. Compilation was not enabled.

Raw timings, sample outputs, exact text lengths, revision and runtime metadata are in [rtx-cpu.json](rtx-cpu.json) and [rtx-cuda.json](rtx-cuda.json). The benchmark source is [benchmark-extraction.py](../../../scripts/benchmark-extraction.py). These synthetic inputs demonstrate performance differences, not representative production accuracy or tail-latency guarantees. FP16 can change confidence scores and threshold decisions; validate application quality on the chosen device.

CPU is the initial fleet configuration because it is already interactive for short texts and can run alongside large GPU chat models. CUDA is available as a runner setting for workloads where its measured benefits justify GPU ownership and potential chat-model reload costs.

## Fleet rollout verification

Gateway and all three runners deployed from implementation commit `e6eb417`. Extraction is enabled in CPU mode on RTX, Halo 1 and Halo 2; RTX has a CUDA-capable environment for later device selection. Existing remote settings were preserved and only the extraction section was appended. Pre-rollout config/binary backups were retained on each runner.

The live smoke check succeeded on every runner for a two-text batch combining entities, classification and relations, with exact Unicode substring validation. Initial requests including model loading took 7.05 s on RTX and 4.54 s on each Halo. These cold timings differ from the warm performance table. Raw requests and responses: [fleet-smoke.json](fleet-smoke.json).

The public gateway advertises the model through `/v1/models` and rejects invalid credentials at `/v1/extractions`. Full authenticated gateway-to-runner inference was verified in the isolated real-binary integration test with mock OIDC; no production user credential was created or modified for testing.

Validation: 439 Rust workspace tests passed, four Python provider-contract tests passed, and actual model examples passed through the provider and runner. Strict Clippy remains blocked by pre-existing warnings (including derivable defaults in OCR/speech); Clippy without warning promotion completed with no warnings in the new extraction files.
