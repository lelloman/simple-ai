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

## Halo measurements — 2026-09-18

Repeated the same benchmark on both AMD Ryzen AI Max+ 395 / Radeon 8060S hosts: same model revision, texts, schemas, eight CPU threads, inference batch size four, two warmups and ten measured repetitions. Both serving environments use PyTorch 2.13.0+cpu. An additional isolated ROCm run used an existing container on Halo 1; no serving configuration or serving Python environment was changed.

| Workload | Halo 1 CPU | Halo 2 CPU | Halo 1 Radeon GPU | RTX 3090 GPU (earlier) |
| --- | ---: | ---: | ---: | ---: |
| Short text | 44.35 ms | 45.41 ms | 6.70 ms | 19.81 ms |
| Medium document, chunked | 387.54 ms | 394.17 ms | 25.41 ms | 21.14 ms |
| Batch of 16 short inputs | 320.98 ms | 341.18 ms | 32.48 ms | 79.56 ms |

These are warm inference medians, not end-to-end request latencies. Halo 1 ROCm returned the same sample entity labels, text and offsets as its CPU run for all three workloads; scores differ with FP16. Peak PyTorch GPU allocation was 1,463.56 MiB, excluding driver/context reservations. This smoke comparison is not full precision/quality qualification.

**Runtime difference:** the Halo GPU container uses PyTorch `2.11.0a0+rocm7.11.0a20251223`, HIP `7.2.53150`, Python 3.13.11 and Transformers 4.57.3, whereas RTX uses PyTorch 2.13.0+cu130 and Transformers 4.57.6. Treat the GPU columns as results for these installed stacks, not an isolated hardware ranking. `device: "cuda"` in the ROCm JSON is PyTorch's API spelling; execution was on Radeon 8060S, not NVIDIA.

Container: `docker.io/kyuz0/vllm-therock-gfx1151:latest`, image ID `6572ae5da5eeaca2ed327ce57df8d8e92a7833a439c82c519b9c8e67d6eff974`. The benchmark used `/dev/kfd` and `/dev/dri`, `--group-add keep-groups`, and `--security-opt label=disable`, with the pinned model mounted read-only. `gliner2==2.0.0` and `peft==0.21.0` were installed with `--no-deps` in a separate mounted package directory and exposed through `PYTHONPATH`; the container's PyTorch was retained. See the `scripts/benchmark-extraction.py` command interface to repeat CPU or GPU runs.

Raw results: [Halo 1 CPU](halo1-cpu.json), [Halo 2 CPU](halo2-cpu.json), [Halo 1 ROCm GPU](halo1-rocm.json). GPU inference was tested on Halo 1 only; Halo 2 had a different container inventory. Both production Halo providers remain configured for CPU.

A CPU control run in the **same ROCm container** gave medians of **43.96 / 389.15 / 309.59 ms** (short / medium / batch16), close to the installed CPU environment. The corresponding GPU speedups were **6.56× / 15.31× / 9.53×**. This control set `OPENBLAS_NUM_THREADS=8` and `OMP_NUM_THREADS=8` as well as the script's eight PyTorch threads: an initial run without the BLAS cap oversubscribed CPU threads and was stopped without producing a completed report. Raw controlled results: [halo1-rocm-cpu.json](halo1-rocm-cpu.json). CPU remains FP32 and GPU FP16, so these are practical configuration comparisons, not precision-matched measurements.
