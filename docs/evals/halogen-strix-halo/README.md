# Halogen on Strix Halo: uncensored benchmark results

Status: **working and benchmarked**, 2026-10-06. Both cache-mode runs completed
without request errors. The benchmark initially restored the original runner.
Subsequently, at the user's request, Halogen was integrated into halo1's runner
and deployed on both halo1 and halo2 for their `code:smart` / `qwen3.8-flash-next` aliases. See
[integration and deployment notes](../../../deploy/halogen/README.md).

Follow-up: [longer-output, 100k-context and batching investigation](decode-investigation/README.md)
found halo2 at 43.8 t/s for 1,024 generated tokens after a 100k prompt, and
quantified the loss of per-user speed under batching. The earlier 34–36 t/s
short deployment checks below are not its sustained ceiling.

[Persistent disk caching](disk-cache/README.md) is now enabled on both hosts,
with a 50 GB budget each and restore-after-reload validation completed.

## Halo2 deployment verification

Halo2 retained Fedora 43, kernel `7.2.7-100.fc43.x86_64`, with SVM/KFD support
verified. After setting BIOS UMA to 512 MiB, Linux reports 124 GiB of RAM.
The same pinned image, uncensored IQ4_XS checkpoint, tokenizer, MTP head, four
slots and cache mode 2 are deployed. No swap was used during validation.

One synthetic 24,442-token cold prompt through the runner took 15.046 seconds
to prefill (**1,624.5 tokens/s**), then generated 128 tokens at **34.05 tokens/s**.
A follow-up reused all 24,442 input tokens, prefilling 155 new tokens in
0.601 seconds and generating 128 tokens at **35.96 tokens/s**. Thinking was
disabled. Cold HTTP wall time was 45.156 seconds including model reload;
warm HTTP wall time was 5.197 seconds. These are single samples, not a controlled
host comparison. Responses and engine metrics are in `halo2/integration/`.

Plain chat, streaming, streamed tool arguments, a 32-token thinking budget,
and Halogen → existing Gemma → Halogen switching passed. An authenticated
gateway `code:smart` smoke request answered `24` for `11 + 13`; audit metadata
confirmed `strix-halo-02`, engine `halogen`, 28 prompt tokens and 3 completion
tokens at `2026-10-06T14:07:15.801271725+00:00`.

## Halo1 results

Halogen 0.16.2 served the uncensored IQ4_XS model with MTP. Normal cache mode 2:

| Cold input tokens | Prefill time | Prefill tokens/s | Decode tokens/s | Follow-up first token |
| ---: | ---: | ---: | ---: | ---: |
| 6,065 | 3.98 s | 1,526 | 45.0 | 0.535 s |
| 24,442 | 15.43 s | 1,584 | 44.5 | 0.556 s |
| 49,932 | 31.15 s | 1,603 | 31.9 | 0.535 s |

Follow-ups reused the entire preceding input, prefilling 155 new tokens, and
decoded at 44.3–46.7 tokens/s. Strict cache mode 1 also worked: cold prefill was
1,548–1,629 tokens/s and decode 35.4–46.0 tokens/s. Its cache only resumed at
the 32,768-token boundary in this experiment; smaller follow-ups re-prefilled.

Four simultaneous cold requests, each with 24,437 input and 128 output tokens,
all finished within **68.6 seconds** in both runs. In normal mode, first-token
latencies were **15.2, 30.5, 45.7, and 61.0 seconds**. Prefill admission pauses
existing streams: single-request decode speed must not be presented as a
per-user concurrency guarantee. Aggregate completion throughput including all
prefill/queue time was 7.47 tokens/s for this short-output workload.

Against the earlier observational production rate of 220–300 prefill tokens/s,
this is approximately 5–7x faster. That is **not a controlled engine-only
speedup**: the quantization, firmware allocation, workload, and context differ.
The earlier production decode observation was approximately 22 tokens/s.

All 24 streams across two smoke tests and two main runs completed with usage,
finish reason, and `[DONE]`. This was a performance smoke test, not a model
quality evaluation. Answers were readable, but the 24k cold response initially
described the offset as `N` instead of `N % 97`; do not infer coding-quality
parity from these throughput results. Outputs were deliberately capped at 128
tokens (64 in smoke tests), with thinking disabled and temperature zero.

Artifacts: `results-halogen-uncensored/` (strict cache mode 1),
`results-halogen-cache2/` (normal mode 2), their smoke directories, and
`halogen-server.log` / `halogen-cache2-server.log`. JSON includes full synthetic
requests, answers, stream chunks, timing metadata, and errors if any.

## Host and staged artifacts

- Host: `halo1.homelab`, GMKtec NucBox EVO-X2.
- Linux memory after reboot: approximately 123 GiB (previously 62.3 GiB).
- GPU reserved VRAM after reboot: 2,147,483,648 bytes (2 GiB; previously 64 GiB).
- Kernel: `7.1.13-200.fc44.x86_64`, `CONFIG_HSA_AMD_SVM=y`.
- Image: `ghcr.io/peonist-ai/halogen-flash-server:0.16.2`.
- Image digest: `sha256:0c61bf84ac22308a53f5d1ca6b86806702d7039e5ebc51cae4c66621b92fe04a`.
- Model staging directory: `/home/lelloman/halogen-benchmark/models`.
- Download container: `halogen-download-uncensored-iq4xs` (no GPU access).
- Main weights: `mradermacher/Qwen3.8-Flash-Next-Uncensored-GGUF`, revision
  `61f739cd47b26ba67764deb28c99c92501892e26`, file
  `Qwen3.8-Flash-Next-Uncensored.IQ4_XS.gguf` (about 98 GB).
- This quantization is based on `orcarouter/Qwen3.8-Flash-Next-Uncensored`.
- Auxiliary files: `peonist-ai/halogen-qwen3.8-flash-next`,
  revision `3648cf1e6a3143e8946d52301570f26003c0bdb0`,
  `qwen38-flash-next-mtp.hgn` and `tokenizer/*`.
- The previous native v2 download was stopped when the user specified uncensored
  weights. Its partial files remain in staging; no native model was served.

The production ROCmFP4 GGUF is incompatible with Halogen 0.16.2. A no-GPU
`flash_serve --repack` probe failed immediately with
`output_hc_down.weight: unknown tensor type 101`. No production weights were
modified. Halogen's `inspect` supports HGN files only, so it cannot establish GGUF
compatibility. The IQ4_XS alternative subsequently loaded successfully: 67.57 GiB
was repacked in 4.8 seconds, and the engine reported 89.1 GiB of total allocations
with the chosen KV/scratch settings. No swap was used during the checks.

The downloaded IQ4_XS header identifies `Qwen3.8 Flash Next Uncensored` and
contains 1,224 tensors with types 0, 1, 7, 13, 14, 20, 23, and 30. All
`ssm_out.weight` tensors are IQ4_XS (23), avoiding the unsupported Q6_K case.
Its chat template is byte-identical to Halogen's tokenizer template:
SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

The first downloader stalled after approximately 16 GB using rootless container
networking. The successful retry path uses `--network host` and
`HF_XET_NUM_CONCURRENT_RANGE_GETS=16`. Plain HTTP through huggingface_hub was
rejected because the file exceeds its size limit; Xet remains enabled.

`run-halogen.sh` waits for the download, starts an isolated loopback API, runs
a short smoke test followed by cold/follow-up and concurrency requests, and
stops Halogen and restores the user runner on exit. The first pass uses four
slots, a 262,144-position shared KV pool, a 16,384-token prefill chunk limit,
and strict prompt-cache mode 1. Tokenized cold prompts contain 6,065, 24,442,
and 49,932 tokens before any engine-specific changes.

The [upstream memory prerequisite](https://github.com/peonist-ai/halogen-flash-server#quickstart)
requires the UMA frame buffer / dedicated graphics memory set to its smallest
explicit BIOS value, not Auto. Its troubleshooting section specifically identifies
the EVO-X2's 64 GiB setting as preventing the weights from loading. Do not try to
compensate for this with swap or benchmark while another large GPU model runs.

## Method

1. Isolate halo1 by stopping its idle user runner; keep other machines unchanged.
2. Run `run-halogen.sh`: short smoke, one cold/follow-up pair at each of three
   sizes, and four concurrent requests. Run once with strict cache mode 1 and
   again with normal mode 2 (`HALOGEN_BENCH_CACHE=2`,
   `HALOGEN_BENCH_RUN_ID=cache2`). This is one sample per size per configuration,
   not a statistical characterization.
3. Use synthetic code conversations from `benchmark.py`; report actual token
   usage. Distinct case prefixes and zero reported cached tokens establish
   the cold cases. OS disk cache was not flushed.
4. Save raw artifacts. A role-only SSE chunk does not count as a first token.
   Tables above use engine throughput timings; HTTP first-token latency includes
   queueing and API overhead. Concurrent wall throughput includes prefill.
5. Stop the isolated server and restore the production runner on exit.

Example harness invocation against an already isolated server:

```sh
python3 benchmark.py --url http://127.0.0.1:8731 \
  --model MODEL_ID_FROM_V1_MODELS --out results-halogen-uncensored
```

The IQ4_XS quantization differs from production's custom ROCmFP4 GGUF. Comparing
those files measures complete serving configurations; an engine-only comparison
requires running the same IQ4_XS file on both engines. Existing live-log timings
are observational context, not a controlled baseline. A same-file llama.cpp
comparison, longer outputs, reasoning/tool-call correctness, sustained-load
testing, and production integration remain outside this initial experiment.
