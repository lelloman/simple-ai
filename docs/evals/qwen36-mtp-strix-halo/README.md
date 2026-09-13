# Qwen3.6 35B-A3B MTP on both Halos — 2026-09-13

This directory contains the benchmark and deployment tools for accelerating the
existing `qwen3.6-35b-a3b` alias. The base MXFP4_MOE weights are retained.

## Deployed result

Enabled on both machines: Q4_0 draft head, maximum 7 draft tokens,
`p_min=0.75`, Q4_0 target/draft KV cache, 16 CPU threads, batch/microbatch 2048,
one slot and 262,144 context allocation. Six and seven draft tokens were close;
seven slightly led the final equal-token workload aggregate and long-prompt
test, while six was slightly faster on the short code prompt. These are the
fastest validated configurations in this comparison, not an exhaustive optimum.

Measured through each runner's normal HTTP API, thinking disabled:

| Workload | Halo1 before, tok/s | Halo1 after, tok/s | Halo2 after, tok/s |
|---|---:|---:|---:|
| Code, short prompt | 56.86 | 81.18 | 85.64 |
| Prose, short prompt | 55.32 | 68.43 | 65.55 |
| JSON, short prompt | 54.64 | 111.08 | 116.82 |
| Code after 17,143 prompt tokens | not measured | 74.28 | 76.42 |

On Halo1 this is about 43% faster for code, 24% for prose, and 103% for JSON.
Warm short-prompt time to first token fell from 0.52–0.58 s to 0.19–0.24 s.
These runner checks use one run per workload; the isolated final-candidate
comparison uses two runs. The before measurement was taken only on Halo1;
Halo2's isolated no-MTP reference was approximately 58 tok/s.

All ten correctness checks and the non-streaming default-effort check passed
through **both** runner APIs. A separate Halo2 lifecycle check verified that
runner restart terminates the native inference child and reloads the alias
successfully. The live gateway lists `qwen3.6-35b-a3b` with default effort
`none`. Full authenticated gateway inference was not exercised in this test.

Compact evidence is in `matrix-results.json` and the `halo1/` and `halo2/`
directories. Complete synthetic request/response transcripts remain on the
machines. Rollback configurations:

- Halo1: `/home/lelloman/config.toml.before-qwen36-mtp-20260913T191028Z`
- Halo2: `/home/lelloman/config.toml.before-qwen36-mtp-20260913T190903Z`

## Artifacts and runtime

- Base: `/home/lelloman/model-catalog/Qwen3.6-35B-A3B-MXFP4_MOE.gguf`, linked to
  the existing Unsloth Hugging Face cache on each machine.
- Draft source: [lym00/Qwen3.6-35B-A3B-MTP-ONLY-GGUF](https://huggingface.co/lym00/Qwen3.6-35B-A3B-MTP-ONLY-GGUF/tree/8b0bc1a7cdb2fcb8a2153b15b0f7051564f34779),
  revision `8b0bc1a7cdb2fcb8a2153b15b0f7051564f34779`.
- Q4_0 draft: 1,190,098,592 bytes; SHA256
  `3af1cdff776ea65ef88db396119d178f6576b9b36f6eb0b578ea360e9189f552`.
- Q8_0 draft also tested: 1,989,103,392 bytes; SHA256
  `3440d938509649cfcdf3564a5e418d245da3b39fbde30f29b80215678d7af4b4`.
- Validated runtime: existing portable Strix Halo Vulkan release v0.7.2,
  source `ad914eb6587d3da8b2bf50f0056cc20b3d3e91f5`, bundled Mesa RADV and libdrm.
  Archive SHA256: `3f0147249de2b85f7be10f379038b4eaac3b2ea27f496505ef7415bbc020d737`.
  Runtime location: `/home/lelloman/flash-next/runtime/vulkan/llama-server`.

## Method

Halo2 was taken out of the runner pool during isolated testing; Halo1 remained
available. Test servers listened only on `127.0.0.1:18036` and were stopped after
each configuration. Full GPU offload, 262,144 context allocation, one slot,
temperature zero, thinking disabled, three different prompts (code, prose,
JSON), 512 generated tokens per prompt. A short warmup precedes measurements.
The reported decode rate is measured from the first streamed text token to the
last chunk, `(completion_tokens - 1) / seconds`; it is not a model-quality score.

The matrix tests draft lengths 1, 2, 3, 4, 6, 7, and 8, probability thresholds
0.5/0.75/0.9, Q4/Q8 draft precision, Q4/Q8 KV cache, 4/16 CPU threads, and
512/2048 batch/microbatch sizes. This is an adaptive comparison, not every
combination. The final candidates repeat each prompt twice and add known-answer
checks, a ~17k-token retrieval prompt, generation after a ~17k-token prompt,
cached repeated/unrelated requests, a tool call, and sampled requests.

The portable runtime ignores the container fork's per-request draft-limit
overrides. Its meaningful comparisons therefore use a fresh server process for
each CLI configuration. The interrupted `portable-sweep` run is excluded.
The initial `mtp1` run overlapped a draft download and is excluded as well.
Raw transcripts, timings, and logs remain under
`/home/lelloman/qwen36-mtp-test/results/` on the hosts.

## Rejected configuration

`localhost/strix-qwen38-mtp:463c0f6` supports loading the separate draft but
produced incorrect answers after successive unrelated requests: the prose
prompt continued the previous Python code, and subsequent outputs repeated
code fences. `--spec-mtp-strict-qwen`, disabling cache storage, and Q8 KV with
smaller batches did not resolve the failures. Its apparent speed gains are
invalid. This resembles the reported [MTP inter-request state issue](https://github.com/ggml-org/llama.cpp/issues/26425),
but the exact underlying cause was not established here.

The portable runtime passed the same checks. `llama-launcher.py` dispatches
only the exact Qwen3.6 MXFP4_MOE filename to it and forwards all other models'
arguments unchanged to Podman. It uses `exec`, preserving runner process-group
ownership and unload behavior. No runner binary or shared model implementation
is changed.

## Deployment and rollback

`configure.py` stages a TOML candidate, validates it, and checks that other model
profiles and container arguments are unchanged. With `--apply` it saves
`/home/lelloman/config.toml.before-qwen36-mtp-TIMESTAMP` before replacing the
live file. Each runner is restarted separately, with the other machine serving.
The fast model's default reasoning effort is explicitly `none`, matching its
only advertised effort. Its model alias and 262k context limit are retained.

Reproduce isolated benchmarks with `benchmark.py --portable --draft N
--draft-quant q4_0 --label NAME`; use `--validate` for correctness and long-prompt
checks. Run `matrix.py` and `matrix.py --phase hardware` for the initial grids.
These scripts require the test models under `/home/lelloman/qwen36-mtp-test/models`
and an otherwise idle Halo; do not run them alongside the production runner.
Use `check-runner.py --validate` to exercise the installed alias through the
runner HTTP API. `check-lifecycle.py` restarts an idle runner to verify native
process cleanup and model reload.

To roll back, restore that host's saved config and restart
`systemctl --user restart simple-ai-runner`. This restores the Podman launcher
and original Qwen3.6 settings; the additional files can remain on disk.

The measurements cover single-request speed at short and ~17k occupied
contexts. A 262k allocation is not a benchmark of a fully occupied 262k prompt.
Two hosts provide two independent inference slots. Larger batches of
concurrent requests and broad model-quality evaluation are outside this test.
