# Site summaries (AI-60, incomplete qualification)

`POST /v1/summaries` uses the existing gateway authentication, model permissions,
audit, scheduler, wake and model preparation. It returns one ordered English
site summary per input, with actual tokenizer usage:

```json
{"model":"class:summarize","input":["Example Ltd runs a bookshop in London."],"max_tokens":60}
```

```json
{"model":"Qwen/Qwen3-4B-Instruct-2507","data":[{"index":0,"summary":"Example Ltd operates a bookshop in London."}],"usage":{"prompt_tokens":45,"completion_tokens":12,"total_tokens":57}}
```

The response is illustrative. Default `max_tokens` is 60, allowed range 1–60.
Accept 1–64 nonblank texts, at most 32,768 Unicode characters each and 1 MiB total
UTF-8. The provider additionally rejects prompts whose chat template plus output
budget exceeds 1,024 model tokens, without truncating source facts. A limit can
truncate generated summaries; two or three complete sentences are a prompt
instruction, not a guaranteed result. The fixed system prompt is in
`scripts/simple_ai_summary_provider.py`: facts only, names as written, no
inferred names/places and no preamble.

## Current delivery boundary

**No RTX matrix, quality gate, live throughput or reload measurements have been
run by this delivery. No quantized model or student has been selected.** The
provider/overlay serve the conservative **fp16 4B baseline**, not a claim of
meeting >=15 sites/s. Do not enable the overlay in production before completing
qualification. Quantization preparation and choosing the measured production
configuration remain AI-60 work; there are no placeholder throughput numbers.
The ticket's 8.8 sites/s is operator context, not a measurement of this code.

The owner comment supersedes the description's stretch: distill only below
roughly 30 sites/s after tuning, use Qwen3-1.7B only, train on ~100k **card**
summaries, and ship a student only within two grounding points of the 4B with no
invented names or places. No 0.6B student or training on another host.

## Lifecycle and compatibility

The engine `summaries` advertises the fixed 4B model using the existing engine
status/model-list fields; no WebSocket enum/variant or wire version is added.
Old runners work unchanged. `class:summarize` is resolved by the gateway class
list (and directly by the runner endpoint), not an `aliases.mappings` entry
that would rewrite the advertised model ID to a class name. The backend is upgraded first; only RTX needs the
new engine. The explicit route never selects `class:big` or a Halo model.

The provider starts offline, bound to loopback, with local weights and an
isolated HF_HOME; implicit token use is disabled. Runtime is CUDA fp16, prefix
caching, context 1024, 16384 batched tokens, 256 sequences, GPU utilisation 0.92,
greedy generation. All GPU inference runs in the provider process (`uni`
executor, `VLLM_ENABLE_V1_MULTIPROCESSING=0`), so killing/reaping it releases CUDA
before another engine owns the shared resource. This setting follows the
[vLLM environment contract](https://docs.vllm.ai/en/latest/configuration/env_vars/).

It admits one HTTP inference batch; concurrent provider callers get 429.
The runner advertises one request slot. Startup budget is 600 seconds; inference
budget 180 seconds. Cancellation retains both the resource lease and the engine
read barrier through the detached inference task. Unload waits for active
inference; transport/body failures terminate the process before releasing the
barrier; crashed providers can reload. Resource groups are per runner: adding
summaries to RTX's existing `cuda:0` group can evict **RTX** embeddings/other
engines, and requires measuring their reload latency. Halo2's slots are untouched.

## Operator preparation and eventual rollout

Agents do not deploy, restart or reconfigure production services.

1. Provide an approved RTX address/SSH route and copy
   `~/sancerchino-eval/cards-20261028.jsonl` there, plus access to the existing
   `~/emb-test/llm/summ.py`, `score.py`, profiles and baseline outputs. Preserve
   aligned site IDs and the held-out 1500 examples. Supply a separate training
   export only if distillation is triggered.
2. Benchmark fp16, W8A8 INT8 (not FP8), and AWQ INT4 on RTX. Quantization must use
   an isolated toolchain and separate calibration data, not the held-out set;
   record the actual compressor/compressed-tensors versions and weight revision.
   Never read/change `~/.cache/huggingface/token`. Set a dedicated HF_HOME and
   `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. Do not disturb deployed models/services.
3. After selecting the quality-qualified winner, update the provider's fixed
   model/config and metadata to that winner and repeat local checks. The current
   fp16 setup script is only a baseline: copy `scripts/setup-summary.sh` to RTX
   and run `bash setup-summary.sh /home/lelloman/.simple-ai-summary LOCAL_4B_DIR`.
   It installs vLLM 0.29.0 in a dedicated venv, reuses the supplied offline weights
   via symlink, and records pip freeze. Install the venv/bin on PATH for ninja.
4. After code review/merge, deploy backend plus merge
   `deploy/summaries/backend-overlay.toml` into existing config. Then copy
   `scripts/simple_ai_summary_provider.py` to
   `/home/lelloman/simple-ai-summary-provider.py`, install the RTX runner binary,
   merge the RTX overlay preserving current GPU members, drain and restart RTX.
   No Halo binary/config changes. Existing backend routes remain. Rollback:
   disable only the summaries engine/route after draining it, then remove its
   backend class mapping. No fleet deploy script behavior is changed.
5. User 2 tests through the gateway with the SanCerchino key and confirms the
   quality/throughput/reload gates below. Inspect audit runner IDs and GPU logs
   to establish RTX placement and resource release; HTTP success alone is not
   proof of coexistence or no regression.

## Matrix commands and reports

The input file schemas and scoring CLI are operator assets unavailable in the
session. Normalize *copies* to canonical JSONL `{"id":"site-id","text":"..."}`:
1500 aligned distinct IDs, original profile text or card text, no added facts.
Keep original files for score.py and record the conversion. Do not use held-out
sites for quantization calibration or student training.

Copy the benchmark and provider scripts alongside each other. In the ticket's
existing RTX container, put its venv on PATH. Use the ticket's
`--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host`, not `--gpus all`.
Expose the benchmark directory under `/w` and set the offline HF environment.
For each weight directory (fp16, W8A8, AWQ), each input and each sequence count:

```bash
for weights in fp16 w8a8 awq; do
  for input in profiles cards; do
    for seqs in 256 512; do
      /app/venv/bin/python /w/benchmark-summaries.py \
        --model-path "/w/$weights" --input "/w/$input.jsonl" \
        --max-num-seqs "$seqs" --output "/w/$weights-$input-$seqs.json"
    done
  done
done
# Repeat the best quality-qualified cell with --ngram (3 speculative tokens).
```

These are commands to run, not completed results. The harness saves the exact
invocation, vLLM version, input SHA256, elapsed time, sites/s, wall GPU-seconds per
1000 (exclusive GPU occupancy, not CUDA kernel time), peak **total device** MiB
sampled from nvidia-smi at 100ms, and ID-aligned summaries. Keep the GPU exclusive
for meaningful timings. It warms only a synthetic site, not held-out examples.
Sampling includes model startup and inference. The optional benchmark-only
sequence/ngram parameters are not production configuration knobs. N-gram window
2–5 follows the [vLLM speculative configuration](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

Run the operator's actual `score.py` CLI on every output against fp16 on the
**same sites/input**. Its format must be inspected before adapting outputs;
there is deliberately no invented replacement score. Add the score and delta to
each report, manually inspect 30 fixed-ID examples for invented places/names,
and attach a table of every completed cell with exact commands. Retain failed
cell logs too. Scores are initially null to prevent an unscored run passing.

After rollout and before close, user 2 exports `SIMPLE_AI_API_KEY` privately and
runs:

```bash
python3 scripts/check-summaries-live.py --input cards.jsonl \
  --reload-probe rtx-probe.json --output ai60-live.json
```

The probe contains `{"path":"/v1/embeddings","body":{"model":"Qwen/Qwen3-Embedding-0.6B","input":["Example"]}}`
or a concrete model for another RTX engine. The harness warms a synthetic input,
then times 1000 cards, batches <=64, fails below 15 sites/s, saves every summary,
and alternates the probe/summary three times for reload timings. It is a timing
capture, not a grounding/coexistence certificate: apply score.py, inspect 30
outputs and compare the other engine to its pre-rollout latency/outputs. User 2
must confirm via the gateway with the SanCerchino key; never commit that key.

## Local checks

```bash
export CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads=4
python3 -m unittest discover -s tests -p test_summary_provider.py
bash -n scripts/setup-summary.sh
```

Rust tests check routing/discovery to RTX, input/output validation, cancellation,
timeout and crash/reload. Python tests use a fake LLM (no GPU/download) to verify
the prompt, token bounds without truncation, order, usage, busy rejection and
recovery after inference failure. They do not establish model quality or speed.
