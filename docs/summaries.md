# Site summaries (AI-60)

`POST /v1/summaries` uses existing gateway authentication, model permissions,
audit, scheduling, wake and model preparation. Send **card text**: name, legal
form, entity, schema type, place, title, description and top-30 keywords from
SanCerchino. Each `input` entry is a string containing one site's card, not a
JSON object or a whole scraped page. Preserve names and supplied facts; omit
missing fields, especially place. Do not derive place from a domain or language.
The caller owns extraction/serialization; the endpoint does not generate cards.
For example (the fields below are illustrative):

```json
{"model":"class:summarize","input":["Name: Example Ltd\nLegal form: Ltd\nEntity: Example Ltd\nSchema type: BookStore\nPlace: London\nTitle: Example Books\nDescription: Independent bookshop selling new and used books.\nKeywords: books, bookshop"],"max_tokens":60}
```

```json
{"model":"Qwen/Qwen3-4B-Instruct-2507","data":[{"index":0,"summary":"Example Ltd operates an independent bookshop in London. It sells new and used books."}],"usage":{"prompt_tokens":120,"completion_tokens":24,"total_tokens":144}}
```

The response and token counts are illustrative. The concrete model ID remains
`Qwen/Qwen3-4B-Instruct-2507`; it identifies the source model, while discovery
reports `W8A8-INT8` quantization. `class:summarize` selects this dedicated engine,
never `class:big`. Inputs are still bounded strings for compatibility with the
ticket's API. Card field presence/order is not schema validation: missing facts
must be omitted, not invented. Whole profiles remain accepted but have lower
measured throughput; the production throughput target applies to cards.

Default `max_tokens` is 60, range 1–60. Accept 1–64 nonblank strings, at most
32,768 Unicode characters each and 1 MiB total UTF-8. The provider rejects
prompts whose chat template plus requested output exceeds 1,024 tokens without
truncating source facts. A token limit may cut a sentence short; 2–3 sentences
are a prompt instruction. The fixed prompt in
`scripts/simple_ai_summary_provider.py` asks for English, facts only, names as
written, no preamble and explicitly omits place when absent, forbidding inference
from domain/TLD/language/names.

## Selected configuration and evidence

Owner/operator ticket comment 4960 (2026-10-28) selects **Qwen3-4B W8A8 INT8,
card input, 512 sequences, no speculative decoding**. Existing qualified
checkpoint: `/home/lelloman/emb-test/llm/Qwen3-4B-W8A8` on RTX (4.2 GB,
compressed-tensors format; vLLM auto-detects it without quantization flags).
Serving constants: vLLM 0.29.0, bf16 activations, prefix caching, context 1024,
16384 batched tokens, 512 sequences, GPU utilization .92, greedy <=60 tokens.
Production startup checks INT8 weights and activations in checkpoint metadata.

These are **operator-reported offline** measurements on RTX 3090, 1,500 sites;
not measurements of the gateway/provider shipped here. Grounding is the share
of proper nouns in the summary present in the input, measured by the operator's
`ground.py`. Exonyms account for most misses. Compare the same input profile.

| Cell | Sites/s | GPU-s/1k | In/out tokens per site | Grounded % | Delta vs fp16, same input | Peak VRAM |
|---|---:|---:|---|---:|---:|---:|
| Old text baseline, defaults, max_tokens 120 | 8.8 | 114 | 612 / 65 | — | — | — |
| fp16 text, seqs 256 | 11.8 | 84.7 | 619 / 58 | 93.7 | 0 | 23.7 GB |
| fp16 card, seqs 256 | 24.2 | 41.3 | 285 / 57 | 94.8 | 0 | 23.7 GB |
| fp16 card, seqs 512 | 25.8 | 38.7 | 285 / 57 | 94.8 | 0 | 23.5 GB |
| fp16 card, seqs 512 + ngram 3 | 20.3 | 49.2 | 285 / 57 | 94.9 | +0.1 | 23.2 GB |
| **W8A8 card, seqs 512** | **43.8** | **22.8** | **285 / 57** | **94.7** | **−0.1** | **23.4 GB** |
| W8A8 card, seqs 256 | 39.3 | 25.4 | 285 / 57 | 94.6 | −0.2 | 23.3 GB |
| W8A8 text, seqs 256 | 20.5 | 48.9 | 619 / 58 | 94.0 | +0.3 | 23.6 GB |
| AWQ int4 control | Not run: quantization OOM | | | | | |

Text/512 cells and W8A8/ngram were not reported. The operator elected not to
retry AWQ's sequential-pipeline OOM; ngram slowed the measured fp16 cell. The
winner exceeds the owner's ~30 sites/s distillation trigger, so no student or
100k training run is needed (owner comment 4938 supersedes the original
20 sites/s/0.6B stretch). 43.8 sites/s is about five times the old 8.8 baseline,
roughly 2.5 GPU hours for 400k sites, excluding loading/transport.

Operator spot-checked **8** fp16/INT8 pairs, not the requested 30; 135/1500
outputs were byte-identical. They saw an occasional invented “based in Italy”
in both precisions. The shipped prompt explicitly addresses that issue; its
quality and live throughput must be rechecked, rather than claiming the old
scores establish the changed prompt's performance.

Exact original command lines, `bench.py`, `quant.py`, Dockerfile, outputs and
logs remain on RTX in `~/emb-test/llm/` (`quantimg/`, `out-*.jsonl`, `run-*.log`).
The comment does not include the original invocations; agents cannot access
RTX by design. Do not label the reproduction commands below as those measured
commands. Operator must attach them and the remaining quality evidence before
final acceptance.

## Quantization and setup (operator, RTX only)

Agents do not deploy, restart or reconfigure production. Reuse the qualified
checkpoint for rollout; rebuilding it is optional. Never access
`~/.cache/huggingface/token`. Scripts use isolated HF_HOME, offline local model
loading and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.

The repository reproduction of the **reported recipe** is
`scripts/summary-quantization/quant.py` and its Dockerfile: llmcompressor 0.14.0,
compressed-tensors 0.20.1 on the local vLLM 0.29 image, SmoothQuant(0.8), GPTQ
W8A8 on Linear, ignoring lm_head. It follows the upstream
[INT8 example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_int8/llama3_example.py).
It is not a copy of inaccessible operator files or a byte-identical checkpoint
reproduction. It uses the tightened production prompt, 512 explicit calibration
records, bf16 loading and context 1024. Regenerated weights need qualification.

Prepare `calibration.jsonl` containing exactly 512 distinct `{id,text}` records
from separate site texts/cards, and `heldout.jsonl` with the evaluation sites
in the same shape. The script rejects overlapping site IDs/texts and overlength
prompts, never trains on evaluation records, refuses an existing output path,
and records the prompt, input hashes and installed versions alongside weights.
No model or data downloads occur. From repository root on RTX:

```bash
docker build -f scripts/summary-quantization/Dockerfile -t ai60-quant-repro .
docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  -v "$HOME/emb-test/llm:/w" ai60-quant-repro \
  --model /w/Qwen3-4B-Instruct-2507 --calibration /w/calibration.jsonl \
  --evaluation /w/heldout.jsonl --output /w/Qwen3-4B-W8A8-reproduced
```

Use `--runtime nvidia`, not `--gpus all` (CDI error on this host). The image puts
the venv on PATH for ninja. Run only with exclusive RTX GPU ownership, preserving
deployed services. Do not overwrite the measured checkpoint. Quantization is
not run or GPU-validated in agent sessions.

For serving, copy `setup-summary.sh` and `simple_ai_summary_provider.py` together
and run on RTX:

```bash
bash scripts/setup-summary.sh /home/lelloman/.simple-ai-summary \
  /home/lelloman/emb-test/llm/Qwen3-4B-W8A8
```

Setup validates the INT8 metadata before installation, installs vLLM 0.29.0 and
compressed-tensors 0.20.1 in a dedicated venv, symlinks the existing checkpoint
to `model/`, and writes pip freeze. It does not touch another engine environment.
Ensure `/home/lelloman/.simple-ai-summary/venv/bin` is on the runner service PATH
for ninja. The runner overlay points at that venv and local checkpoint symlink.

## Lifecycle, compatibility and rollout

The engine advertises the model using existing engine status/model-list fields;
no WebSocket enum/variant or wire version is added. Existing configurations keep
the engine disabled and old runners work unchanged. `class:summarize` resolves
through the gateway class list, not an alias that rewrites the concrete model ID.

The provider is offline and loopback-only. All GPU work stays in its process
(`uni` executor, `VLLM_ENABLE_V1_MULTIPROCESSING=0`), so reaping it releases CUDA
before another engine takes the resource. It admits one HTTP inference batch;
concurrent provider calls get 429. The runner advertises one request slot.
Startup budget is 600 seconds, inference budget 180 seconds. Cancellation retains
resource/lifecycle barriers through detached inference. Unload waits for active
inference; transport/body failures terminate the provider before releasing the
barrier; a crashed provider can reload.

1. After review/merge, operator deploys backend and merges
   `deploy/summaries/backend-overlay.toml` into existing configuration.
2. Prepare the dedicated environment/checkpoint above, copy the provider to
   `/home/lelloman/simple-ai-summary-provider.py`, install the updated **RTX**
   runner, merge `deploy/summaries/rtx.runner-overlay.toml` preserving every
   existing member of `cuda:0`, drain and restart RTX. No Halo binary/config
   change: resource groups are local to a runner, so halo2/class:big slots are
   untouched. Summaries may evict/reload other **RTX** engines.
3. User 2 confirms the live gate through the gateway using the SanCerchino key,
   including the updated prompt and other RTX-engine reloads. Check audit runner
   IDs and GPU logs; HTTP success alone is not proof of placement/coexistence.
   Rollback: drain/disable only summaries, remove its route/class mapping.

Merging only publishes a backend image; agents perform no rollout. Fleet deploy
scripts and host configuration behavior are unchanged.

## Reproduction and live acceptance

For independent offline cells, normalize *copies* to canonical
`{"id":"site-id","text":"card or original profile"}` JSONL with 1500 distinct
aligned IDs. Preserve original assets for `ground.py`. Copy
`benchmark-summaries.py` alongside the provider. Inside the existing RTX vLLM
container, with `/w` mounted and the venv on PATH:

```bash
for weights in Qwen3-4B-Instruct-2507 Qwen3-4B-W8A8; do
  for input in profiles cards; do
    for seqs in 256 512; do
      /app/venv/bin/python /w/benchmark-summaries.py \
        --model-path "/w/$weights" --input "/w/$input.jsonl" \
        --max-num-seqs "$seqs" --output "/w/$weights-$input-$seqs.json"
    done
  done
done
# Diagnostic only, not selected: repeat a cell with --ngram (3 speculative tokens).
```

These reproduce settings with the new prompt, not original operator commands.
The harness records invocation, input SHA256, vLLM version, sites/s, exclusive
GPU wall-seconds/1k, total-device VRAM sampled at 100ms, and ID-aligned outputs.
It warms a synthetic site only. Run `ground.py` using its actual RTX CLI/schema,
compare with fp16 on identical input sites and the same prompt, and inspect
30 fixed-ID summaries for invented places/names. No invented replacement metric
is used. The sequence/ngram switches exist only in benchmark code, not service
configuration.

After rollout, user 2 privately exports `SIMPLE_AI_API_KEY` (SanCerchino key) and
runs against `https://ai.lelloman.com`:

```bash
python3 scripts/check-summaries-live.py --input cards.jsonl \
  --reload-probe rtx-probe.json --output ai60-live.json
```

Probe shape: `{"path":"/v1/embeddings","body":{"model":"Qwen/Qwen3-Embedding-0.6B","input":["Example"]}}`,
or another concrete RTX model on a supported inference endpoint. The harness
warms a synthetic input, times 1000 cards in ordered batches <=64, fails below
15 sites/s, saves all summaries and alternates probe/summary three times for
reload timing. Score saved outputs against same-input fp16 within two points,
inspect 30 outputs, compare other-engine latency/output to pre-rollout evidence,
and confirm via the gateway/audit logs with user 2. **Live throughput, tightened
prompt quality, reload timings and SanCerchino-key confirmation remain pending.**

## Local checks

```bash
export CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads=4
python3 -m unittest discover -s tests -p test_summary_provider.py
bash -n scripts/setup-summary.sh
```

Rust tests cover routing/discovery, bounds, lifecycle cancellation, timeout and
crash/reload. Python fakes check prompt, context, order, usage, busy/error
recovery, selected settings and checkpoint metadata; they establish no GPU speed
or grounding claims. Quantization calibration checks also run without GPU.
