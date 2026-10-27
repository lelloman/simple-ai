# Batched translation

`POST /v1/translations` uses the same authentication, model permissions, audit,
scheduling, wake and routing as classifications. The model is
`google/madlad400-3b-mt`, Apache 2.0. It is advertised in `/v1/models` by the
configured runner, including before loading. No chat reasoning is involved.

```json
{"model":"class:translate","input":["Il parco si trova vicino al centro."],"target":"en","source":"it","max_tokens":256}
```

```json
{"data":[{"index":0,"translation":"The park is located near the centre."}],"model":"google/madlad400-3b-mt"}
```

This response is illustrative, not a measured model output. `input` must be an
array of 1–256 non-blank texts, at most 4,000 Unicode characters each and 100,000
total. `max_tokens` defaults to 256 output tokens per text, allowed 1–512.
Reaching the output limit can truncate a translation: increase it if necessary.
The provider rejects input above 1,024 model tokens rather than truncating it.
Provider timeout is 180 seconds. Use short batches (initially 32 descriptions).
The provider runs one request at a time and returns 429 when busy; it batches
within that request and preserves input order across batches. There is no
cross-request batching or new queue.

`target` accepts MADLAD's language codes and their ISO 639-3 equivalents (for
example `en`/`eng`, `de`/`deu`). `zh-Hans` is accepted as an alias for `zh`.
MADLAD has one `zh` target; this alias does not add script conversion or guarantee
Simplified Chinese output. Codes are case-sensitive. The fixed mapping is in
`simple-ai-common/src/translation-languages.json`, derived from the
[model card](https://huggingface.co/google/madlad400-3b-mt) and ISO 639 aliases.
Unknown target/source codes return 400 before routing. `source` is optional,
validated but not sent to the tokenizer: MADLAD detects source language itself.
The internal prompt is `<2target> text`. Source language hints do not change it.

## Operator rollout (not performed by agents)

Choose **Halo 2**, which the current host-state runbook reserves for SimpleAI
while Halo 1 serves agents. This is a placement choice, not a measurement of live
free RAM. Confirm actual memory headroom (roughly 12 GB during FP32 model load,
6 GB weights in BF16 on GPU, plus generation buffers) and ROCm support first.
Do not change host ownership just to run this model. GPU translation joins the
existing `cuda:0` resource group; it cannot run simultaneously with another
model in that group. Existing GPU model unload/reload latency must be included
in the live coexistence measurements; no non-degradation claim is made here.
CPU mode is available for hosts without a GPU but is not throughput-qualified.
Enabling `auto` or `cuda` requires `engine_resources.translation`; runner startup
rejects the configuration if that resource group is missing.

1. Deploy the backend first with `[models] translate = ["google/madlad400-3b-mt"]`.
   Add `translate = ["halo"]` under existing `[routing.class_preferences]` and
   `[routing.speculative_wake_targets]` tables if speculative waking is used.
   Existing runners need no upgrade and the protocol version is unchanged:
   `Capability::Translation` already exists.
2. On Halo 2 prepare an isolated Python venv using the host-qualified ROCm torch
   wheel. Run `scripts/setup-translation.sh /home/lelloman/.simple-ai-translation/venv/bin/python /home/lelloman/.simple-ai-translation/model`.
   This downloads pinned revision `fa184c675da0b5c9e1c8694fccd4e12e2d422094`.
   Do not replace another provider's environment. The runtime is offline.
3. Merge `scripts/configs/halo-02-translation.example.toml` into Halo 2's existing
   fleet config. Confirm every GPU engine uses the same resource group string.
   `scripts/deploy-fleet.sh` now copies the provider and language map with the
   runner binary. The operator deploys that updated runner/config to Halo 2 only.
   No other runner needs a configuration change. For rollback, disable the
   translation engine before removing the backend route/config.
4. Request a SanCerchino-style client key from operator user 2 if necessary.
   Run the live acceptance check below through `https://ai.lelloman.com`.

## Checks

Local: required workspace fmt/clippy/tests plus
`python3 -m unittest discover -s tests -p test_translation_provider.py`.
Lifecycle tests use a tiny process fixture, not the model weights.

The operator can collect timings and samples with:

```bash
SIMPLE_AI_API_KEY=... python3 scripts/check-translations-live.py \
  --input descriptions-it.json --embedding-probe embedding-request.json \
  --classification-probe classification-request.json --output translation-results.json
```

Both probe files contain normal API request bodies. Verify their runner IDs in
the audit log; choose model-specific IDs/permissions if needed to ensure Halo 2.
The harness records three probes before/after and periodic probes during, and
fails on translation errors, invalid order, or missing/empty outputs. Inspect
probe latency/throughput and any probe errors rather than treating a successful
translation run as proof of non-degradation. No credentials enter the report.

Production acceptance remains required after rollout: 150 Italian site
descriptions, each ≤400 characters, translated into en/de/fr/es/ar/zh-Hans/ru
(1,050 outputs). Record total wall time and sentences/s at batch size 32,
a sample per language, HTTP status and timeouts. Use the same 150 descriptions
for each language. Time existing embeddings and classification on Halo 2
before, during and after translation, with identical payloads, repetitions and
routing; record model IDs, runner IDs and unload/reload effects. A synthetic
smoke test alone does not prove the hundreds-of-sentences/s goal. Do not close
the live acceptance item without those results. Agents do not deploy or restart
production services.
