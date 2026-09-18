# GLiNER2.5 multilingual API evaluation

Date: 2026-09-18. Status: initial feasibility review; no inference measurements or production changes.

Recommendation: evaluate `fastino/gliner2.5-multi-v1` as a new information-extraction capability. Start with entities unless a consuming application's requirements favor records or classification. The integration has an existing architectural precedent; quality, latency, and resource requirements remain unverified.

## Candidate and upstream evidence

The [model card](https://huggingface.co/fastino/gliner2.5-multi-v1) identifies a 287M-parameter multilingual mDeBERTa-based boundary model under Apache-2.0. It supports entities, classification, records, relations, and span attributes. Use `AutoExtractor.from_pretrained`, not the legacy `GLiNER2` loader shown in the generated Hugging Face usage snippet. This is Fastino's GLiNER2.5 checkpoint, distinct from the similarly named `gliner-community/*-v2.5` family.

The card reports `max_len=4096`; verify effective text capacity with schema overhead. Normal extraction can truncate; long-document helpers chunk inputs. Mentions and relation endpoints must coexist in a chunk. The card's weight-size and dtype descriptions are not a runtime memory measurement.

The [package metadata](https://github.com/fastino-ai/GLiNER2/blob/main/pyproject.toml) requires Python >=3.10. Local inference uses `gliner2[local]`, including PyTorch and Transformers <5. Pin the tested package, dependencies, and model revision in an isolated environment before benchmarking.

The [upstream README](https://github.com/fastino-ai/GLiNER2) documents batching, CPU/GPU execution, confidence scores, character spans, and configurable word splitting. Its default splitter follows whitespace; changing splitting for languages such as Chinese can affect accuracy and must be evaluated explicitly.

## Fit with SimpleAI

The current process-backed NLI implementation is a useful integration template:

- `scripts/simple_ai_classification_provider.py`: persistent Python model process with a loopback HTTP interface and serialized inference.
- `inference-runner/src/engine/classification.rs`: process startup, health checks, model loading, capacity management, and shutdown.
- `backend/src/routes/classifications.rs`: authentication, validation, audit records, and scheduled routing.
- `backend/src/gateway/router.rs`: runner selection, model aliases, forwarding, and circuit-breaker accounting.

GLiNER cannot satisfy the existing classification response contract: `simple-ai-common/src/classification.rs` promises entailment, neutral, and contradiction probabilities for every text/hypothesis pair. GLiNER label confidence is a different output. Do not synthesize those probabilities or route the model under `class:text_classification`.

Proposed first capability: `information_extraction`, exposed through `POST /v1/extractions`. These names are proposals, not available endpoints. Keep the public schema owned by SimpleAI so an upstream Python API change does not alter the HTTP contract.

An entities-only first request could be:

```json
{
  "model": "class:information_extraction",
  "input": ["Giulia lavora per Acme a Milano."],
  "schema": {
    "entities": [
      {"label": "person", "description": "Named people"},
      {"label": "organization", "description": "Companies and organizations"},
      {"label": "location", "description": "Named places"}
    ]
  },
  "threshold": 0.5
}
```

Return input-indexed results, resolved model identity, and entity objects containing stable label, extracted text, start, end, and score. Define offsets as Unicode code points with an exclusive end; test Rust and Kotlin consumers explicitly. Scores are model confidence, not calibrated guarantees. Treat 0.5 as an initial evaluation setting, not a validated production threshold.

Specify overlap behavior, ordering, missing entities, and validation errors. Initially reject oversized inputs explicitly; introduce opt-in chunking only after testing global offsets and boundary recall. Bound input count, encoded length, schema size, queued work, and returned entities. Account for both text and schema processing in usage rather than reporting generated tokens.

A production implementation would touch shared request/response and capability types, runner engine/configuration/API/registration, and gateway route/configuration/model-class/scheduler/telemetry/wake-on-LAN paths. Reuse lifecycle and routing patterns while checking active-request protection during unload. Include model discovery, permissions, timeout behavior, and rollout compatibility for older runners.

## First benchmark

Use a standalone evaluation before integrating the gateway. Default initial languages: Italian and English, with a smaller stress set in another Latin-script language and a language without whitespace word boundaries. Confirm actual customer languages before advertising coverage.

Create a human-reviewed corpus of roughly 200 representative documents for the first pass. Separate threshold development from held-out evaluation; report counts and per-language/per-entity results, including uncertainty from small samples. Include absent entities, ambiguous labels, repeated mentions, accents, emoji, nested entities, mixed-language text, noisy/OCR text, and chunk boundaries. Synthetic fixtures test mechanics but cannot establish production quality.

| Dimension | Proposed measurement |
| --- | --- |
| Entity quality | Exact span-and-label precision, recall, F1; false positives on negative documents |
| Schema sensitivity | Short labels versus descriptions; English versus localized descriptions |
| Runtime | Cold load, first request, warm p50/p95, documents/second; batches 1/8/32 |
| Workload scaling | Short/medium/near-limit texts; 4/16/32 labels; concurrency 1/4 |
| Resources | Peak process RAM and GPU memory, idle footprint, unload recovery |
| Robustness | No silent truncation; offsets recover exact source substrings; repeatability |
| Comparison | Same labeled corpus through the current chat extraction route, fixed prompt/model/settings |

Run CPU first and an available CUDA runner second. Record hardware, thread counts, dtype, package lock, model revision, warmup, and raw timings. Benchmark under representative co-resident load before choosing placement. CPU sufficiency, low GPU cost, and speed advantages are hypotheses until measured.

For a classification-focused follow-up, compare GLiNER and existing NLI predictions using task-level accuracy/F1 and calibration on identical labels; their raw score vectors are not comparable. Records and relations require separate annotations and metrics, including correct association between instances.

## Decision and remaining work

Proceed to an experimental provider if held-out quality meets the first consumer's requirements and measured latency/memory fit an identified runner. Agree quality and latency targets before examining held-out results. At minimum, require exact Unicode span reconstruction, deterministic request validation, explicit overflow handling, and successful load/unload/reload without leaked workers.

Publish only the tasks and languages supported by evaluation. A general multilingual claim or an upstream example is not sufficient evidence. If entities succeed but records fail, scope the offering to entities. If CPU is too slow, evaluate GPU placement using measured contention costs.

Next concrete deliverable: a pinned standalone benchmark, reviewed input/annotation corpus, raw outputs and timings, and a go/no-go report. This document records feasibility and proposed work only; no model has been downloaded or executed as part of this review.
