# Local JEV-9B semantic decisions

JEV-9B is integrated into the gateway and runner as `POST /v1/decisions`.
See [the gateway API guide](decisions.md) for normal application use.
The loopback provider also retains the three experimental routes documented
below for local diagnostics and benchmarks. The `/v1/classifications` NLI
contract remains separate.
See [deployment](../deploy/semantic-rtx3090/README.md) and
[JEV RTX 3090 validation/results](evals/jev-rtx3090/README.md). The original Qwen
[RTX 3090 measurements](evals/semantic-rtx3090/README.md) are historical, not JEV results.

## API

| Route | Required JSON fields | Result |
| --- | --- | --- |
| `POST /v1/semantic/score` | `state`, `predicate` | `score`, `scores`, `selected` |
| `POST /v1/semantic/score-many` | `state`, `predicates` | ordered `results` |
| `POST /v1/semantic/classify` | `state`, `options` | `scores`, `selected` |
| `GET /health` | none | backend readiness, model/revision and configuration |

```bash
curl http://127.0.0.1:18040/v1/semantic/score-many \
  -H 'Content-Type: application/json' \
  -d '{"state":"Build failed: DNS lookup failed before compilation.","predicates":["The failure is environmental.","The code contains a confirmed bug."]}'

curl http://127.0.0.1:18040/v1/semantic/classify \
  -H 'Content-Type: application/json' \
  -d '{"state":"I received the wrong shoes and want a refund.","options":["billing","technical_support","returns","security"]}'
```

State accepts a nonempty string, object, or array. JSON metadata is preserved.
Limits: 2 MiB body, 1–128 predicates, **2–16 unique classification options**,
8,192 characters per predicate/option, 4,194,304 submitted input tokens across
branches. Each full prompt must fit 18,432 tokens including the one readout
position. Overlong input is rejected without truncation. Only one evaluation is
admitted at a time; overlapping evaluations receive HTTP 429.

## Decision implementation and score semantics

The checkpoint, tokenizer and decision metadata are pinned to revision
`4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35`. Startup checks the backend's
model and adapter paths against that revision and checks all 24 verbalizer IDs.
The provider renders the upstream `bare-v1` template exactly, tokenizing it as
one string. Binary questions use the trained `false`, `true` order. Classification
uses the trained `A`–`P` slots. Every request selects the **jev-decision adapter**,
which includes both the backbone LoRA and trained output head.

vLLM returns one position's log probabilities restricted to the permitted token
IDs (`processed_logprobs`). The provider adds the published slot bias, divides
by the per-kind temperature, and applies softmax. It never parses generated text.
The API's `score` means P(True), not JEV's separate ordinal 0–5 score primitive;
that ordinal primitive is not exposed by these existing routes.

Compatibility changes from the retired experiment:

- Classification supports 16 options, previously 64.
- `raw_logprobs` now means restricted vLLM verbalizer log probabilities before
  the published bias/temperature correction, not full-vocabulary probabilities.
- `label_probability_mass` is removed because restricted probabilities cannot
  measure full-vocabulary answer mass.
- No prefix-only warmup request: each question takes one readout. Prefix reuse
  is automatic and opportunistic. Unknown cache telemetry remains null.
- `upstream_temperature_applied: true` records the published adjustment.
  `calibrated: false` remains because calibration on our application data has
  not been established. Neither field promises correctness.

Include an `insufficient_information` category when an application needs
abstention. Boolean decisions remain forced choices even if evidence is missing.

## Validation and evaluation

```bash
python3 -m unittest discover -s tests -p test_semantic.py
python3 scripts/check-semantic-tokenizer.py
python3 scripts/evaluate-semantic.py --output jev-quality.json
python3 scripts/benchmark-semantic.py --exclusive-backend \
  --lengths 500 2000 --batches 1 8 --repetitions 5 --output jev-benchmark.json
```

The tokenizer/evaluation commands need `scripts/semantic-requirements.txt`;
use the deployment container to reuse its installed dependencies and snapshot.
Run benchmarks on the GPU host so memory measurements correspond to the backend.
The benchmark resets vLLM's prefix cache and must use an exclusive backend.
Enable its administrative route only for benchmarks by starting Compose with
`JEV_BENCHMARK_MODE=1 docker compose up -d`. Restart with the default (0)
afterward to disable development routes.
The shared mode submits bounded concurrent branches; independent mode clears
cache between sequential questions. `warmup_calls` is always zero; the benchmark
still warms GPU kernels before measured trials. `prefix_tokens` describes the
potential common prefix, not measured saved computation.

The existing 28-case synthetic quality fixture includes 24 labelled and four
ambiguous examples. Its metrics can detect regressions but cannot establish
production accuracy or calibration. Preserve per-example predictions and
ambiguous-case confidence. Historical Qwen data is retained for comparison.
