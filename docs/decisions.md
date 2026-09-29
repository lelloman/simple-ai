# Semantic decisions with JEV-9B

In the web interface, open **Decisions**, enter text or choose JSON, and add
yes/no, choice, or rating questions. Click **Load example** for a ready-to-run
request, then **Evaluate**. Results show answers and probability bars; **Copy
JSON** copies the full response. Your existing login is used.

For API use, call the normal simple-ai gateway with your API key. JEV supports boolean decisions,
choices among 2–16 named options, and ratings on an explicit six-level scale.
The gateway prepares RTX and one available Halo when needed. The first JEV
model ready serves the request; subsequent requests prefer RTX once ready. The runner unloads its other GPU model,
starts JEV, and shares the GPU through the normal resource allocator. The first
request can take several minutes; warm requests are much faster.

```bash
curl --max-time 1200 "$SIMPLE_AI_URL/v1/decisions" \
  -H "Authorization: Bearer $SIMPLE_AI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "class:semantic_decisions",
    "state": {"message": "I was charged twice. Please refund the duplicate."},
    "questions": [
      {"id":"refund", "type":"boolean", "instruction":"Does the customer request a refund?"},
      {"id":"topic", "type":"choice", "instruction":"Which team should handle this?",
       "options":[{"id":"billing","description":"Payments, charges, refunds"},
                  {"id":"technical","description":"Technical problems using the product"}]},
      {"id":"urgency", "type":"rating", "instruction":"Rate the urgency of this request.",
       "levels":["No action needed","Very low","Low","Normal","High","Critical"]}
    ]
  }'
```

`state` accepts a string, object, or array. Questions are returned in input order.
Boolean answers contain `value` and probabilities for `false` and `true`.
Choice answers contain `selected` and probabilities keyed by your option IDs.
Rating answers contain the expected value from 0 to 5, probabilities for all six
levels, and your level descriptions. These are model probabilities, not validated
confidence guarantees: `calibrated` is false. Published upstream temperature
scaling is applied and reported separately.

Responses also identify the concrete model and pinned revision, report token
usage (including cached tokens when available), and contain timing fields.
`class:semantic_decisions` follows ordinary class permissions; selecting
`autotrust/JEV-9B` directly requires the existing `model:specific` role.

Limits: 2 MiB request body, 1–128 questions, unique ASCII IDs (letters, digits,
underscore or hyphen; 1–64 characters), instructions/descriptions up to 8,192
characters. Each question plus state must fit 18,432 tokens; input is never
silently truncated. Ratings require exactly six descriptions. The runner admits
one active request and at most 32 waiting requests, with a 120-second admission
wait. HTTP 429 means retry later; 400 indicates invalid input or excess context;
413 indicates an oversized body; 502/503/504 indicate provider, availability, or
timeout failures.

The existing `/v1/classifications` NLI endpoint is separate and unchanged.
A subsequent chat request switches the GPU back to its requested chat model.
There is no need to start or stop Docker containers manually.

## Deployment

Deploy the gateway first, then the RTX runner. Set
`models.semantic_decisions = ["autotrust/JEV-9B"]` in gateway configuration and
set `routing.decision_ready_race = true`. Configure both
`routing.class_preferences.semantic_decisions` and
`routing.speculative_wake_targets.semantic_decisions` as `["gpu-server", "halo"]`,
and allow at least 600 seconds for `routing.model_prepare_timeout_secs`. The RTX configuration in
`scripts/configs/rtx.toml` enables `[engines.decisions]` and places it in the same
`cuda:0` resource group as the other GPU engines. Halo 1 and Halo 2 use
`deploy/semantic-halo` through Podman Compose and share `gpu:0` with llama.cpp.

Prepare RTX using `deploy/semantic-rtx3090/prepare.sh`; prepare each Halo
following `deploy/semantic-halo/README.md`.
As of 2026-09-29, RTX, Halo 1 and Halo 2 are deployed. Halo 2 required a kernel
update to 7.2.7-100.fc43 before its ROCm runtime passed GPU inference validation.
The fleet deployer uploads the compose file and provider and runs an offline
preflight against the existing model volume. It does not download another copy
of the weights. Startup verifies the provider protocol, model revision, and
vLLM base/adapter identity. Both provider and vLLM ports remain loopback-only.

For rollback, disable `[engines.decisions]`, stop its compose services after
active requests drain, and remove the gateway class mapping. Keep the shared
model volume unless you explicitly intend to remove the downloaded weights.

## First-ready scheduling

For JEV, the readiness race chooses one runner per configured machine type.
An already-loaded/connected Halo is preferred over waking another Halo, with
active-request count breaking ties. If RTX is already ready, requests go
straight to it. Otherwise RTX and the selected Halo are prepared in parallel;
the gateway dispatches inference only once, to the first ready model. RTX
preparation continues after a Halo wins so subsequent requests can use RTX.
Concurrent requests share the same preparation task. Both class and concrete
JEV selectors use this policy. A failed preparation does not block a ready
runner, and preparation has bounded wake/load deadlines with idle-manager
keepalives. The other Halo remains available for future selection.

This is separate from batching independent contexts: the API still accepts
one state and up to 128 questions per request.
