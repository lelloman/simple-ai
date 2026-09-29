# Semantic decisions with JEV-9B

In the web interface, open **Decisions**, enter text or choose JSON, and add
yes/no, choice, or rating questions. Click **Load example** for a ready-to-run
request, then **Evaluate**. Results show answers and probability bars; **Copy
JSON** copies the full response. Your existing login is used.

For API use, call the normal simple-ai gateway with your API key. JEV supports boolean decisions,
choices among 2–16 named options, and ratings on an explicit six-level scale.
The gateway wakes the RTX runner if needed. The runner unloads its other GPU model,
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
prefer `gpu-server` for that class. The RTX configuration in
`scripts/configs/rtx.toml` enables `[engines.decisions]` and places it in the same
`cuda:0` resource group as the other GPU engines. Other runners leave it disabled.

The runtime must first be prepared using `deploy/semantic-rtx3090/prepare.sh`.
The fleet deployer uploads the compose file and provider and runs an offline
preflight against the existing model volume. It does not download another copy
of the weights. Startup verifies the provider protocol, model revision, and
vLLM base/adapter identity. Both provider and vLLM ports remain loopback-only.

For rollback, disable `[engines.decisions]`, stop its compose services after
active requests drain, and remove the gateway class mapping. Keep the shared
model volume unless you explicitly intend to remove the downloaded weights.
