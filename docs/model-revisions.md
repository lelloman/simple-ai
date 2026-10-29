# API: model and runner provenance (AI-62)

Successful gateway-routed responses from `POST /v1/chat/completions`,
`POST /v1/embeddings`, and `POST /v1/summaries` carry `X-Runner-Id` and, when
reported by the selected runner, `X-Model-Revision`. Both headers are exposed by
CORS. The revision belongs to the resolved concrete model, including when the
request uses a class or alias. It is captured from the reserved runner's status,
then retained through batching and response streaming.

Non-streaming chat JSON additionally contains `model_revision` when known:

```json
{"model":"qwen3.8-flash-next-uncensored-iq4xs","model_revision":"<weight SHA-256>","choices":[]}
```

This is a partial example. Streaming chat adds the headers only; SSE chunks and
embedding/summary body schemas are unchanged. Errors before dispatch have no
model provenance. An older runner has no revision, so the revision header and
body field are omitted; clients must accept absence. Legacy direct Ollama mode
has no runner metadata and omits these fields. Do not invent a revision from the
model name or a gateway build version.

Revision strings are opaque identities: GGUF files use lowercase SHA-256;
Hugging Face snapshots use their commit hash; local checkpoint directories use a
deterministic SHA-256 over sorted relative filenames and file content hashes;
Ollama uses its digest (including any digest prefix). These identify weights,
not serving parameters, prompts, software versions, or hardware. Keep weights
immutable while loaded. Reload after replacing a checkpoint.

## Operator rollout and acceptance

Agents do not deploy. User 2 deploys the backend first (compatible with older
runners), then updated runners, including the Halo serving `class:big`. For
Halogen and Docker-managed vLLM, set `model_path` to the **runner host's** path
to the actual served GGUF/checkpoint, not the container mount path. The new
field is optional so existing configurations still start. See the runner
[README](../inference-runner/README.md). No gateway configuration change is needed.
For production Halogen the example configs name the existing GGUF under
`/home/lelloman/halogen-benchmark/models/`; verify it matches the service mount.
For vLLM the operator must map each profile's actual `MODEL` to its host path.
The file directory should be writable by the runner for its SHA-256 sidecar;
if read-only, hashing still works but the sidecar cannot be written. Each actual
load recomputes the digest; the sidecar is an inspectable record, not a trusted
cache input.

After rollout, privately export the existing API credential and run:

```bash
curl -sS -D - "$SIMPLE_AI_URL/v1/chat/completions" \
  -H "Authorization: Bearer $SIMPLE_AI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"class:big","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8}'
```

Confirm both headers and `model_revision` in the JSON, with the same revision.
Repeat with `"stream":true` to check headers without changing SSE, then check
`/v1/embeddings` and `/v1/summaries` against their configured models. Compare the
GGUF identity with `sha256sum` on the serving Halo. This live check requires
operator rollout and is not performed by agents.
