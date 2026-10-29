# Inference runner

The runner manages local engines and reports discovery/status over WebSocket.
See [configuration](config.example.toml), [architecture](PHASE4.md), and the
[API provenance contract and operator rollout](../docs/model-revisions.md).

## Model revisions

`EngineStatus.available_models[].revision` is an optional string next to `id`.
The wire version is unchanged. New gateways accept old runners without it;
old gateways ignore the additional property. Status polling reads stored
identities and does not hash model weights on every request.

* llama.cpp computes the GGUF SHA-256 before starting the model server. A
  `<model-file>.simple-ai-sha256` sidecar records the digest for inspection.
  Every actual load hashes again: metadata alone cannot detect same-size writes
  on filesystems with coarse timestamps. Writes replace the sidecar atomically.
  Read-only directories warn and retain the identity in memory for this load.
* Halogen uses the same GGUF hashing and sidecar. Set `engines.halogen.model_path`
  to the host-visible GGUF used by its systemd service. The runner identifies
  an already-running service when it adopts it as well.
* vLLM uses `engines.vllm.models.<id>.model_path`, a host-visible snapshot or
  checkpoint directory matching that Compose profile. Paths are optional for
  compatibility; missing paths mean no revision. They are needed because the
  runner cannot infer host paths from arbitrary container mounts.
* Managed HF providers identify local `model_path` overrides at load. Canonical
  HF snapshot directories report their commit; other checkpoint directories
  hash relative names and content in sorted order. Providers with pinned HF
  revisions report the existing pin when no local override is used. Unpinned
  HF providers resolve the existing offline cache's `refs/main` at load using
  `HF_HUB_CACHE`, `HF_HOME`, or the default user cache. Ollama reports the
  digest from its model discovery API.

A hashing failure logs the model and error and leaves revision absent; inference
can still load. No guessed revision is returned. Keep model directories immutable
while a model is loaded, and reload after replacing files. First-time hashing can
extend cold start time. Model names and revision strings are independent: aliases
retain the selected concrete model's revision. No inference request reads or
hashes weight files for this metadata.
