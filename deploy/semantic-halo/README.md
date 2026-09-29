# JEV on Strix Halo

This deployment uses the same pinned JEV revision and BF16 weights as RTX.
The AMD gfx1151 image is pinned by digest. `enable-text-model.py` registers
standalone Qwen3.5 text serving in that image using the compatibility changes
validated in `docs/evals/jev-rtx-halo`. It does not alter checkpoint files.
The production runtime does not enable vLLM development/cache-reset endpoints.

The host needs rootless Podman with the Compose provider, crun, GPU device
access and a prepared Hugging Face cache at
`~/simple-ai-semantic/models/hf`. Preserve the cache's `hub/blobs` and snapshot
symlinks when copying it; copy the complete `models--autotrust--JEV-9B` tree,
including `adapter_vllm`, tokenizer files and calibration metadata.

From this directory on the host:

```sh
systemctl --user start podman.socket
podman pull docker.io/rocm/vllm@sha256:394194d36edcf9b36bcb563e143b21b80e64e7d04f33a447b448c0c0c00c04a8
podman compose run --rm --no-deps --pull never provider --preflight
```

The crun `run.oci.keep_original_groups` annotation preserves GPU access through
the Docker-compatible Compose API. `group_add: [keep-groups]` only works with
native Podman and must not be used with that API. SELinux label isolation is
disabled for these containers so their read-only host mounts remain accessible.
The model cache is read-only; provider and vLLM host ports bind to loopback.

`deploy-fleet.sh` installs the compose file, compatibility patch and provider
on Halo runners and runs the offline preflight. It does not download weights
or start inference. The runner starts/stops the composite service on demand;
JEV and llama.cpp share `gpu:0`, while CPU extraction/audio remain independent.

Merge these settings into the host's runner configuration (keep its existing
gateway credentials and other engine settings):

```toml
[engine_resources]
decisions = "gpu:0"
llama_cpp = "gpu:0"

[engines.decisions]
enabled = true
compose_command = ["podman", "compose"]
compose_dir = "/home/lelloman/simple-ai-semantic/deploy/semantic-halo"
base_url = "http://127.0.0.1:18040"
startup_timeout_secs = 600
request_timeout_secs = 180
shutdown_timeout_secs = 60
```

Validate GPU inference before enabling the engine. Offline preflight validates
the checkpoint and configuration, but cannot detect host GPU runtime failures.
On 2026-09-29 this runtime worked on Halo 1 with kernel 7.1.13-200.fc44;
Halo 2 with kernel 6.17.1-300.fc43 crashed in ROCm `hipMemcpyWithStream`, even
for a one-element GPU tensor. After updating to 7.2.7-100.fc43, both the tensor
test and a real JEV decision succeeded; its JEV engine is now enabled.
