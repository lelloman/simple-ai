#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
expected=sha256:4c0aa3f009b6ae7795bbdaf2b696bfcee7a6cea1e17f1f90b8ab72dea7aa55d8
actual=$(docker image inspect qwen38-27b-3090:latest --format '{{.Id}}')
if [[ "$actual" != "$expected" ]]; then
  echo 'Unexpected base runtime image; re-audit vLLM and the LM-head registration before building.' >&2
  exit 1
fi
docker build -t simple-ai-jev:20260929 .
docker volume create jev-rtx3090_models >/dev/null
docker run --rm --entrypoint /app/venv/bin/python \
  -e HF_HOME=/models/hf -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -v jev-rtx3090_models:/models simple-ai-jev:20260929 -c '
from huggingface_hub import snapshot_download
snapshot_download("autotrust/JEV-9B", revision="4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35",
    allow_patterns=["*.json", "*.safetensors", "*.jinja"],
    ignore_patterns=["reports/*", "adapter/*"], max_workers=4)
'
docker compose config --quiet
