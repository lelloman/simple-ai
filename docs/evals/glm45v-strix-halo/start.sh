#!/usr/bin/env bash
set -euo pipefail
cd /home/lelloman/glm45v-test
podman run -d --rm --name glm45v-test --network host \
  --device /dev/dri --group-add keep-groups --security-opt label=disable \
  --volume /home/lelloman/glm45v-test/models:/models:ro \
  localhost/strix-qwen38-mtp:463c0f6 --server \
  -m /models/GLM-4.5V-Q4_K_M-00001-of-00002.gguf \
  --mmproj /models/mmproj-GLM-4.5V-Q8_0.gguf \
  --alias glm-4.5v --host 127.0.0.1 --port 18045 \
  --device Vulkan0 -ngl 999 --fit off -c 8192 -np 1 \
  --flash-attn on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -t 16 \
  --jinja --reasoning off --metrics
