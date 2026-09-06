#!/usr/bin/env bash
set -euo pipefail
cd /home/lelloman/glm45v-test
podman run --rm --name glm45v-bench \
  --device /dev/dri --group-add keep-groups --security-opt label=disable \
  --volume /home/lelloman/glm45v-test/models:/models:ro \
  localhost/strix-qwen38-mtp:463c0f6 --bench \
  -m /models/GLM-4.5V-Q4_K_M-00001-of-00002.gguf \
  -dev Vulkan0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  -b 512 -ub 512 -t 16 -p 512 -n 128 -r 3 -o json \
  >results/llama-bench.json 2>results/llama-bench.log
cat results/llama-bench.json
