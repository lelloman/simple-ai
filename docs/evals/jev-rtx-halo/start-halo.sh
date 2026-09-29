#!/bin/sh
set -eu
cd /home/lelloman/simple-ai-jev-bench
podman run -d --name jev-bench-halo \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable --shm-size 8g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HOME=/models/hf -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_SERVER_DEV_MODE=1 -e OMP_NUM_THREADS=8 \
  -v "$PWD/hf:/models/hf:ro" -v "$PWD/scripts:/work/scripts:ro" \
  -v "$PWD/tests:/work/tests:ro" -v "$PWD/results:/results" \
  -v "$PWD/enable-halo-head-lora.py:/work/enable-head-lora.py:ro" \
  --entrypoint sh \
  docker.io/rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  -c 'python /work/enable-head-lora.py && exec vllm serve \
  /models/hf/hub/models--autotrust--JEV-9B/snapshots/4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35 \
  --served-model-name autotrust/JEV-9B --host 0.0.0.0 --port 30000 \
  --dtype bfloat16 --enable-lora --max-lora-rank 32 \
  --lora-modules jev-decision=/models/hf/hub/models--autotrust--JEV-9B/snapshots/4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35/adapter_vllm \
  --logprobs-mode processed_logprobs --max-model-len 18432 \
  --kv-cache-memory-bytes 4241280204 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --enable-prompt-tokens-details \
  --enable-prefix-caching --mamba-cache-mode align --enforce-eager'
