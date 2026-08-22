#!/usr/bin/env bash
set -euo pipefail

DEST=${1:-/home/lelloman/qwen38-serving}
HERE=$(cd "$(dirname "$0")" && pwd)
PROFILE=$DEST/profiles/base.env
RESULTS=$DEST/acceptance-base

driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
if [ "$driver_major" -lt 580 ]; then
  echo "NVIDIA driver 580 or newer is required" >&2
  exit 1
fi
if [ ! -s "$DEST/models/Qwen3.8-27B-abliterated-AWQ-MTP/model.safetensors.index.json" ]; then
  echo "run prepare-host.sh first" >&2
  exit 1
fi

mkdir -p "$RESULTS"
compose=(docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" --env-file "$PROFILE")
"${compose[@]}" --profile single up -d --force-recreate single

api_key=$(tr -d '\r\n' < "$DEST/api_key.txt")
ready=0
for _ in $(seq 1 600); do
  if curl -sf -H "Authorization: Bearer $api_key" "http://127.0.0.1:18020/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  "${compose[@]}" logs --tail 200 single >&2
  echo "vLLM did not become healthy within 20 minutes" >&2
  exit 1
fi

"${compose[@]}" exec -T single bash verify.sh --no-server
VLLM_API_KEY=$api_key PORT=18020 TEXT_ONLY=1 python3 "$HERE/acceptance.py" \
  | tee "$RESULTS/protocol.txt"
python3 "$HERE/quick-benchmark.py" --api-key "$api_key" --runs 10 \
  | tee "$RESULTS/benchmark.txt"

median_decode=$(awk '/^summary: median decode / { print $4 }' "$RESULTS/benchmark.txt" | tail -n1)
if [ -z "$median_decode" ]; then
  echo "could not parse median decode rate" >&2
  exit 1
fi
awk -v rate="$median_decode" 'BEGIN { exit !(rate >= 100.0) }' || {
  echo "C1 decode gate failed: $median_decode tok/s < 100 tok/s" >&2
  exit 1
}
printf 'PASS prepared uncensored base profile: median C1 decode=%s tok/s\n' "$median_decode" \
  | tee "$RESULTS/summary.txt"
