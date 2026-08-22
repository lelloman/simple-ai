#!/usr/bin/env bash
set -euo pipefail

DEST=${1:-/home/lelloman/qwen38-serving}
HERE=$(cd "$(dirname "$0")" && pwd)
PROFILE=$DEST/profiles/fast.env
RESULTS=$DEST/acceptance

driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
if [ "$driver_major" -lt 580 ]; then
  echo "NVIDIA driver 580 or newer is required" >&2
  exit 1
fi
if [ ! -s "$DEST/models/qwen38-uncensored-fast.sha256" ]; then
  echo "run calibrate-fast.sh first" >&2
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

"${compose[@]}" exec -T single bash verify.sh
VLLM_API_KEY=$api_key PORT=18020 python3 "$HERE/acceptance.py" | tee "$RESULTS/protocol.txt"

for run in 1 2; do
  "${compose[@]}" exec -T single bash bench/run_benchmarks.sh single \
    | tee "$RESULTS/benchmark-run-$run.txt"
done

greedy_decode=$(awk '
  /ROW cohort C1 real prompts T=0 / {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^decode\(C\/meanTPOT\)=/) { split($i, value, "="); print value[2] }
    }
  }
' "$RESULTS/benchmark-run-2.txt" | tail -n1)

if [ -z "$greedy_decode" ]; then
  echo "could not parse the second-run C1 greedy decode rate" >&2
  exit 1
fi
awk -v rate="$greedy_decode" 'BEGIN { exit !(rate >= 100.0) }' || {
  echo "C1 greedy decode gate failed: $greedy_decode tok/s < 100 tok/s" >&2
  exit 1
}

printf 'PASS optimized uncensored profile: C1 greedy decode=%s tok/s\n' "$greedy_decode" \
  | tee "$RESULTS/summary.txt"
