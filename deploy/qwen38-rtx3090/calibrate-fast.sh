#!/usr/bin/env bash
set -euo pipefail

DEST=${1:-/home/lelloman/qwen38-serving}
BASE=/app/models/Qwen3.8-27B-abliterated-AWQ-MTP
FAST=${BASE}-fast

if [ ! -s "$DEST/api_key.txt" ]; then
  echo "run prepare-host.sh first" >&2
  exit 1
fi
if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)" -lt 580 ]; then
  echo "NVIDIA driver 580 or newer is required" >&2
  exit 1
fi
busy_mib=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
  | awk '{ total += $1 } END { print total + 0 }')
if [ "$busy_mib" -gt 1024 ]; then
  echo "GPU is already using ${busy_mib} MiB; stop the live Simple-AI runner before calibration:" >&2
  echo "  systemctl --user stop simple-ai-runner.service" >&2
  exit 1
fi

docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" \
  --env-file "$DEST/profiles/fast.env" run --rm \
  -e MODEL="$BASE" -e FAST_VARIANT=0 single bash -lc '
    set -euo pipefail
    cd /app
    V=venv/bin/python
    $V drafter/collect_prompts.py
    VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_MARLIN_INT8_INCLUDE_RE=mlp \
      VLLM_MARLIN_INT8_EXCLUDE_RE="mtp|draft" $V drafter/gen_data.py
    $V drafter/capture.py
    $V drafter/train_mtp.py --out drafter/runs/uncensored --eval-only 1 \
      --draft-ids prepare/draft_vocab_ids.json --max-seqs 400 --val-frac 0.4 \
      --depths 2 --dump-hessians drafter/runs/uncensored/mtp_hessians.pt
    $V drafter/gptq_lm_head.py '"$BASE"' models/tmp-uncensored-lm4 --bits 4 --calib-rows 300000
    $V prepare/build_draft_vocab.py models/tmp-uncensored-lm4 --ids prepare/draft_vocab_ids.json
    $V drafter/requant_mtp_gptq.py models/tmp-uncensored-lm4 '"$FAST"' \
      drafter/runs/uncensored/mtp_hessians.pt --bits 4
    MODEL='"$FAST"' bash verify.sh --no-server
  '

find "$DEST/models/Qwen3.8-27B-abliterated-AWQ-MTP-fast" \
  -type f -print0 | sort -z | xargs -0 sha256sum > "$DEST/models/qwen38-uncensored-fast.sha256"
echo "uncensored checkpoint-specific fast artifact is ready"
