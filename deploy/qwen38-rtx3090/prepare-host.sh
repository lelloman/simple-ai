#!/usr/bin/env bash
set -euo pipefail

DEST=${1:-/home/lelloman/qwen38-serving}
UPSTREAM=https://github.com/syv-ai/qwen38-27b-rtx3090.git
UPSTREAM_REF=a75ee4be40098e9d0b239cf4550ee12c4ac49338
MODEL_REPO=twolven/Qwen3.8-27B-abliterated-AWQ-MTP
MODEL_REV=c2f5341e7c9a1c55e6d61cfc8e6d0ca897bd7443

available_kib=$(df -Pk "$(dirname "$DEST")" | awk 'NR==2 {print $4}')
required_kib=125829120
if [ -s "$DEST/models/Qwen3.8-27B-abliterated-AWQ-MTP/model.safetensors.index.json" ]; then
  # A resumed run only needs room for the remaining derived tensors.
  required_kib=20971520
fi
if [ "$available_kib" -lt "$required_kib" ]; then
  echo "insufficient disk: need $((required_kib / 1024 / 1024)) GiB free" >&2
  exit 1
fi

if [ ! -d "$DEST/.git" ]; then
  git clone "$UPSTREAM" "$DEST"
fi
git -C "$DEST" fetch origin "$UPSTREAM_REF"
git -C "$DEST" checkout --detach "$UPSTREAM_REF"

# The pinned upstream calibration collector imports pyarrow but its container
# requirements currently omit it. Keep the dependency explicit and reproducible.
if ! grep -q '^pyarrow==' "$DEST/docker/requirements.txt"; then
  printf '\npyarrow==21.0.0\n' >> "$DEST/docker/requirements.txt"
fi
# The upstream custom-dataset benchmark imports pandas through vLLM's optional
# bench extra. Install the one dependency its scripts actually use.
if ! grep -q '^pandas==' "$DEST/docker/requirements.txt"; then
  printf 'pandas==2.3.2\n' >> "$DEST/docker/requirements.txt"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cp "$SCRIPT_DIR/docker-compose.override.yml" "$DEST/docker-compose.override.yml"
mkdir -p "$DEST/profiles"
cp "$SCRIPT_DIR/profiles/base.env.example" "$DEST/profiles/base.env"
cp "$SCRIPT_DIR/profiles/fast.env.example" "$DEST/profiles/fast.env"
cp "$SCRIPT_DIR/profiles/long.env.example" "$DEST/profiles/long.env"
cp "$SCRIPT_DIR/acceptance.py" "$DEST/acceptance.py"
cp "$SCRIPT_DIR/quick-benchmark.py" "$DEST/quick-benchmark.py"
cp "$SCRIPT_DIR/fix-checkpoint-metadata.py" "$DEST/fix-checkpoint-metadata.py"
git -C "$DEST" restore --source "$UPSTREAM_REF" -- \
  single-user/start_qwen.sh prepare/quant_mtp.py prepare/build_draft_vocab.py
git -C "$DEST" apply --unidiff-zero --check "$SCRIPT_DIR/syv-vision.patch"
git -C "$DEST" apply --unidiff-zero "$SCRIPT_DIR/syv-vision.patch"

if [ ! -s "$DEST/api_key.txt" ]; then
  openssl rand -hex 24 > "$DEST/api_key.txt"
  chmod 600 "$DEST/api_key.txt"
fi
api_key=$(tr -d '\r\n' < "$DEST/api_key.txt")
for profile in base fast long; do
  printf '\nVLLM_API_KEY=%s\nMODEL_REVISION=%s\n' "$api_key" "$MODEL_REV" >> "$DEST/profiles/$profile.env"
done

docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" build
revision_marker="$DEST/models/Qwen3.8-27B-abliterated-AWQ-MTP/.simple-ai-source-revision"
if ! grep -qx "$MODEL_REV" "$revision_marker" 2>/dev/null; then
  docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" \
    --env-file "$DEST/profiles/fast.env" run --rm prepare \
    venv/bin/hf download "$MODEL_REPO" --revision "$MODEL_REV" \
    --local-dir /app/models/Qwen3.8-27B-abliterated-AWQ-MTP
  docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" \
    --env-file "$DEST/profiles/fast.env" run --rm prepare \
    bash -lc 'printf "%s\n" "$MODEL_REVISION" > "$BASE_MODEL_DIR/.simple-ai-source-revision"'
fi
# Refuse to let upstream's unpinned fallback downloader repair a partial tree.
# A missing shard must be recovered explicitly from this deployment's pinned
# revision before any in-place quantization starts.
docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" \
  --env-file "$DEST/profiles/fast.env" run --rm prepare venv/bin/python - <<'PY'
import json
import os

root = os.environ["BASE_MODEL_DIR"]
with open(os.path.join(root, "model.safetensors.index.json")) as stream:
    index = json.load(stream)["weight_map"]
missing = sorted(
    shard for shard in set(index.values())
    if not os.path.isfile(os.path.join(root, shard))
)
if missing:
    raise SystemExit(f"pinned checkpoint is incomplete; missing shards: {missing}")
config = json.load(open(os.path.join(root, "config.json")))
groups = config.get("quantization_config", {}).get("config_groups", {})
if "group_0" not in groups:
    raise SystemExit("checkpoint is not a compressed-tensors quantization supported by this pipeline")
print(f"pinned checkpoint layout valid ({len(index)} tensors, {len(set(index.values()))} shards)")
PY
docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" \
  --env-file "$DEST/profiles/fast.env" run --rm prepare
docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" \
  --env-file "$DEST/profiles/fast.env" run --rm prepare venv/bin/python \
  /app/fix-checkpoint-metadata.py /app/models/Qwen3.8-27B-abliterated-AWQ-MTP
docker compose -f "$DEST/docker-compose.yml" -f "$DEST/docker-compose.override.yml" \
  --env-file "$DEST/profiles/fast.env" run --rm \
  -e MODEL=/app/models/Qwen3.8-27B-abliterated-AWQ-MTP single verify --no-server

echo "base uncensored artifact prepared; checkpoint-specific fast calibration is still required"
