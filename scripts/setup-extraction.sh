#!/usr/bin/env bash
# Run on each runner host. Installs an isolated provider environment and pinned model.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_dir="${1:-$HOME/.simple-ai-extraction}"
wheel_kind="${2:-cpu}"
case "$wheel_kind" in cpu|cu130) ;; *) echo 'Usage: setup-extraction.sh [install-dir] [cpu|cu130]' >&2; exit 2;; esac
mkdir -p "$install_dir"
python3 -m venv "$install_dir/venv"
"$install_dir/venv/bin/pip" install --no-cache-dir 'torch==2.13.0' --index-url "https://download.pytorch.org/whl/$wheel_kind"
"$install_dir/venv/bin/pip" install --no-cache-dir -r "$script_dir/extraction-requirements.txt"
HF_HOME="$install_dir/cache" "$install_dir/venv/bin/python" - "$install_dir/model" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download('fastino/gliner2.5-multi-v1', revision='235cf92d6d4318da9bfca0d08975c8fa7250d13b', local_dir=sys.argv[1], allow_patterns=['*.json', '*.safetensors'])
PY
"$install_dir/venv/bin/pip" freeze > "$install_dir/requirements.lock"
echo "Extraction environment and model ready: $install_dir"
