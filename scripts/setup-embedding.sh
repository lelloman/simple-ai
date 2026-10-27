#!/usr/bin/env bash
# Run on each runner host. Installs an isolated provider environment and pinned model.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_dir="${1:-$HOME/.simple-ai-embedding}"
wheel_kind="cu130"
mkdir -p "$install_dir"
python3 -m venv "$install_dir/venv"
"$install_dir/venv/bin/pip" install --no-cache-dir 'torch==2.13.0' --index-url "https://download.pytorch.org/whl/$wheel_kind"
"$install_dir/venv/bin/pip" install --no-cache-dir -r "$script_dir/embedding-requirements.txt"
HF_HOME="$install_dir/cache" "$install_dir/venv/bin/python" - "$install_dir/model" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Embedding-0.6B', revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3', local_dir=sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt'])
PY
"$install_dir/venv/bin/pip" freeze > "$install_dir/requirements.lock"
echo "Embedding environment and model ready: $install_dir"
