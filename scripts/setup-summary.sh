#!/usr/bin/env bash
# Operator-only: dedicated fp16 baseline environment; does not alter a service.
set -euo pipefail
install_dir="${1:?Usage: setup-summary.sh INSTALL_DIR EXISTING_LOCAL_MODEL}"
model_source="${2:?Supply the existing Qwen3-4B-Instruct-2507 model directory}"
model_source="$(realpath "$model_source")"
test -f "$model_source/config.json"
mkdir -p "$install_dir"
python3 -m venv "$install_dir/venv"
"$install_dir/venv/bin/pip" install --no-cache-dir 'vllm==0.29.0'
# Reuse existing offline weights; neither downloads nor the default HF cache are consulted.
if [[ -e "$install_dir/model" || -L "$install_dir/model" ]]; then
  [[ "$(realpath "$install_dir/model")" == "$model_source" ]] || {
    echo "Existing model path points elsewhere; refusing to replace it" >&2
    exit 1
  }
else
  ln -s -- "$model_source" "$install_dir/model"
fi
"$install_dir/venv/bin/pip" freeze > "$install_dir/requirements.lock"
echo "Baseline environment ready at $install_dir; benchmark before enabling the runner."
