#!/usr/bin/env bash
# Operator-only: isolated serving environment; does not alter a service.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_dir="${1:?Usage: setup-summary.sh INSTALL_DIR EXISTING_W8A8_CHECKPOINT}"
model_source="${2:?Supply ~/emb-test/llm/Qwen3-4B-W8A8 (or a reproduced INT8 checkpoint)}"
model_source="$(realpath "$model_source")"
# Fail before installing anything if handed unquantized or FP8 weights.
PYTHONPATH="$script_dir" python3 - "$model_source" <<'PY'
import sys
from simple_ai_summary_provider import check_checkpoint
check_checkpoint(sys.argv[1])
PY
mkdir -p "$install_dir"
export HF_HOME="$install_dir/cache" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1
python3 -m venv "$install_dir/venv"
"$install_dir/venv/bin/pip" install --no-cache-dir 'vllm==0.29.0' 'compressed-tensors==0.20.1'
# Reuse local weights; neither downloads nor the default HF cache are consulted.
if [[ -e "$install_dir/model" || -L "$install_dir/model" ]]; then
  [[ "$(realpath "$install_dir/model")" == "$model_source" ]] || {
    echo "Existing model path points elsewhere; refusing to replace it" >&2
    exit 1
  }
else
  ln -s -- "$model_source" "$install_dir/model"
fi
"$install_dir/venv/bin/pip" freeze > "$install_dir/requirements.lock"
echo "W8A8 environment ready at $install_dir; add venv/bin to the runner PATH for ninja."
