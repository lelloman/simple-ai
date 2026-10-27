#!/usr/bin/env bash
# Operator prerequisite: use a Python environment with the host-qualified ROCm torch.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
translation_python="${1:?Usage: setup-translation.sh /path/to/venv/bin/python /path/to/model}"
translation_model_dir="${2:?Model directory required}"
"$translation_python" -m pip install -r "$script_dir/translation-requirements.txt"
"$translation_python" - "$translation_model_dir" <<'PY'
import sys
import torch
from huggingface_hub import snapshot_download
if not torch.cuda.is_available():
    raise SystemExit('Halo production translation requires a qualified ROCm torch environment')
snapshot_download('google/madlad400-3b-mt', revision='fa184c675da0b5c9e1c8694fccd4e12e2d422094', local_dir=sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.model'])
PY
