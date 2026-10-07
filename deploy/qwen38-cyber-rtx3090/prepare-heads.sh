#!/usr/bin/env bash
set -euo pipefail
# Run inside the pinned serving image with a writable derived model directory.
model=${1:?model directory required}
cd /app
venv/bin/python /work/quant-heads-stream.py "$model" --mtp-bits 8
venv/bin/python - "$model" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "config.json"
config = json.loads(path.read_text())
quant = config["quantization_config"]
# The converter's regex ignores also match newly packed tensors. They must no
# longer exclude the heads that the three derived quantization groups own.
quant["ignore"] = [rule for rule in quant["ignore"]
                   if not any(name in rule for name in ("mtp", "lm_head", "embed_tokens"))]
for name in ("group_1", "group_2", "group_3"):
    weights = quant["config_groups"][name]["weights"]
    assert weights["num_bits"] == 8 and weights["symmetric"] is True
path.write_text(json.dumps(config, indent=2) + "\n")
PY
venv/bin/python prepare/build_draft_vocab.py "$model" --ids prepare/draft_vocab_ids.json
venv/bin/python - "$model" <<'PY'
import json
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])
path = root / "model.safetensors.index.json"
index = json.loads(path.read_text())
headers = {}
for shard in set(index["weight_map"].values()):
    with (root / shard).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        headers[shard] = json.loads(stream.read(size))
total = 0
for name, shard in index["weight_map"].items():
    tensor = headers[shard][name]
    start, end = tensor["data_offsets"]
    total += end - start
index["metadata"]["total_size"] = total
path.write_text(json.dumps(index, indent=2) + "\n")
print(f"Verified {len(index['weight_map'])} indexed tensors; {total} weight bytes")
PY
MODEL="$model" bash verify.sh --no-server
