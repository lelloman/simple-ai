"""Fetch a bounded, reproducible calibration sample without full dataset download."""
import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset

REPO = "mlabonne/open-perfectblend"
REVISION = "af60f3c18201652a83a93f46fcfee1b646ba3df7"
output = Path(sys.argv[1])
dataset = load_dataset(REPO, revision=REVISION, split="train", streaming=True)
roles = {"human": "user", "gpt": "assistant"}
with output.open("w") as stream:
    for row in dataset.take(128):
        messages = [{"role": roles.get(m["from"], m["from"]), "content": m["value"]}
                    for m in row["conversations"]]
        stream.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
assert len(output.read_text().splitlines()) == 128
output.with_suffix(".manifest.json").write_text(json.dumps({
    "repo": REPO, "revision": REVISION, "samples": 128,
    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}, indent=2) + "\n")
print("Calibration sample ready", flush=True)
