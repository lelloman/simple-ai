#!/usr/bin/env python3
"""Make the derived int8 tensors match vLLM's compressed-tensors metadata.

The source AWQ checkpoint excludes its original bf16 MTP module with a broad
regex. After quant_mtp.py replaces those tensors, leaving that rule in place
causes vLLM to construct unquantized modules and reject the packed weights.
The upstream requantizers also inherit the source checkpoint's asymmetric W4
metadata even though the derived int8 tensors are symmetric and have no zero
points. This repair is deterministic and idempotent; it never touches tensors.
"""

import json
import os
import sys


root = sys.argv[1].rstrip("/")
path = os.path.join(root, "config.json")
with open(path) as stream:
    config = json.load(stream)

quant = config["quantization_config"]
before = list(quant.get("ignore", []))
quant["ignore"] = [item for item in before if "mtp" not in item.lower()]

groups = quant["config_groups"]
for name in ("group_1", "group_2", "group_3"):
    weights = groups[name]["weights"]
    if weights["num_bits"] != 8:
        raise SystemExit(f"{name} is not the expected derived int8 group")
    weights["symmetric"] = True
    weights["zp_dtype"] = None

temp = path + ".simple-ai.tmp"
with open(temp, "w") as stream:
    json.dump(config, stream, indent=2)
    stream.write("\n")
os.replace(temp, path)

removed = sorted(set(before) - set(quant["ignore"]))
print(f"checkpoint metadata repaired; removed MTP ignores: {removed}")
