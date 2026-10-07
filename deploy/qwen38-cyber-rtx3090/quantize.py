"""Calibrate Nico's pinned BF16 checkpoint for the RTX W4A16 serving stack.

Run in Dockerfile.quantize with the staging parent mounted at /work, GPU access,
and UID/GID matching the staging owner. Source/output/offload can live on the
temporary SSHFS storage mount; the serving artifact is copied locally afterward.
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    with load_context(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(args.source), dtype=torch.bfloat16,
            device_map="auto_offload", max_memory={"cpu": 16 * 1024**3},
            offload_folder=str(args.offload),
        )
    processor = AutoProcessor.from_pretrained(str(args.source))
    calibration = []
    for line in args.calibration.read_text().splitlines()[:args.samples]:
        row = json.loads(line)
        calibration.append({"text": processor.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False,
        )})
    assert len(calibration) == args.samples
    recipe = [
        GPTQModifier(
            targets="Linear", scheme="W4A16",
            ignore=[
                "re:visual.*", "re:model.visual.*", "re:.*lm_head",
                "re:.*embed_tokens$", r"re:.*linear_attn\.in_proj_a$",
                r"re:.*linear_attn\.in_proj_b$", r"re:.*mtp.*",
            ],
        ),
    ]
    oneshot(
        model=model, processor=processor, recipe=recipe,
        dataset=Dataset.from_list(calibration),
        max_seq_length=1024, num_calibration_samples=args.samples,
        pipeline="sequential",
    )
    model.save_pretrained(str(args.output), save_compressed=True, max_shard_size="2GB")
    processor.save_pretrained(str(args.output))

    # Transformers may discard native MTP tensors as unexpected keys. Preserve
    # the exact fine-tune's MTP weights, never substitute another checkpoint's.
    source_index = json.loads((args.source / "model.safetensors.index.json").read_text())
    mtp_map = {k: v for k, v in source_index["weight_map"].items() if k.startswith("mtp.")}
    if not mtp_map:
        raise RuntimeError("Source checkpoint has no MTP weights")
    tensors = {}
    for shard in sorted(set(mtp_map.values())):
        with safe_open(str(args.source / shard), framework="pt") as handle:
            for name, filename in mtp_map.items():
                if filename == shard:
                    tensors[name] = handle.get_tensor(name).clone()
    save_file(tensors, str(args.output / "model-mtp.safetensors"), metadata={"format": "pt"})
    index_path = args.output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update({k: "model-mtp.safetensors" for k in tensors})
    index["metadata"]["total_size"] = sum(
        (args.output / shard).stat().st_size for shard in set(index["weight_map"].values())
    )
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    config_path = args.output / "config.json"
    config = json.loads(config_path.read_text())
    source_config = json.loads((args.source / "config.json").read_text())
    for key, value in source_config["text_config"].items():
        if key.startswith("mtp_"):
            config["text_config"][key] = value
    ignores = config["quantization_config"].setdefault("ignore", [])
    for name in tensors:
        if name.endswith(".weight") and tensors[name].ndim == 2:
            module = name.removesuffix(".weight")
            if module not in ignores:
                ignores.append(module)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print("W4A16 conversion complete; native MTP weights preserved", flush=True)


if __name__ == "__main__":
    main()
