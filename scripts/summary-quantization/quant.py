#!/usr/bin/env python3
"""Reproduce the AI-60 SmoothQuant + GPTQ INT8 recipe on RTX only.

Input: exactly 512 separate calibration records {id,text}, already card/profile
texts. Evaluation records use the same shape, only for leakage detection.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def calibration_rows(calibration, evaluation):
    def read(path):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if any(not isinstance(r.get('id'), str) or not r['id']
               or not isinstance(r.get('text'), str) or not r['text'].strip() for r in rows):
            raise ValueError('records must contain nonblank string id and text')
        return rows
    rows, held_out = read(calibration), read(evaluation)
    if len(rows) != 512 or len({r['id'] for r in rows}) != 512:
        raise ValueError('exactly 512 distinct calibration site IDs required')
    if (not held_out or {r['id'] for r in rows} & {r['id'] for r in held_out}
            or {r['text'] for r in rows} & {r['text'] for r in held_out}):
        raise ValueError('calibration and held-out evaluation must be separate sites/texts')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='local unquantized Qwen3-4B-Instruct-2507')
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--evaluation', required=True, help='held-out records, never used for calibration')
    parser.add_argument('--output', required=True, help='new directory; never overwrites the qualified checkpoint')
    args = parser.parse_args()
    rows = calibration_rows(args.calibration, args.evaluation)
    output = Path(args.output)
    if output.exists():
        raise ValueError('output already exists; refusing to replace a checkpoint')
    os.environ['HF_HOME'] = str(output.resolve().parent / 'summary-quant-cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    from simple_ai_summary_provider import PROMPT, check_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
    samples = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': row['text']}],
            tokenize=True, add_generation_prompt=True)
        if len(ids) > 1024:
            raise ValueError('calibration prompt exceeds 1024 tokens; supply bounded cards/profiles')
        samples.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})
    oneshot(model=model, dataset=Dataset.from_list(samples), recipe=[
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets='Linear', scheme='W8A8', ignore=['lm_head']),
    ], max_seq_length=1024, num_calibration_samples=512)
    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    check_checkpoint(output)
    manifest = {'command': sys.argv, 'source_model': str(Path(args.model).resolve()),
                'prompt': PROMPT, 'calibration_sha256': hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest(),
                'evaluation_sha256': hashlib.sha256(Path(args.evaluation).read_bytes()).hexdigest(),
                'versions': {p: importlib.metadata.version(p) for p in
                             ['llmcompressor', 'compressed-tensors', 'transformers', 'torch']}}
    (output / 'summary-quantization.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'requirements.lock').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))


if __name__ == '__main__':
    main()
