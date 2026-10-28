#!/usr/bin/env python3
"""Run one offline AI-60 matrix cell on RTX. Canonical JSONL: {id,text}."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--max-num-seqs', type=int, choices=[256, 512], required=True)
    p.add_argument('--ngram', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    # Separate offline environment; never read the default Hugging Face token.
    os.environ['HF_HOME'] = str(Path(args.model_path).resolve().parent / 'cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    from simple_ai_summary_provider import Provider, MODEL, prompts
    from vllm import SamplingParams, __version__
    raw = Path(args.input).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 1500 or any(not isinstance(r.get('text'), str) or not r['text'].strip() for r in rows):
        raise ValueError('exactly 1500 nonblank texts required; align profile/card IDs before running')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != 1500:
        raise ValueError('duplicate site IDs')
    peak, sample_errors = [], []
    stop = threading.Event()

    def sample():
        try:
            while not stop.is_set():
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                        check=True, capture_output=True, text=True, timeout=5)
                peak.append(max(int(x) for x in result.stdout.splitlines()))
                stop.wait(.1)
        except Exception as error:
            sample_errors.append(str(error))
    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        provider = Provider(args.model_path, max_num_seqs=args.max_num_seqs, ngram=args.ngram)
        # Separate synthetic warmup: no held-out site can enter the prefix cache before timing.
        provider.summarize({'model': MODEL, 'input': ['Example operates a website.']})
        inputs = prompts(provider.tokenizer, [r['text'] for r in rows], 60)
        start = time.perf_counter()
        outputs = provider.llm.generate(inputs, SamplingParams(temperature=0, max_tokens=60), use_tqdm=False)
        elapsed = time.perf_counter() - start
        if len(outputs) != 1500 or any(not o.outputs[0].text.strip() for o in outputs):
            raise RuntimeError('missing summaries')
        stop.set()
        sampler.join()
        if sample_errors or not peak:
            raise RuntimeError('VRAM measurement failed: ' + '; '.join(sample_errors))
        Path(args.output + '.outputs.jsonl').write_text(''.join(json.dumps({
            'id': row['id'], 'text': row['text'], 'summary': output.outputs[0].text.strip()
        }, ensure_ascii=False) + '\n' for row, output in zip(rows, outputs)))
        report = {'command': shlex.join(sys.argv), 'vllm_version': __version__,
                  'input_sha256': hashlib.sha256(raw).hexdigest(), 'sites': len(rows),
                  'seconds': elapsed, 'sites_per_second': len(rows) / elapsed,
                  'gpu_wall_seconds_per_1000_sites': elapsed * 1000 / len(rows),
                  'peak_total_gpu_memory_mib': max(peak) if peak else None,
                  'max_num_seqs': args.max_num_seqs, 'ngram': args.ngram,
                  'model_path': str(Path(args.model_path).resolve()),
                  'grounding_score': None, 'manual_place_review': None}
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    finally:
        stop.set()
        sampler.join()


if __name__ == '__main__':
    main()
