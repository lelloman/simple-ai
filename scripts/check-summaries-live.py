#!/usr/bin/env python3
"""Operator-only gateway timing and reload capture; grounding scored separately."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def call(base, key, path, body):
    start = time.perf_counter()
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        result = json.load(response)
    return result, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='canonical card JSONL {id,text}, first 1000 tested')
    p.add_argument('--reload-probe', required=True, help='JSON {path,body}, concrete model on RTX')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    key = os.environ['SIMPLE_AI_API_KEY']
    base = 'https://ai.lelloman.com'
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()][:1000]
    if len(rows) != 1000 or len({r['id'] for r in rows}) != 1000:
        raise ValueError('1000 distinct card IDs required')
    probe = json.loads(Path(args.reload_probe).read_text())
    if probe['path'] not in ['/v1/embeddings', '/v1/classifications', '/v1/extractions', '/v1/decisions'] or probe['body']['model'].startswith('class:'):
        raise ValueError('probe must use a concrete RTX model on a supported inference endpoint')
    report = {'complete': False, 'results': [], 'reload_probes': []}
    try:
        call(base, key, '/v1/summaries', {'model': 'class:summarize', 'input': ['Example operates a website.'], 'max_tokens': 60})
        start = time.perf_counter()
        for offset in range(0, 1000, 64):
            batch = rows[offset:offset + 64]
            result, _ = call(base, key, '/v1/summaries', {
                'model': 'class:summarize', 'input': [r['text'] for r in batch], 'max_tokens': 60})
            data = result['data']
            if len(data) != len(batch) or any(r['index'] != i or not r['summary'].strip() for i, r in enumerate(data)):
                raise RuntimeError('invalid summary output')
            report['results'].extend({'id': row['id'], 'text': row['text'], 'summary': output['summary']}
                                     for row, output in zip(batch, data))
        report['seconds'] = time.perf_counter() - start
        report['sites_per_second'] = 1000 / report['seconds']
        for _ in range(3):
            _, other_seconds = call(base, key, probe['path'], probe['body'])
            _, summary_seconds = call(base, key, '/v1/summaries', {
                'model': 'class:summarize', 'input': [rows[0]['text']], 'max_tokens': 60})
            report['reload_probes'].append({'other_model_seconds': other_seconds, 'summary_seconds': summary_seconds})
        report['complete'] = True
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    if report['sites_per_second'] < 15:
        raise SystemExit('below 15 sites/s; inspect report')


if __name__ == '__main__':
    main()
