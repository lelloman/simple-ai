#!/usr/bin/env python3
"""Operator live acceptance harness. Uses SIMPLE_AI_API_KEY without recording it.

Input: JSON array of exactly 150 Italian descriptions, each <=400 characters.
Probe files: ordinary JSON request payloads for embeddings and classifications.
Verify both probes route to Halo 2 in the gateway audit before trusting results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--embedding-probe', type=Path, required=True)
    parser.add_argument('--classification-probe', type=Path, required=True)
    parser.add_argument('--url', default='https://ai.lelloman.com')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    descriptions = json.loads(args.input.read_text())
    if not isinstance(descriptions, list) or len(descriptions) != 150 or any(not isinstance(s, str) or not s.strip() or len(s) > 400 for s in descriptions):
        parser.error('input must contain exactly 150 non-blank descriptions of at most 400 characters')
    token = os.environ['SIMPLE_AI_API_KEY']
    probes = {
        'embeddings': json.loads(args.embedding_probe.read_text()),
        'classifications': json.loads(args.classification_probe.read_text()),
    }
    report = {'url': args.url, 'batch_size': 32, 'probes': {}, 'languages': {}, 'errors': []}
    stop = threading.Event()

    def post(endpoint, body):
        start = time.perf_counter()
        req = Request(args.url.rstrip('/') + '/v1/' + endpoint, data=json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with urlopen(req, timeout=180) as response:
            value = json.load(response)
            status = response.status
        return value, {'seconds': time.perf_counter() - start, 'status': status, 'model': value.get('model')}

    def probe_phase(phase, repeat):
        for endpoint, payload in probes.items():
            rows = []
            for _ in range(repeat):
                _, measurement = post(endpoint, payload)
                rows.append(measurement)
            report['probes'].setdefault(phase, {})[endpoint] = rows

    def during():
        # Low-frequency identical probes avoid overwhelming translation traffic.
        while not stop.is_set():
            for endpoint, payload in probes.items():
                try:
                    _, row = post(endpoint, payload)
                    report['probes'].setdefault('during', {}).setdefault(endpoint, []).append(row)
                except Exception as error:
                    report['errors'].append({'probe': endpoint, 'error': str(error)})
            stop.wait(1)

    try:
        probe_phase('before', 3)
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = executor.submit(during)
            start = time.perf_counter()
            try:
                for language in ['en', 'de', 'fr', 'es', 'ar', 'zh-Hans', 'ru']:
                    rows = []
                    language_start = time.perf_counter()
                    for offset in range(0, 150, 32):
                        batch = descriptions[offset:offset + 32]
                        response, measurement = post('translations', {'model': 'class:translate', 'input': batch, 'target': language, 'source': 'it'})
                        data = response['data']
                        if len(data) != len(batch) or [row['index'] for row in data] != list(range(len(batch))) or any(not isinstance(row['translation'], str) or not row['translation'].strip() for row in data):
                            raise RuntimeError('invalid or empty translation batch')
                        rows.extend(data)
                    elapsed = time.perf_counter() - language_start
                    report['languages'][language] = {'seconds': elapsed, 'sentences_per_second': 150 / elapsed, 'sample': rows[0]['translation'], 'count': len(rows)}
                report['translation_seconds'] = time.perf_counter() - start
                report['sentences_per_second'] = 1050 / report['translation_seconds']
            finally:
                stop.set()
                worker.result()
        probe_phase('after', 3)
    except Exception as error:
        report['errors'].append({'error': str(error)})
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"1050 sentences in {report['translation_seconds']:.2f}s, {report['sentences_per_second']:.2f} sentences/s; {len(report['errors'])} errors")
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
