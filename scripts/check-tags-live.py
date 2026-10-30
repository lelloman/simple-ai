#!/usr/bin/env python3
"""Operator-only gateway throughput probe; retain outputs for tags-v2 comparison."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='JSONL: {id,request}, using real tags-v2 chat bodies')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = open(args.input, 'rb').read()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) < 32 or len({r['id'] for r in rows}) != len(rows):
        parser.error('need at least 32 distinct site IDs')
    for row in rows:
        body = row['request']
        if not body.get('messages') or body.get('response_format', {}).get('type') != 'json_schema':
            parser.error('each request must include messages and a json_schema response_format')
        body.update(model='class:tag', stream=False)
        body.setdefault('max_tokens', 100)
    url = os.environ.get('SIMPLE_AI_URL', 'https://ai.lelloman.com').rstrip('/') + '/v1/chat/completions'
    token = os.environ['SIMPLE_AI_API_KEY']

    def call(row):
        request = urllib.request.Request(url, json.dumps(row['request']).encode(), headers={
            'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.load(response)
            runner = response.headers.get('X-Runner-Id')
        # Fail immediately on invalid JSON rather than counting HTTP success as guidance success.
        tags = json.loads(body['choices'][0]['message']['content'])
        return {'id': row['id'], 'response': body, 'tags': tags, 'runner_id': runner,
                'seconds': time.monotonic() - start}

    call(rows[0])  # Exclude cold loading; report it separately in operator notes.
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        outputs = list(pool.map(call, rows))
    elapsed = time.monotonic() - start
    report = {'concurrency': 16, 'sites': len(rows), 'seconds': elapsed,
              'sites_per_second': len(rows) / elapsed, 'input_sha256': hashlib.sha256(raw).hexdigest(),
              'outputs': outputs}
    with open(args.output, 'x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != 'outputs'}))
    if report['sites_per_second'] < 4.6:
        raise SystemExit('throughput gate failed: need >=4.6 sites/s')


if __name__ == '__main__':
    main()
