#!/usr/bin/env python3
"""Score saved {text,summary} or live {input,summary} JSONL; no GPU required."""
import argparse
import hashlib
import json
from pathlib import Path
from simple_ai_summary_provider import place_leaks


def score(rows):
    if not rows:
        raise ValueError('no summaries')
    details = []
    for index, row in enumerate(rows):
        text = row.get('text', row.get('input'))
        summary = row.get('summary')
        if not isinstance(text, str) or not text.strip() or not isinstance(summary, str) or not summary.strip():
            raise ValueError(f'row {index + 1}: nonblank source and summary strings required')
        tokens = place_leaks(text, summary)
        if tokens:
            details.append({'id': row.get('id', index + 1), 'tokens': tokens})
    return {'sites': len(rows), 'place_leak_count': len(details),
            'place_leak_rate': len(details) / len(rows), 'place_leaks_by_id': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--grounding-score', type=float,
                        help='unchanged ground.py percentage from its operator-run report')
    args = parser.parse_args()
    if args.grounding_score is not None and not 0 <= args.grounding_score <= 100:
        parser.error('grounding score must be a percentage in [0,100]')
    raw = Path(args.input).read_bytes()
    report = score([json.loads(line) for line in raw.splitlines() if line.strip()])
    report.update(input_sha256=hashlib.sha256(raw).hexdigest(), grounding_score=args.grounding_score)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
