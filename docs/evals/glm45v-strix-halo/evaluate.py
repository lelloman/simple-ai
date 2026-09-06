#!/usr/bin/env python3
"""Small GLM-4.5V smoke test with streaming timing and raw responses."""
import base64
import json
import pathlib
import statistics
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
URL = 'http://127.0.0.1:18045/v1/chat/completions'

def run(name, content, max_tokens=256):
    payload = dict(model='glm-4.5v', messages=[dict(role='user', content=content)],
                   max_tokens=max_tokens, temperature=0, seed=7319, stream=True,
                   stream_options={'include_usage': True},
                   chat_template_kwargs={'enable_thinking': False}, cache_prompt=False)
    request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = None
    chunks = []
    answer = ''
    reasoning = ''
    usage = {}
    timings = {}
    finish = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            usage = chunk.get('usage') or usage
            timings = chunk.get('timings') or timings
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                if (text or thought) and first is None:
                    first = time.monotonic()
                answer += text
                reasoning += thought
                finish = choice.get('finish_reason') or finish
    end = time.monotonic()
    tokens = usage.get('completion_tokens', 0)
    result = dict(name=name, elapsed_s=end-start, ttft_s=first-start if first else None,
                  decode_tps=(tokens-1)/(end-first) if first and tokens>1 else None,
                  usage=usage, timings=timings, finish_reason=finish,
                  answer=answer, reasoning=reasoning)
    (ROOT/'results'/f'{name}.json').write_text(json.dumps(dict(request=payload, result=result, chunks=chunks), indent=2))
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    (ROOT/'results').mkdir(exist_ok=True)
    run('warmup', 'Reply with exactly: Ready.', 32)
    results = []
    prompts = [
        'Write a Python function that merges overlapping half-open integer intervals. Explain how it handles empty intervals and give examples.',
        'Explain how a hash table works, including collisions, resizing, and average versus worst-case lookup complexity. Give a small example.',
        'Write a detailed step-by-step guide to finding which process is listening on a TCP port on Linux and checking its logs.',
    ]
    for i, prompt in enumerate(prompts):
        results.append(run(f'text-{i+1}', prompt))
    uri = 'data:image/png;base64,' + base64.b64encode((ROOT/'vision.png').read_bytes()).decode()
    vision = run('vision', [
        dict(type='text', text='Read both lines of text exactly. Then list the three shapes and their colors from left to right. Be concise.'),
        dict(type='image_url', image_url=dict(url=uri)),
    ], 192)
    ocr = run('ocr', [
        dict(type='text', text='Transcribe all text visible in this image, including the heading and amount. Return only the text.'),
        dict(type='image_url', image_url=dict(url=uri)),
    ], 128)
    summary = dict(text_median_decode_tps=statistics.median(r['decode_tps'] for r in results),
                   text_median_ttft_s=statistics.median(r['ttft_s'] for r in results),
                   text_runs=results, vision=vision, ocr=ocr)
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'summary': summary}), flush=True)
