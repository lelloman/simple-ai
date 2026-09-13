#!/usr/bin/env python3
"""Check the deployed alias through the runner API, including stream/tool paths."""
import argparse
import json
import pathlib
import statistics
import urllib.request
import benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--label', default='runner-after')
parser.add_argument('--validate', action='store_true')
args = parser.parse_args()
benchmark.URL = 'http://127.0.0.1:8080'
out = benchmark.ROOT/'results'/args.label
out.mkdir(parents=True, exist_ok=True)
original_request = benchmark.request

def request(prompt, tokens=512, cache=False, overrides=None):
    overrides = {'reasoning_effort':'none', **(overrides or {})}
    return original_request(prompt, tokens, cache, overrides)

benchmark.request = request
warmup = request('Reply with exactly: Ready.',32)
(out/'warmup.json').write_text(json.dumps(warmup,indent=2))
results = []
for name,prompt in benchmark.PROMPTS:
    r = request(prompt)
    (out/(name+'.json')).write_text(json.dumps(r,indent=2))
    results.append({'name':name,'decode_tps':r['decode_tps'],'ttft_s':r['ttft_s'],'usage':r['usage']})
summary = {'label':args.label,'runs':results,'median_tps':statistics.median(r['decode_tps'] for r in results)}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
if args.validate:
    benchmark.validate(out)
    payload = {'model':'qwen3.6-35b-a3b','messages':[{'role':'user','content':'Reply with exactly: Ready.'}],
               'max_tokens':32,'temperature':0,'stream':False}
    req = urllib.request.Request(benchmark.URL+'/v1/chat/completions',data=json.dumps(payload).encode(),
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response:
        answer = json.load(response)
    (out/'default-nonstream.json').write_text(json.dumps(answer,indent=2))
    assert answer['choices'][0]['message']['content'].strip()=='Ready.',answer
    assert not answer['choices'][0]['message'].get('reasoning_content'),answer
    print('Non-streaming default-effort check passed',flush=True)
