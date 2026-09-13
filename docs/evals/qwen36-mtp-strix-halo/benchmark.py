#!/usr/bin/env python3
"""Isolated Halo Qwen3.6 baseline/MTP comparison; saves requests and responses."""
import argparse
import json
import pathlib
import statistics
import subprocess
import time
import urllib.request

ROOT = pathlib.Path('/home/lelloman/qwen36-mtp-test')
BASE = '/home/lelloman/model-catalog/Qwen3.6-35B-A3B-MXFP4_MOE.gguf'
DRAFT = str(ROOT / 'models/Qwen3.6-35B-A3B-MTP-q8_0.gguf')
URL = 'http://127.0.0.1:18036'
NAME = 'qwen36-mtp-bench'
PROMPTS = [
    ('code', 'Write a complete Python implementation of an LRU cache using a dictionary and a doubly linked list. Include get, put, and tests. Explain the complexity.'),
    ('prose', 'Explain how a database transaction provides ACID guarantees, using a bank transfer as an example. Cover isolation levels, write-ahead logging, and crash recovery in detail.'),
    ('json', 'Return only a JSON array of 50 objects. Each object has id (1 through 50), name (item_N), square (id squared), and even (boolean).'),
]

def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)

def request(prompt, tokens=512, cache=False, overrides=None):
    payload = dict(model='qwen3.6-35b-a3b', messages=[dict(role='user', content=prompt)],
                   max_tokens=tokens, temperature=0, seed=7319, stream=True,
                   stream_options={'include_usage': True}, cache_prompt=cache,
                   chat_template_kwargs={'enable_thinking': False})
    payload.update(overrides or {})
    req = urllib.request.Request(URL+'/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = None
    chunks = []
    answer = ''
    timings = {}
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            timings = chunk.get('timings') or timings
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                content = delta.get('content') or delta.get('reasoning_content') or ''
                if content and first is None:
                    first = time.monotonic()
                answer += content
    end = time.monotonic()
    n = usage.get('completion_tokens', 0)
    return dict(request=payload, answer=answer, timings=timings, usage=usage,
                elapsed_s=end-start, ttft_s=first-start if first else None,
                decode_tps=(n-1)/(end-first) if first and n>1 else None, chunks=chunks)

def validate(out):
    """Known-answer checks, including reuse after unrelated and long prompts."""
    checks = []
    a = 'Return only a JSON object with keys sum, product, sorted. sum is 17+25, product is 7*8, sorted is [9,1,4,1] sorted ascending.'
    for i, prompt in enumerate([a, 'Reply with exactly: BLUE HERON 913.', a, a]):
        result = request(prompt, 128, cache=True)
        answer = result['answer'].strip()
        if i == 1:
            passed = answer == 'BLUE HERON 913.'
        else:
            try:
                parsed = json.loads(answer.removeprefix('```json').removeprefix('```').removesuffix('```').strip())
                passed = parsed == {'sum':42, 'product':56, 'sorted':[1,1,4,9]}
            except ValueError:
                passed = False
        (out/f'check-{i}.json').write_text(json.dumps(result, indent=2))
        checks.append(dict(name=f'cached-{i}', passed=passed, answer=answer))
    filler = '\n'.join(f'Record {i:04d}: normal inventory entry; status pending; location warehouse north.' for i in range(900))
    prompt = ('Read this inventory and find the special access code.\n'+filler[:len(filler)//2]+
              '\nSPECIAL ACCESS CODE: HALO-MTP-7319\n'+filler[len(filler)//2:]+
              '\nReturn only the special access code, with no explanation.')
    result = request(prompt, 64, cache=True)
    (out/'check-long.json').write_text(json.dumps(result, indent=2))
    checks.append(dict(name='long-retrieval', passed=result['answer'].strip()=='HALO-MTP-7319',
                       answer=result['answer'], usage=result['usage'], timings=result['timings']))
    long_speed = request(filler+'\n\n'+PROMPTS[0][1], 512)
    (out/'long-throughput.json').write_text(json.dumps(long_speed, indent=2))
    print(json.dumps({'long_decode_tps':long_speed['decode_tps'], 'timings':long_speed['timings']}),flush=True)
    result = request(a, 128, cache=True)
    (out/'check-after-long.json').write_text(json.dumps(result, indent=2))
    try:
        parsed = json.loads(result['answer'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
        passed = parsed == {'sum':42, 'product':56, 'sorted':[1,1,4,9]}
    except ValueError:
        passed = False
    checks.append(dict(name='after-long', passed=passed, answer=result['answer']))
    tool = {'type':'function','function':{'name':'lookup_weather','description':'Look up weather for a city',
             'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}
    result = request('Use lookup_weather to get the weather in Rome.', 128,
                     overrides={'tools':[tool], 'tool_choice':{'type':'function','function':{'name':'lookup_weather'}}})
    (out/'check-tool.json').write_text(json.dumps(result, indent=2))
    calls = {}
    for chunk in result['chunks']:
        for choice in chunk.get('choices',[]):
            for call in choice.get('delta',{}).get('tool_calls',[]):
                item = calls.setdefault(call.get('index',0), {'name':'','arguments':''})
                for key in item:
                    item[key] += call.get('function',{}).get(key,'')
    passed = False
    for call in calls.values():
        try:
            passed |= call['name']=='lookup_weather' and json.loads(call['arguments'])=={'city':'Rome'}
        except ValueError:
            pass
    checks.append(dict(name='tool-call',passed=passed,calls=calls))
    for i in range(3):
        expected = {'code':f'MTP-{7319+i}', 'total':42+i}
        result = request('Return only this JSON object, unchanged: '+json.dumps(expected), 64,
                         cache=True, overrides={'temperature':0.7,'seed':42+i})
        (out/f'check-sampled-{i}.json').write_text(json.dumps(result,indent=2))
        try:
            parsed = json.loads(result['answer'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
            passed = parsed == expected
        except ValueError:
            passed = False
        checks.append(dict(name=f'sampled-{i}',passed=passed,answer=result['answer']))
    (out/'validation.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps({'checks':checks}),flush=True)
    if not all(c['passed'] for c in checks):
        raise RuntimeError('Correctness validation failed')

def run(args):
    if args.portable and args.sweep:
        raise ValueError('Portable runtime does not apply per-request draft limits; launch each setting separately')
    label = args.label or f'n{args.draft}-p{args.pmin}-{args.kv}-b{args.batch}-t{args.threads}-c{args.context}'
    out = ROOT/'results'/label
    out.mkdir(parents=True, exist_ok=True)
    cmd = ['podman', 'run', '--rm', '--name', NAME, '--network', 'host',
           '--device', '/dev/dri', '--group-add', 'keep-groups', '--security-opt', 'label=disable',
           '-v', '/home/lelloman/model-catalog:/home/lelloman/model-catalog:ro',
           '-v', '/home/lelloman/.cache/huggingface/hub:/home/lelloman/.cache/huggingface/hub:ro',
           '-v', f'{ROOT}:{ROOT}:ro', 'localhost/strix-qwen38-mtp:463c0f6', '--server',
           '-m', BASE, '--alias', 'qwen3.6-35b-a3b', '--host', '127.0.0.1', '--port', '18036',
           '--device', 'Vulkan0', '-ngl', '999', '--fit', 'off', '-c', str(args.context), '-np', '1',
           '-fa', 'on', '-ctk', args.kv, '-ctv', args.kv, '-b', str(args.batch), '-ub', str(args.batch),
           '-t', str(args.threads), '--jinja', '--reasoning', 'off', '--metrics']
    if args.no_cache:
        cmd += ['--cache-ram', '0', '--no-cache-prompt']
    if args.draft:
        cmd += ['-md', DRAFT.replace('q8_0.gguf', args.draft_quant+'.gguf'), '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(args.draft),
                '--spec-draft-ngl', 'all', '--spec-draft-device', 'Vulkan0',
                '--spec-draft-p-min', str(args.pmin), '--spec-draft-p-split', '0.1',
                '--spec-draft-type-k', args.kv, '--spec-draft-type-v', args.kv]
        if args.strict:
            cmd += ['--spec-mtp-strict-qwen']
    if args.portable:
        cmd = ['/home/lelloman/flash-next/runtime/vulkan/llama-server'] + cmd[cmd.index('--server')+1:]
    (out/'command.json').write_text(json.dumps(cmd, indent=2))
    with (out/'server.log').open('w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+300
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited: {out}/server.log')
                try:
                    with urllib.request.urlopen(URL+'/health', timeout=2) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError('Server startup timed out')
            warmup = request('Reply with exactly: Ready.', 32)
            (out/'warmup.json').write_text(json.dumps(warmup, indent=2))
            results = []
            settings = [(args.draft, args.pmin)]
            if args.sweep:
                settings = [(n,p) for n in [1,2,3,4,6] for p in [0.5,0.75,0.9]]
            for n,p in settings:
                for repeat in range(args.repeats):
                    for name, prompt in PROMPTS:
                        overrides = {'speculative.n_max': n, 'speculative.p_min': p} if args.draft else {}
                        result = request(prompt, args.tokens, overrides=overrides)
                        (out/f'n{n}-p{p}-{name}-{repeat}.json').write_text(json.dumps(result, indent=2))
                        brief = {k:result[k] for k in ['decode_tps','ttft_s','elapsed_s','timings','usage']}
                        brief.update(name=name, repeat=repeat, draft=n, pmin=p)
                        results.append(brief)
                        print(json.dumps(dict(label=label, **brief)), flush=True)
            summary = dict(label=label, args=vars(args), runs=results,
                           median_tps=statistics.median(r['decode_tps'] for r in results))
            (out/'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(dict(label=label, median_tps=summary['median_tps'])), flush=True)
            if args.validate:
                validate(out)
        finally:
            if args.portable:
                process.terminate()
            else:
                subprocess.run(['podman', 'stop', '-t', '15', NAME], stdout=subprocess.DEVNULL)
            process.wait(timeout=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', type=int, default=0)
    parser.add_argument('--draft-quant', default='q8_0')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--portable', action='store_true')
    parser.add_argument('--pmin', type=float, default=0.75)
    parser.add_argument('--kv', default='q4_0')
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--context', type=int, default=262144)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--label')
    run(parser.parse_args())
