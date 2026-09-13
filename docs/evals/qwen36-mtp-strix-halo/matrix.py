#!/usr/bin/env python3
"""Use fresh processes because the portable runtime ignores per-request limits."""
import json
import pathlib
import subprocess
import argparse

root = pathlib.Path('/home/lelloman/qwen36-mtp-test')
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['draft','hardware'], default='draft')
args = parser.parse_args()
if args.phase == 'draft':
    jobs = [(f'portable-n{n}-p{p}', ['--draft',str(n),'--pmin',str(p)])
            for n,p in [(0,.75),(1,.75),(2,.75),(4,.75),(6,.75),(3,.5),(3,.9)]]
else:
    jobs = [
        ('portable-n6-q4draft', ['--draft','6','--draft-quant','q4_0']),
        ('portable-n6-q8kv', ['--draft','6','--kv','q8_0']),
        ('portable-n6-t4', ['--draft','6','--threads','4']),
        ('portable-n6-b512', ['--draft','6','--batch','512']),
        ('portable-n8-p0.5', ['--draft','8','--pmin','0.5']),
    ]
for label, options in jobs:
    with (root/'results'/f'{label}.console.log').open('w') as log:
        subprocess.run(['python3',str(root/'benchmark.py'),'--portable',
                        *options,'--label',label],stdout=log,stderr=subprocess.STDOUT,check=True)
    summary = json.loads((root/'results'/label/'summary.json').read_text())
    print(json.dumps({'label':label,'median_tps':summary['median_tps'],
                      'runs':[{k:r[k] for k in ['name','decode_tps','timings']} for r in summary['runs']]}),flush=True)
