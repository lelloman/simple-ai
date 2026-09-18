#!/usr/bin/env python3
"""Compare the same extraction workloads on CPU/CUDA. No external API required."""
import argparse
import json
import platform
import statistics
import time
import torch
from simple_ai_extraction_provider import Provider, MODEL_ID, REVISION

parser=argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
parser.add_argument('--model-path', required=True)
parser.add_argument('--output', required=True)
args=parser.parse_args()
args.model=MODEL_ID; args.revision=REVISION; args.num_threads=8; args.batch_size=4
start=time.perf_counter();p=Provider(args);load=time.perf_counter()-start
short='Giulia Rossi lavora per Acme a Milano. Il 15 settembre ha incontrato Marco Bianchi a Roma.'
medium=('Il progetto continua e il gruppo discute i prossimi passi. '*24)+short
schema={'entities':['person','organization','location','date']}
results=[]
for name,texts,long in [('short',[short],False),('medium',[medium],True),('batch16',[short]*16,False)]:
    request={'model':MODEL_ID,'input':texts,'schema':schema,'long_text':long}
    for _ in range(2):p.extract(request)
    values=[p.extract(request) for _ in range(10)]
    times=[v['inference_ms'] for v in values]
    results.append({'workload':name,'input_count':len(texts),'characters':sum(map(len,texts)),'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times),'documents_per_second':len(texts)*1000/statistics.median(times),'times_ms':times,'sample':values[-1]['data'][0]})
report={'host':platform.node(),'device':args.device,'torch':torch.__version__,'revision':REVISION,'threads':8,'batch_size':4,'load_seconds':load,'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else None,'results':results}
with open(args.output,'w') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({r['workload']:r['median_ms'] for r in results}))
