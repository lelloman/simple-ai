#!/usr/bin/env python3
"""Verify runner restart terminates its native child and can reload the alias."""
import json
import pathlib
import subprocess
import time
import urllib.request

children = []
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        argv = (entry/'cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeError):
        continue
    if argv[0] == '/home/lelloman/flash-next/runtime/vulkan/bin/llama-server':
        port = argv[argv.index('--port')+1]
        with urllib.request.urlopen('http://127.0.0.1:'+port+'/slots') as response:
            slots = json.load(response)
        assert not any(s['is_processing'] for s in slots), 'Wait until inference is idle'
        children.append(int(entry.name))
assert children, 'No native inference child found'
subprocess.run(['systemctl','--user','restart','simple-ai-runner'], check=True)
assert all(not pathlib.Path('/proc',str(pid)).exists() for pid in children), 'Inference child survived restart'
for attempt in range(30):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2) as response:
            assert response.status == 200
        break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError('Runner did not become healthy')
payload = {'model':'qwen3.6-35b-a3b','messages':[{'role':'user','content':'Reply with exactly: Ready.'}],
           'max_tokens':32,'temperature':0,'stream':False}
req = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',
                             data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=300) as response:
    answer = json.load(response)
assert answer['choices'][0]['message']['content'].strip()=='Ready.',answer
print(json.dumps({'terminated_child_pids':children,'reload':'passed'}))
