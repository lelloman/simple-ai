#!/usr/bin/env python3
"""Real gateway -> runner -> GLiNER smoke test, isolated ports/config/database.
Requires system Python PyJWT+cryptography and a prepared extraction environment.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer

ROOT=Path(__file__).resolve().parents[1]
MODEL='fastino/gliner2.5-multi-v1'
p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--model-path',required=True);p.add_argument('--language-model',required=True);args=p.parse_args()
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
api,runner,oidc=port(),port(),port()
def call(path,data=None,token=None):
    headers={'Content-Type':'application/json'}
    if token:headers['Authorization']='Bearer '+token
    request=urllib.request.Request(f'http://127.0.0.1:{api}'+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
    with urllib.request.urlopen(request,timeout=90) as r:return json.load(r)
spec=importlib.util.spec_from_file_location('oidc',ROOT/'tests/e2e/mock-oidc/server.py');mock=importlib.util.module_from_spec(spec);spec.loader.exec_module(mock)
mock.ISSUER=f'http://127.0.0.1:{oidc}';mock.DISCOVERY={'issuer':mock.ISSUER,'jwks_uri':mock.ISSUER+'/.well-known/jwks.json'}
server=HTTPServer(('127.0.0.1',oidc),mock.OIDCHandler);threading.Thread(target=server.serve_forever,daemon=True).start()
procs=[]
with tempfile.TemporaryDirectory(prefix='gliner-e2e-') as tmp:
    tmp=Path(tmp);(tmp/'gateway').mkdir();(tmp/'runner').mkdir()
    (tmp/'gateway/config.toml').write_text(f'''host="127.0.0.1"
port={api}
[oidc]
issuer="{mock.ISSUER}"
audience="test-audience"
[database]
url="sqlite:{tmp}/audit.db"
[language]
model_path="{args.language_model}"
[gateway]
enabled=true
auth_token="isolated-extraction-test"
auto_wake_enabled=false
[models]
information_extraction=["{MODEL}"]
''')
    (tmp/'runner/config.toml').write_text(f'''[runner]
id="extraction-integration"
name="Extraction integration"
[api]
host="127.0.0.1"
port={runner}
[gateway]
ws_url="ws://127.0.0.1:{api}/ws/runners"
auth_token="isolated-extraction-test"
heartbeat_interval_secs=1
[engines.ollama]
enabled=false
[engines.extraction]
enabled=true
command=["{args.python}","{ROOT}/scripts/simple_ai_extraction_provider.py"]
model_path="{args.model_path}"
''')
    try:
        for name,binary in [('gateway','simple-ai-backend'),('runner','simple-ai-runner')]:
            log=open(tmp/f'{name}.log','w')
            procs.append(subprocess.Popen([ROOT/'target/release'/binary],cwd=tmp/name,stdout=log,stderr=subprocess.STDOUT,start_new_session=True));log.close()
            if name=='gateway':
                for _ in range(100):
                    try:call('/health');break
                    except OSError:time.sleep(.1)
        req=urllib.request.Request(mock.ISSUER+'/token',data=json.dumps({'roles':['model:specific']}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as r:token=json.load(r)['token']
        for _ in range(100):
            models=call('/v1/models',token=token)
            if MODEL in json.dumps(models):break
            time.sleep(.1)
        body={'model':'class:information_extraction','input':['Giulia lavora a Milano.','Zoë works in Zürich.'],'schema':{'entities':['person','location']}}
        for auth in [None,'sk-invalid']:
            try:call('/v1/extractions',body,auth);raise AssertionError('unauthenticated request accepted')
            except urllib.error.HTTPError as e:assert e.code==401,e.code
        result=call('/v1/extractions',body,token)
        assert result['model']==MODEL,result
        assert len(result['data'])==2,result
        for text,item in zip(body['input'],result['data']):
            for values in item['result']['entities'].values():
                for span in values:assert text[span['start']:span['end']]==span['text'],span
        assert result['data'][0]['result']['entities']['person'][0]['text']=='Giulia',result
        bad=dict(body,threshold=2)
        try:call('/v1/extractions',bad,token);raise AssertionError('bad threshold accepted')
        except urllib.error.HTTPError as e:assert e.code==400,e.code
        print('PASS: discovery, auth rejection, class routing, lazy load, batch inference, Unicode spans, validation')
    except Exception:
        for name in ['gateway','runner']:print((tmp/f'{name}.log').read_text()[-8000:])
        raise
    finally:
        for proc in reversed(procs):
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            proc.wait(timeout=10)
        server.shutdown()
