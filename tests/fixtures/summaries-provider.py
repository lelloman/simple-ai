"""Tiny managed provider fixture for Rust lifecycle tests; no ML dependencies."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--model');args,_=p.parse_known_args()
busy=threading.Event()
poisoned=threading.Event()
class Handler(BaseHTTPRequestHandler):
    def send(self,status,body):
        encoded=json.dumps(body).encode();self.send_response(status);self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
    def do_GET(self): self.send(200,{'busy':busy.is_set()})
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));busy.set();time.sleep(2 if data['input'] == ['slow'] else .3)
        if data['input'] == ['poison']: poisoned.set()
        status = 500 if poisoned.is_set() else 400 if data['input'] == ['invalid'] else 429 if data['input'] == ['busy'] else 200
        if status != 200:
            self.send(status, {'error': 'sticky inference failure' if poisoned.is_set() else 'caller error'});busy.clear();return
        self.send(200,{'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},'model':args.model,'data':[{'index':i,'summary':t} for i,t in enumerate(data['input'])]});busy.clear()
    def log_message(self,*_):pass
ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
