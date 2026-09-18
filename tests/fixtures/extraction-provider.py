"""Tiny managed provider fixture for Rust lifecycle tests; no ML dependencies."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--model');args,_=p.parse_known_args()
busy=threading.Event()
class Handler(BaseHTTPRequestHandler):
    def send(self,status,body):
        encoded=json.dumps(body).encode();self.send_response(status);self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
    def do_GET(self): self.send(200,{'busy':busy.is_set()})
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));busy.set();time.sleep(.3)
        self.send(200,{'object':'list','model':args.model,'revision':'fixture','data':[],'usage':{'input_count':1,'input_characters':len(data['input'])},'inference_ms':300});busy.clear()
    def log_message(self,*_):pass
ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
