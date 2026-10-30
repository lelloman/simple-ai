"""Managed concurrent tag provider fixture; no GPU or ML dependencies."""
import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--port', type=int)
args, _ = parser.parse_known_args()
child = subprocess.Popen(['sleep', '600'])
active = 0
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {'active': active, 'child': child.pid})

    def do_POST(self):
        global active
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert body['chat_template_kwargs']['enable_thinking'] is False
        assert body['response_format']['type'] == 'json_schema'
        text = body['messages'][0]['content']
        with lock:
            active += 1
        time.sleep(1 if text == 'slow' else .2)
        with lock:
            active -= 1
        if text == 'poison':
            self.reply(500, {'error': 'sticky CUDA failure'})
            return
        if text == 'invalid':
            self.reply(400, {'error': 'invalid schema'})
            return
        self.reply(200, {'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 1,
                         'model': args.model, 'choices': [{'index': 0, 'message': {
                             'role': 'assistant', 'content': '{"tag":"ok"}'}, 'finish_reason': 'stop'}]})

    def log_message(self, *_):
        pass


ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
