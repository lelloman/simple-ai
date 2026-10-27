"""Managed embedding fixture: controlled valid and malformed results."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()
busy = threading.Event()


class Handler(BaseHTTPRequestHandler):
    def send(self, body):
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        self.send({"busy": busy.is_set()})

    def do_POST(self):
        assert self.path == "/v1/embeddings"
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        busy.set()
        time.sleep(.3)
        rows = []
        for i, text in enumerate(data["input"]):
            vector = [0.] * (3 if text == "bad-dimension" else 1024)
            vector[i] = 2. if text == "bad-norm" else 1.
            rows.append({"index": 10 if text == "bad-index" else i, "embedding": vector})
        self.send({"data": rows})
        busy.clear()

    def log_message(self, *_):
        pass


ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
