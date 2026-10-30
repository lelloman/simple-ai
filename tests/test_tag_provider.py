"""CPU checks of the vLLM server launch contract; GPU acceptance is operator-owned."""
import importlib.util
import pathlib
import sys
import json
import os
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('tag_provider', SCRIPTS / 'simple_ai_tag_provider.py')
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)


class LaunchContract(unittest.TestCase):
    def test_loopback_async_openai_server_uses_configured_checkpoint(self):
        args = provider.server_args('tag-model', '/local/W8A8', 1234)
        self.assertEqual(args[0], 'vllm.entrypoints.openai.api_server')
        for flag, value in [('--model', '/local/W8A8'), ('--served-model-name', 'tag-model'),
                            ('--host', '127.0.0.1'), ('--port', '1234'),
                            ('--max-num-seqs', '32'), ('--max-model-len', '2048'),
                            ('--distributed-executor-backend', 'uni')]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('--enable-prefix-caching', args)

    def test_live_probe_uses_32_requests_and_keeps_aligned_outputs(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert body['model'] == 'class:tag'
                assert body['response_format']['type'] == 'json_schema'
                output = json.dumps({'choices': [{'message': {'content': '{"tag":"ok"}'}}]}).encode()
                self.send_response(200)
                self.send_header('X-Runner-Id', 'gpu-server')
                self.send_header('Content-Length', str(len(output)))
                self.end_headers()
                self.wfile.write(output)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.request_queue_size = 64
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                source = pathlib.Path(directory) / 'input.jsonl'
                output = pathlib.Path(directory) / 'output.json'
                source.write_text(''.join(json.dumps({'id': str(i), 'request': {
                    'messages': [{'role': 'user', 'content': 'site'}],
                    'response_format': {'type': 'json_schema', 'json_schema': {'name': 'tags', 'schema': {'type': 'object'}}}
                }}) + '\n' for i in range(32)))
                result = subprocess.run([sys.executable, str(SCRIPTS / 'check-tags-live.py'),
                                         '--input', str(source), '--output', str(output)],
                                        env=dict(os.environ, SIMPLE_AI_URL='http://127.0.0.1:' + str(server.server_port),
                                                 SIMPLE_AI_API_KEY='fixture'), capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                report = json.loads(output.read_text())
                self.assertEqual(report['concurrency'], 32)
                self.assertEqual([row['id'] for row in report['outputs']], [str(i) for i in range(32)])
                self.assertTrue(all(row['runner_id'] == 'gpu-server' for row in report['outputs']))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
