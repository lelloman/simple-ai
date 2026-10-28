#!/usr/bin/env python3
"""Offline CUDA vLLM site summaries, with one bounded batch in flight."""
import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
PROMPT = ('Write 2–3 English sentences summarizing this website: who runs it, '
          'what it does, and where. Use facts only from the supplied website card. '
          'Keep names as written. If the card gives no place, omit the location. '
          'Never infer a location from the domain, TLD, language or names. '
          'Do not infer missing names. No preamble.')
MAX_BODY = 2 * 1024 * 1024


def check_checkpoint(path):
    """Reject accidental fp16/FP8 substitutions for the qualified INT8 recipe."""
    with open(os.path.join(path, 'config.json')) as handle:
        config = json.load(handle).get('quantization_config', {})
    groups = config.get('config_groups', {}).values()
    linear = [g for g in groups if 'Linear' in g.get('targets', [])]
    if (config.get('quant_method') != 'compressed-tensors' or not linear
            or any(any(g.get(k, {}).get('num_bits') != 8
                       or g.get(k, {}).get('type') != 'int'
                       for k in ('weights', 'input_activations')) for g in linear)):
        raise ValueError('expected the qualified compressed-tensors W8A8 INT8 checkpoint')


def validate(payload):
    if (not isinstance(payload, dict) or set(payload) - {'model', 'input', 'max_tokens'}
            or payload.get('model') != MODEL):
        raise ValueError('expected summary model and input')
    texts = payload.get('input')
    if (not isinstance(texts, list) or not 1 <= len(texts) <= 64
            or not all(isinstance(t, str) and t.strip() and len(t) <= 32768 for t in texts)
            or sum(len(t.encode('utf-8')) for t in texts) > 1048576):
        raise ValueError('input requires 1–64 nonblank texts, at most 32768 characters each and 1 MiB total UTF-8')
    tokens = payload.get('max_tokens')
    if tokens is None:
        tokens = 60
    if type(tokens) is not int or not 1 <= tokens <= 60:
        raise ValueError('max_tokens must be between 1 and 60')
    return texts, tokens


def prompts(tokenizer, texts, max_tokens):
    result = []
    for text in texts:
        ids = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': text}],
            tokenize=True, add_generation_prompt=True)
        if len(ids) + max_tokens > 1024:
            raise ValueError('input plus requested output exceeds 1024 model tokens')
        result.append({'prompt_token_ids': ids})
    return result


class Provider:
    def __init__(self, path, *, max_num_seqs=512, ngram=False):
        from vllm import LLM
        speculation = ({"method": "ngram", "num_speculative_tokens": 3,
                        "prompt_lookup_min": 2, "prompt_lookup_max": 5} if ngram else None)
        self.llm = LLM(model=path, dtype='bfloat16', trust_remote_code=False,
                       enable_prefix_caching=True, max_model_len=1024,
                       max_num_batched_tokens=16384, max_num_seqs=max_num_seqs,
                       speculative_config=speculation, distributed_executor_backend="uni",
                       gpu_memory_utilization=0.92)
        self.tokenizer = self.llm.get_tokenizer()
        self.lock = threading.Lock()

    def summarize(self, payload):
        texts, max_tokens = validate(payload)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('summary provider is busy')
        try:
            from vllm import SamplingParams
            inputs = prompts(self.tokenizer, texts, max_tokens)
            outputs = self.llm.generate(inputs, SamplingParams(temperature=0, max_tokens=max_tokens), use_tqdm=False)
            if len(outputs) != len(texts):
                raise RuntimeError('wrong number of vLLM outputs')
            data, prompt_tokens, completion_tokens = [], 0, 0
            for i, output in enumerate(outputs):
                if len(output.outputs) != 1 or not output.outputs[0].text.strip():
                    raise RuntimeError('empty or invalid vLLM output')
                data.append({'index': i, 'summary': output.outputs[0].text.strip()})
                prompt_tokens += len(output.prompt_token_ids)
                completion_tokens += len(output.outputs[0].token_ids)
            return {'model': MODEL, 'data': data, 'usage': {
                'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens}}
        finally:
            self.lock.release()


def serve(provider, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.reply(200 if self.path == '/health' else 404, {'model': MODEL})

        def do_POST(self):
            if self.path != '/v1/summaries':
                self.reply(404, {'error': 'unknown endpoint'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    raise ValueError('invalid body size')
                payload = json.loads(self.rfile.read(length))
                result = provider.summarize(payload)
            except (ValueError, UnicodeDecodeError) as error:
                self.reply(400, {'error': str(error)})
                return
            except BlockingIOError as error:
                self.reply(429, {'error': str(error)})
                return
            except Exception:
                logging.exception('Summary inference failed')
                self.reply(500, {'error': 'summary inference failed; see provider log'})
                return
            self.reply(200, result)

        def log_message(self, *_):
            pass
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=[MODEL])
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    # Never consult the service user's default Hugging Face token/cache.
    os.environ['HF_HOME'] = os.path.join(os.path.dirname(os.path.abspath(args.model_path)), 'cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    check_checkpoint(args.model_path)
    serve(Provider(args.model_path), args.port)


if __name__ == '__main__':
    main()
