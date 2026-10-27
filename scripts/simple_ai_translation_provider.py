#!/usr/bin/env python3
"""Offline MADLAD provider; one inference request at a time, batched on device."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Copied with this provider by deploy-fleet; same map the gateway validates.
language_path = Path(__file__).with_name('translation-languages.json')
if not language_path.exists():
    language_path = Path(__file__).parents[1] / 'simple-ai-common/src/translation-languages.json'
LANGUAGES = json.loads(language_path.read_text())
MAX_BODY_BYTES = 2 * 1024 * 1024


def validate(payload):
    inputs = payload.get('input')
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 256:
        raise ValueError('input must contain between 1 and 256 texts')
    if any(not isinstance(t, str) or not t.strip() or len(t) > 4000 for t in inputs) or sum(map(len, inputs)) > 100_000:
        raise ValueError('input texts must be non-empty, at most 4000 characters each and 100000 total')
    if payload.get('target') not in LANGUAGES:
        raise ValueError('unknown target language')
    if payload.get('source') is not None and payload['source'] not in LANGUAGES:
        raise ValueError('unknown source language')
    limit = payload.get('max_tokens')
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 512):
        raise ValueError('max_tokens must be between 1 and 512')
    return inputs, LANGUAGES[payload['target']], limit or 256


def translate_batches(inputs, target, limit, batch_size, generate):
    data = []
    for start in range(0, len(inputs), batch_size):
        texts = [f'<2{target}> {text}' for text in inputs[start:start + batch_size]]
        translations = generate(texts, limit)
        if len(translations) != len(texts):
            raise RuntimeError('translation batch length mismatch')
        data.extend({'index': start + i, 'translation': text} for i, text in enumerate(translations))
    return data


class Provider:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(args.num_threads)
        device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
        if device == 'auto':
            device = 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('GPU requested but unavailable (ROCm also uses cuda device API)')
        self.device = device
        self.model_id = args.model
        self.batch_size = args.batch_size
        self.lock = threading.Lock()
        path = args.model_path or args.model
        self.tokenizer = AutoTokenizer.from_pretrained(path, revision=args.revision, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            path, revision=args.revision, local_files_only=True,
            torch_dtype=torch.float32,  # T5 fp16 can overflow; bfloat16 on GPU.
        ).to(device=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32).eval()
        logging.info('translation ready model=%s device=%s batch_size=%s', args.model, device, args.batch_size)

    def generate(self, texts, limit):
        encoded = self.tokenizer(texts, padding=True, return_tensors='pt')
        # Reject instead of silently truncating source text.
        if encoded['input_ids'].shape[1] > 1024:
            raise ValueError('input exceeds 1024 model tokens including target prefix')
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, max_new_tokens=limit, do_sample=False, num_beams=1)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def translate(self, payload):
        inputs, target, limit = validate(payload)
        return {'model': self.model_id, 'data': translate_batches(inputs, target, limit, self.batch_size, self.generate)}


class Handler(BaseHTTPRequestHandler):
    provider: Provider

    def reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200 if self.path == '/health' else 404,
                   {'model': self.provider.model_id, 'device': self.provider.device})

    def do_POST(self):
        if self.path != '/v1/translations':
            self.reply(404, {'error': 'not found'})
            return
        if not self.provider.lock.acquire(blocking=False):
            self.reply(429, {'error': 'translation provider busy'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError('invalid request body size')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('request must be an object')
            self.reply(HTTPStatus.OK, self.provider.translate(payload))
        except ValueError as error:
            self.reply(400, {'error': str(error)})
        except Exception:
            error_id = uuid.uuid4().hex
            logging.exception('translation request failed error_id=%s', error_id)
            self.reply(500, {'error': f'translation inference failed error_id={error_id}'})
        finally:
            self.provider.lock.release()

    def log_message(self, *_):
        pass


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--model-path')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-threads', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_threads <= 0:
        parser.error('batch size and threads must be positive')
    provider = Provider(args)
    handler = type('TranslationHandler', (Handler,), {'provider': provider})
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == '__main__':
    main()
