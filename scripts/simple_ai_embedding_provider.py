#!/usr/bin/env python3
"""Offline CUDA Qwen3 embeddings. Raw texts; no implicit instruction prefix."""
from __future__ import annotations
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MAX_BODY_BYTES = 2 * 1024 * 1024


def validate(payload):
    if not isinstance(payload, dict) or set(payload) - {"model", "input"} or payload.get("model") != MODEL_ID:
        raise ValueError("expected Qwen embedding model and input")
    texts = payload.get("input")
    if isinstance(texts, str):
        texts = [texts]
    if (not isinstance(texts, list) or not 1 <= len(texts) <= 256
            or not all(isinstance(x, str) and x.strip() and len(x) <= 32768 for x in texts)
            or sum(len(x.encode("utf-8")) for x in texts) > 1048576):
        raise ValueError("input requires 1–256 non-empty texts, at most 32768 characters each and 1 MiB total UTF-8")
    return texts


def last_token_pool(hidden, attention_mask):
    # Left padding means the last column is the final real token for every row.
    if not attention_mask[:, -1].all():
        raise RuntimeError("embedding tokenizer must use left padding")
    return hidden[:, -1]


def vectors(torch, hidden, attention_mask):
    pooled = last_token_pool(hidden, attention_mask).float()
    if pooled.shape[1] != 1024 or not torch.isfinite(pooled).all():
        raise RuntimeError("invalid embedding dimensions or non-finite values")
    norms = pooled.norm(p=2, dim=1, keepdim=True)
    if (norms <= 0).any():
        raise RuntimeError("zero embedding")
    return (pooled / norms).tolist()


class Provider:
    def __init__(self, args):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import snapshot_download
        if args.model != MODEL_ID or args.revision != REVISION or args.device != "cuda":
            raise ValueError("only the pinned Qwen CUDA model is supported")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.torch, self.batch_size = torch, args.batch_size
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(8)
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
        path = args.model_path or snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", local_files_only=True)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float16,
                                              attn_implementation="sdpa", local_files_only=True).to("cuda").eval()

    def health(self):
        return {"status": "ok", "model": MODEL_ID, "revision": REVISION, "device": "cuda"}

    def embed(self, payload):
        texts = validate(payload)
        if not self.slots.acquire(blocking=False):
            raise BlockingIOError("embedding provider queue is full")
        try:
            # Keep tokenizer/model access serialized; concurrent callers wait without GPU overlap.
            with self.lock, self.torch.inference_mode():
                data, tokens = [], 0
                for start in range(0, len(texts), self.batch_size):
                    encoded = self.tokenizer(texts[start:start+self.batch_size], padding=True,
                                             truncation=True, max_length=512, return_tensors="pt").to("cuda")
                    tokens += int(encoded["attention_mask"].sum().item())
                    hidden = self.model(**encoded).last_hidden_state
                    for vector in vectors(self.torch, hidden, encoded["attention_mask"]):
                        data.append({"object": "embedding", "index": len(data), "embedding": vector})
                return {"object": "list", "model": MODEL_ID, "data": data,
                        "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    provider: Provider
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(200, self.provider.health()) if self.path == "/health" else self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/embeddings":
            self._json(404, {"error": "not found"})
            return
        try:
            self.connection.settimeout(15)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("invalid request body size")
            payload = json.loads(self.rfile.read(length))
            self._json(200, self.provider.embed(payload))
        except (ValueError, TypeError, KeyError) as error:
            self._json(400, {"error": str(error)})
        except BlockingIOError as error:
            self._json(429, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(error)})

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--model-path")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 256 or args.num_threads < 1:
        parser.error("batch-size and num-threads must be positive")
    handler = type("EmbeddingHandler", (Handler,), {"provider": Provider(args)})
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
