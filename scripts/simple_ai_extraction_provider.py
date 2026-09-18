#!/usr/bin/env python3
"""Managed GLiNER2.5 provider for /v1/extractions. CPU or CUDA, loopback only."""
from __future__ import annotations
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL_ID = "fastino/gliner2.5-multi-v1"
REVISION = "235cf92d6d4318da9bfca0d08975c8fa7250d13b"
MAX_BODY_BYTES = 2 * 1024 * 1024


def name(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 128


def labels(value):
    if not isinstance(value, (list, dict)) or not 1 <= len(value) <= 32:
        raise ValueError("labels require 1–32 names or descriptions")
    if not all(name(x) for x in value) or len(set(value)) != len(value):
        raise ValueError("labels require unique non-empty names of at most 128 characters")
    if isinstance(value, dict) and not all(isinstance(x, str) and x.strip() and len(x) <= 512 for x in value.values()):
        raise ValueError("invalid label descriptions")
    return len(value)


def validate(payload):
    if not isinstance(payload, dict) or set(payload) - {"model", "input", "schema", "threshold", "overlap", "splitter", "long_text", "chunk_size", "chunk_overlap"}:
        raise ValueError("invalid extraction request fields")
    if not name(payload.get("model")):
        raise ValueError("model is required")
    texts = payload.get("input")
    if isinstance(texts, str):
        texts = [texts]
    long_text = payload.get("long_text", False)
    if not isinstance(long_text, bool):
        raise ValueError("long_text must be boolean")
    limit = 100_000 if long_text else 1500
    if not isinstance(texts, list) or not 1 <= len(texts) <= 16 or not all(isinstance(x, str) and x.strip() and len(x) <= limit for x in texts):
        raise ValueError(f"input requires 1–16 non-empty texts of at most {limit} characters; enable long_text for documents")
    if sum(map(len, texts)) > 200_000:
        raise ValueError("input exceeds 200000 characters total")
    threshold = payload.get("threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if payload.get("overlap", "flat") not in {"flat", "nested", "allow", "longest"} or payload.get("splitter", "whitespace") not in {"whitespace", "char"}:
        raise ValueError("invalid overlap or splitter")
    size, overlap = payload.get("chunk_size", 256), payload.get("chunk_overlap", 64)
    if type(size) is not int or type(overlap) is not int or not 64 <= size <= 512 or not 0 <= overlap < size or (long_text and size - overlap < 32):
        raise ValueError("invalid chunk_size/chunk_overlap; minimum advance is 32 words")
    schema = payload.get("schema")
    if not isinstance(schema, dict) or set(schema) - {"entities", "classifications", "structures", "relations"} or len(json.dumps(schema, ensure_ascii=False, separators=(',', ':')).encode()) > 8192:
        raise ValueError("invalid schema or schema exceeds 8192 bytes")
    count = 0
    for key in ("entities", "relations"):
        if schema.get(key) is not None:
            count += labels(schema[key])
    outputs = {"entities", "relation_extraction"}
    tasks = schema.get("classifications", [])
    if not isinstance(tasks, list):
        raise ValueError("classifications must be a list")
    for task in tasks:
        if not isinstance(task, dict) or set(task) - {"task", "labels", "multi_label"} or not name(task.get("task")) or task["task"] in outputs or type(task.get("multi_label", False)) is not bool:
            raise ValueError("invalid or duplicate classification task")
        outputs.add(task["task"])
        count += labels(task.get("labels"))
    structures = schema.get("structures", {})
    if not isinstance(structures, dict):
        raise ValueError("structures must be an object")
    for key, record in structures.items():
        if not name(key) or key in outputs or not isinstance(record, dict) or set(record) - {"fields", "anchor"}:
            raise ValueError("invalid or duplicate structure")
        outputs.add(key)
        fields = record.get("fields")
        if not isinstance(fields, list) or not 1 <= len(fields) <= 16:
            raise ValueError("structure requires 1–16 fields")
        names = set()
        for field in fields:
            if not isinstance(field, dict) or set(field) - {"name", "dtype", "description", "cardinality"} or not name(field.get("name")) or field["name"] in names:
                raise ValueError("invalid or duplicate field")
            names.add(field["name"])
            if field.get("dtype", "str") not in {"str", "list"} or field.get("cardinality") not in {None, "optional_one", "required_one", "zero_or_more", "one_or_more"}:
                raise ValueError("invalid field type/cardinality")
            desc = field.get("description")
            if desc is not None and (not isinstance(desc, str) or not desc.strip() or len(desc) > 512):
                raise ValueError("invalid field description")
        if record.get("anchor") not in names:
            raise ValueError("anchor must name a declared field")
        count += len(fields)
    if not 1 <= count <= 64:
        raise ValueError("schema requires 1–64 labels/fields total")
    return texts


def build_schema(data, threshold):
    from gliner2.inference.schema import Schema
    schema = Schema()
    if data.get("entities") is not None:
        schema.entities(data["entities"])
    if data.get("relations") is not None:
        schema.relations(data["relations"])
    for task in data.get("classifications", []):
        schema.classification(task["task"], task["labels"], multi_label=task.get("multi_label", False), cls_threshold=threshold)
    for key, record in data.get("structures", {}).items():
        builder = schema.structure(key, mode="natural", anchor=record["anchor"])
        for field in record["fields"]:
            builder.field(field["name"], dtype=field.get("dtype", "str"), description=field.get("description"), cardinality=field.get("cardinality"))
    return schema


class Provider:
    def __init__(self, args):
        import torch
        from gliner2 import AutoExtractor
        from huggingface_hub import snapshot_download
        self.torch = torch
        self.model_id, self.revision = args.model, args.revision
        self.device, self.batch_size = args.device, args.batch_size
        self.lock = threading.Lock()
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        path = args.model_path or snapshot_download(args.model, revision=args.revision, allow_patterns=["*.json", "*.safetensors"])
        self.model = AutoExtractor.from_pretrained(path, map_location=args.device, quantize=args.device == "cuda")
        self.model.eval()

    def health(self):
        return {"status": "ok", "model": self.model_id, "revision": self.revision, "device": self.device}

    def extract(self, payload):
        texts = validate(payload)
        if payload["model"] != self.model_id:
            raise ValueError("model does not match loaded provider")
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("extraction provider is busy")
        try:
            self.model.set_word_splitter(payload.get("splitter", "whitespace"))
            threshold = payload.get("threshold", 0.5)
            schema = build_schema(payload["schema"], threshold)
            kwargs = dict(batch_size=self.batch_size, threshold=threshold, include_confidence=True, include_spans=True, overlap_policy=payload.get("overlap", "flat"))
            start = time.perf_counter()
            with self.torch.inference_mode():
                if payload.get("long_text", False):
                    result = self.model.batch_extract_long(texts, schema, chunk_size=payload.get("chunk_size", 256), chunk_overlap=payload.get("chunk_overlap", 64), **kwargs)
                else:
                    result = self.model.batch_extract(texts, schema, **kwargs)
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            return {"object": "list", "model": self.model_id, "revision": self.revision,
                    "data": [{"index": i, "result": r} for i, r in enumerate(result)],
                    "usage": {"input_count": len(texts), "input_characters": sum(map(len, texts))},
                    "inference_ms": round((time.perf_counter()-start)*1000, 3)}
        finally:
            self.lock.release()


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
        if self.path != "/v1/extractions":
            self._json(404, {"error": "not found"})
            return
        try:
            self.connection.settimeout(15)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("invalid request body size")
            payload = json.loads(self.rfile.read(length))
            self._json(200, self.provider.extract(payload))
        except (ValueError, TypeError, KeyError) as error:
            self._json(400, {"error": str(error)})
        except BlockingIOError as error:
            self._json(429, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self._json(500, {"error": str(error)})

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--model-path")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_threads < 1:
        parser.error("batch-size and num-threads must be positive")
    handler = type("ExtractionHandler", (Handler,), {"provider": Provider(args)})
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
