#!/usr/bin/env python3
"""Long-lived Hugging Face NLI provider for Simple AI.

The inference runner starts one process per loaded model and proxies
`POST /v1/classifications` to its loopback HTTP server.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MAX_BODY_BYTES = 64 * 1024 * 1024


def _label_id(config: Any, wanted: str, *, required: bool = True) -> int | None:
    label2id = {
        str(label).strip().lower(): int(index)
        for label, index in dict(getattr(config, "label2id", {}) or {}).items()
    }
    if wanted in label2id:
        return label2id[wanted]
    for label, index in label2id.items():
        if wanted in label:
            return index
    id2label = {
        int(index): str(label).strip().lower()
        for index, label in dict(getattr(config, "id2label", {}) or {}).items()
    }
    for index, label in id2label.items():
        if label == wanted:
            return index
    for index, label in id2label.items():
        if wanted in label:
            return index
    if not required:
        return None
    raise ValueError(f"model does not declare an NLI {wanted!r} label")


class Provider:
    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available() and args.device == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
        self.model_id = args.model
        self.device = torch.device(args.device)
        self.batch_size = max(1, args.batch_size)
        self.max_length = max(8, args.max_length)
        self.loaded_at = time.time()
        self.lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        declared_limit = int(
            getattr(self.model.config, "max_position_embeddings", 0)
            or getattr(self.tokenizer, "model_max_length", 0)
            or self.max_length
        )
        if 0 < declared_limit < 1_000_000:
            self.max_length = min(self.max_length, declared_limit)
        self.entailment_id = _label_id(self.model.config, "entailment")
        self.neutral_id = _label_id(self.model.config, "neutral", required=False)
        self.contradiction_id = _label_id(self.model.config, "contradiction", required=False)
        if self.contradiction_id is None:
            self.contradiction_id = _label_id(self.model.config, "not_entailment")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_id,
            "device": str(self.device),
            "batchSize": self.batch_size,
            "maxLength": self.max_length,
            "loadedAt": self.loaded_at,
        }

    def classify(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_input = payload.get("input")
        inputs = [raw_input] if isinstance(raw_input, str) else raw_input
        labels = payload.get("labels")
        if not isinstance(inputs, list) or not inputs or not all(isinstance(x, str) and x.strip() for x in inputs):
            raise ValueError("input must be a non-empty string or array of non-empty strings")
        if not isinstance(labels, list) or not labels:
            raise ValueError("labels must be a non-empty array")
        for label in labels:
            if not isinstance(label, dict) or not str(label.get("label", "")).strip() or not str(label.get("hypothesis", "")).strip():
                raise ValueError("each label requires non-empty label and hypothesis strings")

        premises: list[str] = []
        hypotheses: list[str] = []
        for text in inputs:
            for label in labels:
                premises.append(text)
                hypotheses.append(str(label["hypothesis"]))

        rows: list[list[float]] = []
        prompt_tokens = 0
        with self.lock, torch.inference_mode():
            for start in range(0, len(premises), self.batch_size):
                encoded = self.tokenizer(
                    premises[start : start + self.batch_size],
                    hypotheses[start : start + self.batch_size],
                    padding=True,
                    truncation="only_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                prompt_tokens += int(encoded["attention_mask"].sum().item())
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                probabilities = self.model(**encoded).logits.float().softmax(dim=-1).cpu()
                rows.extend(probabilities.tolist())

        data = []
        width = len(labels)
        for input_index in range(len(inputs)):
            scores = []
            for label_index, label in enumerate(labels):
                probability = rows[input_index * width + label_index]
                values = (
                    probability[self.entailment_id],
                    probability[self.neutral_id] if self.neutral_id is not None else 0.0,
                    probability[self.contradiction_id],
                )
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError("model returned a non-finite probability")
                scores.append(
                    {
                        "label": str(label["label"]),
                        "entailment": values[0],
                        "neutral": values[1],
                        "contradiction": values[2],
                    }
                )
            data.append({"index": input_index, "scores": scores})

        return {
            "object": "list",
            "data": data,
            "model": self.model_id,
            "usage": {
                "input_count": len(inputs),
                "pair_count": len(rows),
                "prompt_tokens": prompt_tokens,
            },
        }


class Handler(BaseHTTPRequestHandler):
    provider: Provider

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/health":
            self._json(HTTPStatus.OK, self.provider.health())
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/v1/classifications":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("invalid request body size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            self._json(HTTPStatus.OK, self.provider.classify(payload))
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - HTTP error boundary
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    provider = Provider(args)
    handler = type("ClassificationHandler", (Handler,), {"provider": provider})
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
