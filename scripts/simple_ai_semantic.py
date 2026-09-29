#!/usr/bin/env python3
"""JEV-9B semantic decisions through vLLM and the trained decision-head adapter.

Uses the pinned upstream bare-v1 template, head bias and per-kind temperature.
No generated text is consumed; scores are not validated on application data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
from uuid import uuid4
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "autotrust/JEV-9B"
REVISION = "4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35"
MAX_BODY = 2 * 1024 * 1024


class BackendError(RuntimeError):
    pass


class BusyError(RuntimeError):
    pass


def distribution(values: list[float]) -> list[float]:
    if not values or any(math.isnan(v) or v == math.inf for v in values):
        raise BackendError("invalid label logprobs")
    peak = max(values)
    if not math.isfinite(peak):
        raise BackendError("all label probabilities are zero")
    weights = [math.exp(v - peak) for v in values]
    total = sum(weights)
    return [v / total for v in weights]


class VLLM:
    def __init__(self, url: str = "http://127.0.0.1:30000", timeout: float = 180):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def request(self, path: str, payload: dict | None = None, *, timeout: float | None = None,
                parse_json: bool = True) -> Any:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.url + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                data = response.read()
                return json.loads(data) if data and parse_json else None
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise BackendError(f"vLLM {path} failed: {exc}") from exc

    def generate(self, ids: list[int], labels: list[int]) -> dict:
        data = self.request("/v1/completions", {
            "model": "jev-decision", "prompt": ids, "max_tokens": 1,
            "temperature": 1.0, "logprobs": len(labels), "allowed_token_ids": labels,
            "add_special_tokens": False, "return_tokens_as_token_ids": True,
        })
        return parse_readout(data, labels)

    def flush(self) -> None:
        """Benchmark only: requires exclusive ownership of the backend."""
        if self.request("/reset_prefix_cache", {}).get("success") is not True:
            raise BackendError("vLLM could not reset prefix cache; benchmark requires an idle backend")

    def verify_model(self, model: str, revision: str) -> None:
        models = {row["id"]: row for row in self.request("/v1/models")["data"]}
        if model not in models or "jev-decision" not in models:
            raise BackendError("vLLM must serve JEV and its jev-decision adapter")
        root = models[model].get("root", "")
        adapter = models["jev-decision"].get("root", "")
        if revision not in root or revision not in adapter or not adapter.endswith("/adapter_vllm"):
            raise BackendError("vLLM model/adapter paths must identify the pinned snapshot revision")


def parse_readout(data: dict, labels: list[int]) -> dict:
    try:
        choices = data["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] not in ("length", "stop"):
            raise ValueError("expected one completed decision")
        positions = choices[0]["logprobs"]["top_logprobs"]
        if len(positions) != 1 or data["usage"]["completion_tokens"] != 1:
            raise ValueError("expected exactly one readout position")
        lookup = {}
        for key, value in positions[0].items():
            # vLLM return_tokens_as_token_ids uses token_id:<integer> keys.
            token = int(key.removeprefix("token_id:"))
            if token in lookup or value is None or isinstance(value, bool) or float(value) > 0:
                raise ValueError("duplicate or invalid logprob")
            lookup[token] = float(value)
        values = [lookup[token] for token in labels]
        distribution(values)
        usage = data["usage"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return {"logprobs": values, "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": 1, "cached_tokens": int(cached) if cached is not None else None}
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
        raise BackendError(f"malformed vLLM readout: {exc}") from exc


@dataclass
class Prepared:
    prefix: list[int]
    branches: list[list[int]]
    label_ids: list[int]
    keys: list[str]
    biases: list[float]
    temperature: float


class SemanticScorer:
    def __init__(self, tokenizer: Any, backend: VLLM, head: dict, temperatures: dict, *,
                 context: int = 18432, concurrency: int = 2,
                 model: str = MODEL, revision: str = REVISION):
        if context < 128 or concurrency < 1:
            raise ValueError("invalid context or concurrency")
        self.tokenizer, self.backend = tokenizer, backend
        self.head, self.temperatures = head, temperatures
        self.context, self.concurrency = context, concurrency
        self.model, self.revision = model, revision
        self.admission = threading.Lock()
        if head["slots"]["template_version"] != "bare-v1":
            raise ValueError("unsupported JEV template")
        if head["slots"]["ranges"] != {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}:
            raise ValueError("unexpected JEV decision-head slots")
        if len(head["bias"]) != 24 or len(head["verbalizer_ids"]) != 24:
            raise ValueError("expected 24 decision-head slots")
        if not all(math.isfinite(v) for v in head["bias"]):
            raise ValueError("nonfinite decision-head bias")
        for kind in ("noul", "choice", "score"):
            if not math.isfinite(temperatures[kind]) or temperatures[kind] <= 0:
                raise ValueError("invalid decision-head temperature")
        words = ["false", "true"] + [str(i) for i in range(6)] + list("ABCDEFGHIJKLMNOP")
        for word, token in zip(words, head["verbalizer_ids"]):
            if tokenizer.encode(word, add_special_tokens=False) != [token]:
                raise ValueError("tokenizer does not match decision-head verbalizers")

    @staticmethod
    def strings(values: Any, name: str, low: int, high: int) -> list[str]:
        if not isinstance(values, list) or not low <= len(values) <= high:
            raise ValueError(f"{name} requires {low}..{high} strings")
        if not all(isinstance(v, str) and v.strip() and len(v) <= 8192 for v in values):
            raise ValueError(f"{name} requires nonempty strings of at most 8192 characters")
        return values

    def prepare(self, state: Any, questions: list[str], keys: list[str], *, kind="noul") -> Prepared:
        if not isinstance(state, (str, dict, list)) or not state:
            raise ValueError("state must be a nonempty string, object or array")
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False)
        if not state_text.strip() or len(state_text.encode()) > MAX_BODY:
            raise ValueError("state must be nonempty and at most 2 MiB")
        if kind == "noul":
            if keys != ["False", "True"]:
                raise ValueError("noul keys must be False, True in trained order")
            options = ["false", "true"]
        elif kind == "choice":
            self.strings(keys, "options", 2, 16)
            options = [f"{chr(65+i)}) {key}" for i, key in enumerate(keys)]
        else:
            raise ValueError("unsupported decision kind")
        self.strings(questions, "questions", 1, 128)
        prefix_text = f"[kind] {kind}\n[state] {state_text}\n[question] "
        branches = []
        for question in questions:
            prompt = prefix_text + question + "\n[options]\n" + "\n".join(options) + "\n[decision]:"
            # Tokenize the whole trained template; never splice token boundaries.
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + 1 > self.context:
                raise ValueError(f"branch needs {len(ids)+1} tokens; context limit is {self.context}; no truncation performed")
            branches.append(ids)
        if sum(map(len, branches)) > 4_194_304:
            raise ValueError("submitted token budget exceeded")
        start = self.head["slots"]["ranges"][kind][0]
        end = start + len(keys)
        return Prepared(self.tokenizer.encode(prefix_text, add_special_tokens=False), branches,
                        self.head["verbalizer_ids"][start:end], keys,
                        self.head["bias"][start:end], self.temperatures[kind])

    def run(self, state: Any, questions: list[str], keys: list[str], *, warm: bool = False,
            kind: str = "noul") -> dict:
        # Prefix reuse is automatic in vLLM. No separate prefix-only generation.
        if not self.admission.acquire(blocking=False):
            raise BusyError("semantic evaluator busy; retry later")
        try:
            start = time.perf_counter()
            prepared = self.prepare(state, questions, keys, kind=kind)
            compiled = time.perf_counter()
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                reads = list(pool.map(lambda ids: self.backend.generate(ids, prepared.label_ids), prepared.branches))
            finished = time.perf_counter()
            rows = []
            for read in reads:
                scores = distribution([(lp+bias)/prepared.temperature
                                       for lp, bias in zip(read["logprobs"], prepared.biases)])
                rows.append({"scores": dict(zip(keys, scores)),
                             "raw_logprobs": dict(zip(keys, read["logprobs"])),
                             "selected": keys[max(range(len(keys)), key=scores.__getitem__)]})
            hits = [r["cached_tokens"] for r in reads]
            return {"results": rows, "calibrated": False, "model": self.model, "revision": self.revision,
                    "upstream_temperature_applied": True,
                    "usage": {"prefix_tokens": len(prepared.prefix), "warmup_calls": 0,
                              "branch_calls": len(reads), "branch_cached_tokens": hits,
                              "cached_tokens": sum(hits) if all(v is not None for v in hits) else None,
                              "prompt_tokens": sum(r["prompt_tokens"] for r in reads),
                              "completion_tokens": sum(r["completion_tokens"] for r in reads)},
                    "timing_ms": {"prepare": (compiled-start)*1000, "prefill": 0,
                                  "branches": (finished-compiled)*1000, "total": (finished-start)*1000}}
        finally:
            self.admission.release()

    def decisions(self, payload: dict) -> dict:
        validate_decisions(payload)
        if not self.admission.acquire(blocking=False):
            raise BusyError("decision evaluator busy; retry later")
        try:
            started = time.perf_counter()
            prepared = []
            for q in payload["questions"]:
                kind = {"boolean": "noul", "choice": "choice", "rating": "score"}[q["type"]]
                instruction = q["instruction"]
                if kind == "noul":
                    keys, options = ["false", "true"], ["false", "true"]
                elif kind == "choice":
                    keys = [o["id"] for o in q["options"]]
                    options = [f"{chr(65+i)}) {o['description']}" for i,o in enumerate(q["options"])]
                else:
                    keys = options = [str(i) for i in range(6)]
                    instruction += "\nRating scale:\n" + "\n".join(f"{i}: {level}" for i,level in enumerate(q["levels"]))
                state = payload["state"]
                state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False)
                prompt = f"[kind] {kind}\n[state] {state_text}\n[question] {instruction}\n[options]\n" + "\n".join(options) + "\n[decision]:"
                ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids)+1 > self.context:
                    raise ValueError(f"question {q['id']} exceeds {self.context} token context; no truncation performed")
                start = self.head["slots"]["ranges"][kind][0]
                prepared.append((ids, self.head["verbalizer_ids"][start:start+len(keys)],
                                 keys, self.head["bias"][start:start+len(keys)], self.temperatures[kind]))
            if sum(len(p[0]) for p in prepared) > 4_194_304:
                raise ValueError("submitted token budget exceeded")
            compiled = time.perf_counter()
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                # Exiting the pool drains all sibling work even if one branch fails.
                reads = list(pool.map(lambda p: self.backend.generate(p[0], p[1]), prepared))
            answers = []
            for q,p,read in zip(payload["questions"],prepared,reads):
                scores = distribution([(lp+b)/p[4] for lp,b in zip(read["logprobs"],p[3])])
                selected = p[2][max(range(len(scores)), key=scores.__getitem__)]
                answer = {"id":q["id"], "type":q["type"], "probabilities":dict(zip(p[2],scores))}
                if q["type"] == "boolean": answer["value"] = selected == "true"
                elif q["type"] == "choice": answer["selected"] = selected
                else: answer.update(value=sum(i*v for i,v in enumerate(scores)), levels=q["levels"])
                answers.append(answer)
            ended = time.perf_counter()
            hits = [r["cached_tokens"] for r in reads]
            return {"request_id":str(uuid4()), "model":self.model, "revision":self.revision,
                    "answers":answers, "calibrated":False, "upstream_temperature_applied":True,
                    "usage":{"question_count":len(answers), "prompt_tokens":sum(r["prompt_tokens"] for r in reads),
                             "completion_tokens":len(reads), "cached_tokens":sum(hits) if all(h is not None for h in hits) else None},
                    "timing":{"prepare_ms":(compiled-started)*1000, "inference_ms":(ended-compiled)*1000,
                              "total_ms":(ended-started)*1000}}
        finally:
            self.admission.release()

    def semantic_score_many(self, state: Any, predicates: list[str]) -> dict:
        self.strings(predicates, "predicates", 1, 128)
        result = self.run(state, predicates, ["False", "True"])
        for predicate, row in zip(predicates, result["results"]):
            row.update(predicate=predicate, score=row["scores"]["True"])
        return result

    def semantic_score(self, state: Any, predicate: str) -> dict:
        result = self.semantic_score_many(state, [predicate])
        result.update(result.pop("results")[0])
        return result

    def semantic_classify(self, state: Any, options: list[str]) -> dict:
        self.strings(options, "options", 2, 16)
        if len(set(options)) != len(options):
            raise ValueError("options must be unique")
        result = self.run(state, ["Which category best describes this state?"], options, kind="choice")
        result.update(result.pop("results")[0])
        return result


def validate_decisions(payload):
    if not isinstance(payload, dict) or set(payload) != {"model", "state", "questions"}:
        raise ValueError("body requires exactly model, state and questions")
    if payload["model"] != MODEL:
        raise ValueError("unsupported decision model")
    state = payload["state"]
    if not isinstance(state, (str, dict, list)) or not state or (isinstance(state,str) and not state.strip()):
        raise ValueError("state must be nonempty text, object or array")
    if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()) > MAX_BODY:
        raise ValueError("request exceeds 2 MiB")
    questions = payload["questions"]
    if not isinstance(questions,list) or not 1 <= len(questions) <= 128:
        raise ValueError("questions requires 1..128 items")
    ids = set()
    def identifier(value):
        return isinstance(value,str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}",value) is not None
    def description(value):
        return isinstance(value,str) and value.strip() and len(value) <= 8192
    for q in questions:
        if not isinstance(q,dict): raise ValueError("invalid question")
        fields = {"id","type","instruction"}
        kind = q.get("type")
        if kind == "choice": fields.add("options")
        elif kind == "rating": fields.add("levels")
        elif kind != "boolean": raise ValueError("unknown decision type")
        if set(q) != fields or not identifier(q["id"]) or q["id"] in ids or not description(q["instruction"]):
            raise ValueError("invalid question fields, ID or instruction")
        ids.add(q["id"])
        if kind == "choice":
            options = q["options"]
            if not isinstance(options,list) or not 2 <= len(options) <= 16:
                raise ValueError("choice requires 2..16 options")
            keys = set()
            for o in options:
                if not isinstance(o,dict) or set(o) != {"id","description"} or not identifier(o["id"]) or o["id"] in keys or not description(o["description"]):
                    raise ValueError("invalid or duplicate choice option")
                keys.add(o["id"])
        if kind == "rating" and (not isinstance(q["levels"],list) or len(q["levels"]) != 6 or not all(description(v) for v in q["levels"])):
            raise ValueError("rating requires six nonempty levels")


class Handler(BaseHTTPRequestHandler):
    scorer: SemanticScorer

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def reply(self, status: int, value: dict):
        # JSON has no infinity: zero-probability labels retain -inf as a string.
        body = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"error": "not found"})
            return
        try:
            self.scorer.backend.request("/health", timeout=5, parse_json=False)
            self.scorer.backend.verify_model(self.scorer.model, self.scorer.revision)
            self.reply(200, {"status": "ok", "model": self.scorer.model, "revision": self.scorer.revision,
                             "context": self.scorer.context, "concurrency": self.scorer.concurrency, "protocol": "decisions-v1",
                             "types": ["boolean", "choice", "rating"], "max_options": 16, "max_questions": 128})
        except BackendError as exc:
            self.reply(503, {"error": str(exc)})

    def do_POST(self):
        routes = {"/v1/semantic/score": ("predicate", self.scorer.semantic_score),
                  "/v1/semantic/score-many": ("predicates", self.scorer.semantic_score_many),
                  "/v1/semantic/classify": ("options", self.scorer.semantic_classify)}
        if self.path not in routes and self.path != "/v1/decisions":
            self.reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.reply(413, {"error": "body must be 1 byte..2 MiB"})
                return
            value = json.loads(self.rfile.read(length))
            if self.path == "/v1/decisions":
                self.reply(200, json_safe(self.scorer.decisions(value)))
                return
            field, method = routes[self.path]
            if not isinstance(value, dict) or set(value) != {"state", field}:
                raise ValueError(f"body requires exactly state and {field}")
            result = method(value["state"], value[field])
            # Preserve actual zero probabilities without nonstandard JSON numbers.
            self.reply(200, json_safe(result))
        except (ValueError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})
        except BusyError as exc:
            self.reply(429, {"error": str(exc)})
        except BackendError as exc:
            self.reply(502, {"error": str(exc)})
        except Exception:
            self.reply(500, {"error": "semantic provider failed"})

    def log_message(self, fmt, *args):
        pass  # State and predicates can contain private application data.


def json_safe(value):
    if isinstance(value, float) and value == -math.inf:
        return "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--backend", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--context", type=int, default=18432)
    parser.add_argument("--concurrency", type=int, default=2)


def from_args(args, *, verify_backend=True):
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    backend = VLLM(args.backend)
    if verify_backend:
        backend.verify_model(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    def metadata(name):
        with open(hf_hub_download(args.model, name, revision=args.revision)) as source:
            return json.load(source)
    return SemanticScorer(tokenizer, backend, metadata("adapter_vllm/decision_head.json"),
                          metadata("calibration.json")["per_kind"], context=args.context,
                          concurrency=args.concurrency, model=args.model, revision=args.revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--port", type=int, default=18040)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--preflight", action="store_true", help="Check prepared offline artifacts without starting inference")
    args = parser.parse_args()
    if args.preflight:
        from huggingface_hub import hf_hub_download
        from_args(args, verify_backend=False)
        index = hf_hub_download(args.model, "model.safetensors.index.json", revision=args.revision, local_files_only=True)
        with open(index) as source:
            shards = set(json.load(source)["weight_map"].values())
        for name in shards | {"adapter_vllm/adapter_config.json", "adapter_vllm/adapter_model.safetensors"}:
            hf_hub_download(args.model, name, revision=args.revision, local_files_only=True)
        print(json.dumps({"prepared": True, "model": args.model, "revision": args.revision}))
        return
    scorer = from_args(args)
    handler = type("SemanticHandler", (Handler,), {"scorer": scorer})
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
