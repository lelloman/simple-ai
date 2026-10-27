#!/usr/bin/env python3
"""AI-57 gateway qualification. Standard library only; never changes fleet config."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request

MODELS = ("Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-m3", "class:embed_large")
TEXT = ("Questo sito presenta un'impresa italiana che offre servizi e prodotti per "
        "famiglie e aziende. Il profilo descrive le attività, i contatti, le sedi, "
        "le competenze del personale e le opportunità di collaborazione. Sono "
        "disponibili informazioni sui progetti, sulla qualità dei materiali, "
        "sull'assistenza ai clienti e sulle modalità per richiedere un preventivo. ")


def inputs(batch):
    return [(f"Profilo {batch * 32 + i:08d}. " + TEXT * 3)[:700] for i in range(32)]


def validate_embeddings(response, count, dimension=None):
    rows = response["data"]
    if len(rows) != count or [r["index"] for r in rows] != list(range(count)):
        raise ValueError("Embedding count/order mismatch")
    for row in rows:
        vector = row["embedding"]
        if not vector or (dimension is not None and len(vector) != dimension):
            raise ValueError("Embedding dimension mismatch")
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in vector):
            raise ValueError("Non-finite or non-numeric embedding")
        if not any(vector):
            raise ValueError("Zero embedding")
    return len(rows[0]["embedding"])


def summary(samples):
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("No latency samples")
    return {"count": len(samples), "p50_s": statistics.median(samples),
            "p95_s": ordered[math.ceil(len(ordered) * .95) - 1]}


class Client:
    def __init__(self, url, key):
        self.url = url.rstrip("/")
        self.key = key

    def request(self, endpoint, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + endpoint, data=data, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        # Errors propagate: a failed request must never count as throughput.
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def probe(self, probes):
        timings = {}
        for endpoint, payload in probes.items():
            start = time.perf_counter()
            response = self.request(endpoint, payload)
            if endpoint == "/v1/embeddings":
                count = 1 if isinstance(payload["input"], str) else len(payload["input"])
                validate_embeddings(response, count)
            elif not response.get("data"):
                raise ValueError("Empty classification response")
            timings[endpoint] = time.perf_counter() - start
        return timings


def collect_probes(client, probes, count=12):
    samples = {endpoint: [] for endpoint in probes}
    for _ in range(count):
        for endpoint, elapsed in client.probe(probes).items():
            samples[endpoint].append(elapsed)
    return {endpoint: summary(values) for endpoint, values in samples.items()}


def benchmark(client, model, concurrency, probes=None):
    dimension = 1024 if model != "class:embed_large" else None
    # Warm separately; report this latency so reload cost is visible.
    start = time.perf_counter()
    response = client.request("/v1/embeddings", {"model": model, "input": inputs(-1)})
    actual_dim = validate_embeddings(response, 32, dimension)
    warm_s = time.perf_counter() - start
    stop = threading.Event()
    started = threading.Event()

    def monitor():
        samples = {endpoint: [] for endpoint in probes}
        started.set()
        while True:
            for endpoint, elapsed in client.probe(probes).items():
                samples[endpoint].append(elapsed)
            if stop.is_set():
                break
        return {endpoint: summary(values) for endpoint, values in samples.items()}

    def batch(index):
        start = time.perf_counter()
        response = client.request("/v1/embeddings", {"model": model, "input": inputs(index)})
        validate_embeddings(response, 32, actual_dim)
        return time.perf_counter() - start

    # Same 512 texts at each concurrency; contention is a separate measurement.
    with ThreadPoolExecutor(max_workers=concurrency + 1) as pool:
        monitoring = pool.submit(monitor) if probes else None
        if monitoring:
            started.wait()
        start = time.perf_counter()
        try:
            latencies = list(pool.map(batch, range(16)))
            elapsed = time.perf_counter() - start
        finally:
            stop.set()
        during = monitoring.result() if monitoring else None  # Propagate failed probes.
    return {"model": model, "concurrency": concurrency, "batch_size": 32,
            "texts": 512, "characters_per_text": 700, "dimension": actual_dim,
            "warm_s": warm_s, "elapsed_s": elapsed, "texts_per_s": 512 / elapsed,
            "request_latency": summary(latencies), "during_probes": during}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://ai.lelloman.com")
    parser.add_argument("--probes", type=Path, required=True,
                        help="JSON object mapping /v1/embeddings and /v1/classifications to payloads")
    parser.add_argument("--baseline", type=Path, help="Before-rollout report; omit to record baseline only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = Client(args.url, os.environ["SIMPLE_AI_API_KEY"])
    probes = json.loads(args.probes.read_text())
    if set(probes) != {"/v1/embeddings", "/v1/classifications"}:
        parser.error("Supply exactly embeddings and classifications probes")
    probe_hash = hashlib.sha256(json.dumps(probes, sort_keys=True).encode()).hexdigest()
    report = {"url": args.url, "started_unix": time.time(), "probe_sha256": probe_hash,
              "complete": False, "matrix": []}
    try:
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())
            if not baseline["complete"] or baseline["probe_sha256"] != probe_hash or baseline["url"] != args.url:
                raise ValueError("Baseline incomplete or URL/probe payloads differ")
            report["baseline"] = baseline["probes"]
            ids = {m["id"] for m in client.request("/v1/models")["data"]}
            if not set(MODELS[:2]).issubset(ids):
                raise ValueError("Candidate model ids missing from /v1/models")
            report["discovered_candidates"] = list(MODELS[:2])
            report["before_matrix_probes"] = collect_probes(client, probes)
            for model in MODELS:
                for concurrency in (1, 4, 8):
                    result = benchmark(client, model, concurrency)
                    report["matrix"].append(result)
                    print(f"{model} c={concurrency}: {result['texts_per_s']:.2f} texts/s", flush=True)
            report["contention"] = [benchmark(client, model, 4, probes) for model in MODELS[:2]]
            report["probes"] = collect_probes(client, probes)
        else:
            report["probes"] = collect_probes(client, probes)
        report["complete"] = True
    except Exception as error:
        # Do not serialize HTTP bodies/headers or credentials.
        report["error"] = type(error).__name__
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
