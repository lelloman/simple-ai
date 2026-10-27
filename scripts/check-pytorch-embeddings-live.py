#!/usr/bin/env python3
"""Read-only AI-59 gateway acceptance; operator supplies reference vectors/probes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time
import urllib.request

MODEL = "Qwen/Qwen3-Embedding-0.6B"
TEXT = "Questo sito presenta un'impresa italiana che offre servizi e prodotti per famiglie e aziende. "


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request("https://ai.lelloman.com" + path, data=data, headers={
        "Authorization": "Bearer " + os.environ["SIMPLE_AI_API_KEY"],
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def embed(texts):
    rows = request("/v1/embeddings", {"model": MODEL, "input": texts})["data"]
    if len(rows) != len(texts) or [r["index"] for r in rows] != list(range(len(texts))):
        raise ValueError("embedding count/order mismatch")
    for row in rows:
        vector = row["embedding"]
        if (len(vector) != 1024 or not all(math.isfinite(v) for v in vector)
                or abs(sum(v*v for v in vector) - 1) > .002):
            raise ValueError("expected finite 1024-d unit vectors")
    return [r["embedding"] for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help='operator JSON: {"input": ["same raw text"], "embeddings": [[1024 floats]]}')
    parser.add_argument("--probes", type=Path, required=True,
                        help='JSON array of {"path": "/v1/...", "body": {...}} for every other RTX engine')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"complete": False, "matrix": [], "reload_probes": []}
    try:
        assert MODEL in [r["id"] for r in request("/v1/models")["data"]]
        probes = json.loads(args.probes.read_text())
        if not isinstance(probes, list) or not probes:
            raise ValueError("provide probes for the other RTX engines")
        reference = json.loads(args.reference.read_text())
        expected = reference["embeddings"]
        actual = embed(reference["input"])
        if len(actual) != len(expected) or not actual:
            raise ValueError("reference count mismatch")
        cosines = []
        for a, b in zip(actual, expected):
            if len(b) != 1024 or not all(math.isfinite(v) for v in b):
                raise ValueError("invalid reference vector")
            cosine = sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
            if cosine < .99:
                raise ValueError(f"reference cosine {cosine} < 0.99")
            cosines.append(cosine)
        report["reference_min_cosine"] = min(cosines)
        for batch in (32, 64):
            texts = [(f"Profilo {i:04}. " + TEXT * 10)[:700] for i in range(batch)]
            for concurrency in (1, 4):
                start = time.perf_counter()
                embed(texts)
                warmup = time.perf_counter() - start
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    # Fixed 2048 completed texts, without counting warmup or failures.
                    list(pool.map(embed, [texts] * (2048 // batch)))
                rate = 2048 / (time.perf_counter() - start)
                report["matrix"].append({"batch": batch, "concurrency": concurrency,
                                         "texts_per_s": rate, "warmup_s": warmup})
                if rate < 120:
                    raise ValueError(f"throughput {rate:.1f} < 120 texts/s")
        for probe in probes:
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                request(probe["path"], probe["body"])
                samples.append(time.perf_counter() - start)
                embed(["embedding after another RTX engine"])
            report["reload_probes"].append({"model": probe["body"]["model"], "seconds": samples})
        report["complete"] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
