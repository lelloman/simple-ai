#!/usr/bin/env python3
"""Operator-only: compare 200 real multilingual texts on two direct runner URLs."""
import argparse
import json
import math
from pathlib import Path
import urllib.request

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def compare(left, right):
    if len(left) != 200 or len(right) != 200:
        raise ValueError("both runners must return 200 vectors")
    scores = []
    for a, b in zip(left, right):
        if len(a) != 1024 or len(b) != 1024:
            raise ValueError("expected 1024 dimensions")
        if not all(math.isfinite(x) for x in a + b):
            raise ValueError("non-finite embedding")
        na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
        if abs(na - 1) > .001 or abs(nb - 1) > .001:
            raise ValueError("expected unit vectors")
        scores.append(sum(x*y for x, y in zip(a, b)) / (na*nb))
    return {"pair_cosines": scores, "min_cosine": min(scores),
            "mean_cosine": sum(scores)/len(scores), "accepted": min(scores) >= .999}


def embed(url, texts):
    # Fixed small batches bound CPU working memory and preserve input order.
    vectors = []
    for start in range(0, len(texts), 8):
        batch = texts[start:start+8]
        request = urllib.request.Request(url.rstrip("/") + "/v1/embeddings",
            data=json.dumps({"model": MODEL, "input": batch, "allow_engine_swap": True}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)["data"]
        if [item["index"] for item in data] != list(range(len(batch))):
            raise ValueError("missing or reordered embeddings")
        vectors.extend(item["embedding"] for item in data)
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True,
                        help='JSON {"input": [200 real multilingual texts]}')
    parser.add_argument("--halo-url", required=True, help="direct halo2 runner, never gateway")
    parser.add_argument("--rtx-url", required=True, help="direct RTX runner; drain batch work first")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    texts = json.loads(args.texts.read_text())["input"]
    if (len(texts) != 200 or len(set(texts)) != 200
            or not all(isinstance(t, str) and t.strip() for t in texts)):
        parser.error("provide exactly 200 distinct real multilingual texts")
    # Write an incomplete report before doing work; failures cannot leave a stale pass.
    import hashlib
    report = {"accepted": False, "complete": False, "model": MODEL,
              "input_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
              "halo_url": args.halo_url, "rtx_url": args.rtx_url}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        halo = embed(args.halo_url, texts)
        rtx = embed(args.rtx_url, texts)
        report.update(compare(halo, rtx))
        report["complete"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "pair_cosines"}, indent=2))
    if not report["accepted"]:
        raise SystemExit("Parity below 0.999: block rollout; do not switch routing.")


if __name__ == "__main__":
    main()
