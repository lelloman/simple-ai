#!/usr/bin/env python3
"""Dependency-free streaming decode benchmark for a vLLM OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


PROMPTS = (
    "Implement a production-quality Rust LRU cache with tests. Explain the invariants as code comments.",
    "Review a concurrent job queue design and propose concrete fixes for races, cancellation, and backpressure.",
    "Write a detailed Python implementation of an incremental JSON-lines ETL pipeline with validation and tests.",
    "Design a PostgreSQL schema and transaction flow for idempotent payment processing, including failure recovery.",
    "Explain how to diagnose a memory leak in a long-running Linux service, with commands and decision points.",
)


def run_request(url: str, key: str, model: str, prompt: str, tokens: int) -> dict[str, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": tokens,
        "min_tokens": tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    first_token_at: float | None = None
    finished_at = started
    completion_tokens: int | None = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            finished_at = time.monotonic()
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
            for choice in chunk.get("choices", ()):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    first_token_at = first_token_at or finished_at
    if first_token_at is None or completion_tokens is None:
        raise RuntimeError("stream did not contain token data and final usage")
    decode_seconds = max(finished_at - first_token_at, 1e-9)
    # The first token arrives at first_token_at; only subsequent token intervals
    # belong in steady-state decode throughput.
    decode_tokens = max(completion_tokens - 1, 0)
    return {
        "tokens": float(completion_tokens),
        "ttft": first_token_at - started,
        "elapsed": finished_at - started,
        "decode_tps": decode_tokens / decode_seconds,
        "e2e_tps": completion_tokens / (finished_at - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18020")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print("warming up (excluded)...", flush=True)
    run_request(args.url, args.api_key, args.model, PROMPTS[0], 64)
    results = []
    for index in range(args.runs):
        try:
            result = run_request(
                args.url,
                args.api_key,
                args.model,
                PROMPTS[index % len(PROMPTS)],
                args.output_tokens,
            )
        except urllib.error.HTTPError as error:
            raise SystemExit(error.read().decode()) from error
        results.append(result)
        print(
            f"run {index + 1}: {int(result['tokens'])} tokens, "
            f"TTFT {result['ttft']:.3f}s, decode {result['decode_tps']:.1f} tok/s, "
            f"end-to-end {result['e2e_tps']:.1f} tok/s",
            flush=True,
        )
    rates = [result["decode_tps"] for result in results]
    ttfts = [result["ttft"] for result in results]
    print(
        f"summary: median decode {statistics.median(rates):.1f} tok/s, "
        f"mean decode {statistics.mean(rates):.1f} tok/s, "
        f"median TTFT {statistics.median(ttfts):.3f}s"
    )


if __name__ == "__main__":
    main()
