#!/usr/bin/env python3
"""Compare isolated OpenAI-compatible servers with identical synthetic code prompts.

Cold cases have distinct first-message prefixes. Report server token counts and
timings, not a guessed tokens-per-word conversion. No production prompts are used.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import threading
import time
import urllib.request


def messages(rows, case):
    source = [f"Benchmark case {case}. Review this synthetic Python module.\n"]
    for i in range(rows):
        source.append(f"def item_{i}(value):\n    return (value + {i % 97}) % 251\n")
    return [
        {"role": "user", "content": "".join(source)},
        {"role": "user", "content": "Explain the module's behavior, identify edge cases, and write a detailed test plan with Python test examples."},
    ]


def request(args, name, conversation, barrier=None):
    payload = dict(model=args.model, messages=conversation, temperature=0,
                   max_tokens=args.output_tokens, stream=True,
                   reasoning_effort="none",
                   stream_options={"include_usage": True},
                   chat_template_kwargs={"enable_thinking": False})
    encoded = json.dumps(payload).encode()
    result = {"name": name, "request_sha256": hashlib.sha256(encoded).hexdigest(),
              "request": payload, "chunks": [], "chunk_times_s": [], "usage": {}, "timings": {},
              "answer": "", "reasoning": "", "finish_reason": None}
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                 data=encoded, headers={"Content-Type": "application/json"})
    if barrier:
        barrier.wait()
    start = time.monotonic()
    first = last = None
    done = False
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            result["headers_s"] = time.monotonic() - start
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                result["chunks"].append(chunk)
                result["chunk_times_s"].append(time.monotonic() - start)
                result["usage"] = chunk.get("usage") or result["usage"]
                result["timings"] = chunk.get("timings") or result["timings"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if content or reasoning:
                        last = time.monotonic()
                        first = first if first is not None else last
                    result["answer"] += content
                    result["reasoning"] += reasoning
                    result["finish_reason"] = choice.get("finish_reason") or result["finish_reason"]
        if not done or result["finish_reason"] is None or first is None:
            raise RuntimeError("Incomplete stream or no generated text")
    except Exception as error:
        result["error"] = str(error)
    result["elapsed_s"] = time.monotonic() - start
    result["ttft_s"] = first - start if first is not None else None
    tokens = result["usage"].get("completion_tokens", 0)
    result["observed_decode_tps"] = ((tokens - 1) / (last - first)
                                      if tokens > 1 and first is not None and last > first else None)
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in
                      ("name", "usage", "timings", "ttft_s", "elapsed_s", "error")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--rows", default="256,1024,2048", help="Code rows; actual token counts come from the server")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--corpus-id", default="halo-comparison-v1", help="Use the same ID for both engines; change for a fresh rerun")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for rows in map(int, args.rows.split(",")):
        for repeat in range(args.repeats):
            name = f"cold-{rows}-{repeat}"
            conversation = messages(rows, f"{args.corpus_id}-{name}")
            cold = request(args, name, conversation)
            results.append(cold)
            if "error" not in cold:
                followup = conversation + [
                    {"role": "assistant", "content": cold["answer"]},
                    {"role": "user", "content": "Now write additional tests for negative integers and explain their expected results."},
                ]
                results.append(request(args, f"warm-{rows}-{repeat}", followup))
    if args.concurrency > 1:
        barrier = threading.Barrier(args.concurrency)
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(request, args, f"concurrent-{i}",
                                   messages(1024, f"{args.corpus_id}-concurrent-{i}"), barrier)
                       for i in range(args.concurrency)]
            concurrent_results = [future.result() for future in futures]
        elapsed = time.monotonic() - start
        results.extend(concurrent_results)
        (args.out / "concurrency.json").write_text(json.dumps({
            "elapsed_s": elapsed, "requests": args.concurrency,
            "completed": sum("error" not in r for r in concurrent_results),
            "completion_tokens_per_wall_second": sum(r["usage"].get("completion_tokens", 0) for r in concurrent_results) / elapsed,
        }, indent=2))
    summary = [{k: v for k, v in r.items() if k not in {"chunks", "request", "answer", "reasoning"}} for r in results]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return int(any("error" in r for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
