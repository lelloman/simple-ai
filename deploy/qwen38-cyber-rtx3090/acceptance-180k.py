"""Measure this checkpoint at short and occupied 180K context, without truncation."""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "Implement a Rust LRU cache with tests and explain its invariants.",
    "Explain how to diagnose a memory leak in a Linux service with commands and decision points.",
    "Write a Python JSON-lines ETL pipeline with validation and tests.",
    "Design a PostgreSQL transaction flow for idempotent payment processing.",
    "Review a concurrent job queue design for races, cancellation, and backpressure.",
]


class Client:
    def __init__(self, url, key, model):
        self.url, self.key, self.model = url.rstrip("/"), key, model

    def request(self, path, data):
        request = urllib.request.Request(
            self.url + path, data=json.dumps(data).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=1800)

    def count(self, messages):
        with self.request("/tokenize", {
            "model": self.model, "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }) as response:
            return json.load(response)["count"]

    def generate(self, messages, tokens, force_length=False):
        payload = {
            "model": self.model, "messages": messages, "temperature": 0,
            "max_tokens": tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if force_length:
            payload["min_tokens"] = tokens
        started = time.monotonic()
        first = None
        usage = None
        text = ""
        with self.request("/v1/chat/completions", payload) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content") or ""
                    if content:
                        first = first or time.monotonic()
                        text += content
        finished = time.monotonic()
        assert first and usage, "Missing streamed tokens or final usage"
        return {
            "usage": usage, "text": text,
            "ttft_seconds": first - started,
            "elapsed_seconds": finished - started,
            "decode_tps": (usage["completion_tokens"] - 1) / (finished - first),
        }


def long_messages(blocks):
    parts = ["Read the following synthetic inventory. Retrieve the three audit marker values exactly.\n"]
    markers = {blocks // 4: "ALPHA=731904682", blocks // 2: "BETA=296581043", 3 * blocks // 4: "GAMMA=854207319"}
    for i in range(blocks):
        parts.append(f"Record {i}: service node-{i:05d}, daily backup completed; patch review pending. "
                     "Owner: infrastructure team. This record contains routine inventory information, "
                     "with no audit marker. Retention is thirty days and the next review is scheduled.\n")
        if i in markers:
            parts.append("AUDIT MARKER: " + markers[i] + "\n")
    parts.append("First return ALPHA, BETA, and GAMMA with their exact values. "
                 "Then write a detailed summary of the inventory and its maintenance procedures.")
    return [{"role": "user", "content": "".join(parts)}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18021")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--short-results", type=Path,
                        help="Reuse completed short tests from this same running profile")
    args = parser.parse_args()
    client = Client(args.url, args.key_file.read_text().strip(), args.model)
    result = {"short": [], "long": [], "status": "running"}
    if args.short_results:
        result["short"] = json.loads(args.short_results.read_text())["short"]
        assert len(result["short"]) == 10
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not args.short_results:
        client.generate([{"role": "user", "content": "Explain hash tables."}], 64)
    for i, prompt in enumerate([] if args.short_results else PROMPTS * 2):
        row = client.generate([{"role": "user", "content": prompt}], 512, True)
        result["short"].append(row)
        save()
        print(f"short {i + 1}: {row['decode_tps']:.1f} tok/s", flush=True)
    result["short_median_decode_tps"] = statistics.median(x["decode_tps"] for x in result["short"])
    if not args.short_only:
        low, high = 1, 6000
        while low < high:
            mid = (low + high + 1) // 2
            if client.count(long_messages(mid)) <= 179000:
                low = mid
            else:
                high = mid - 1
        messages = long_messages(low)
        result["long_tokenize_count"] = client.count(messages)
        assert 178000 <= result["long_tokenize_count"] <= 179000
        for i in range(2):
            row = client.generate(messages, 512, True)
            result["long"].append(row)
            save()
            assert 178000 <= row["usage"]["prompt_tokens"] < 180000, "Prompt was truncated or mistokenized"
            assert all(value in row["text"] for value in ["731904682", "296581043", "854207319"]), row["text"]
            print(f"long {i + 1}: {row['usage']['prompt_tokens']} input tokens, "
                  f"{row['decode_tps']:.1f} tok/s, markers PASS", flush=True)
    result["status"] = "PASS" if result["short_median_decode_tps"] >= 100 else "BELOW_SPEED_TARGET"
    save()
    print(json.dumps({k: v for k, v in result.items() if k not in ("short", "long")}), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
