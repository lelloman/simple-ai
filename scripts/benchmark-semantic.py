#!/usr/bin/env python3
"""Exclusive-backend cold-cache semantic latency/throughput benchmark.

Run on the GPU host: nvidia-smi samples local device-wide memory, not remote VRAM.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time

from simple_ai_semantic import arguments, from_args, json_safe


class MemorySampler:
    def __init__(self, gpu):
        self.gpu = gpu
        self.samples = []
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run([
                    "nvidia-smi", "-i", str(self.gpu),
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                ], check=True, capture_output=True, text=True, timeout=5)
                self.samples.append(float(result.stdout.strip()))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.error = str(exc)
                return
            self.stop.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


def state_at_length(tokenizer, target):
    sentence = "Build log: dependency download failed because DNS resolution timed out. No compiler or tests ran.\n"
    ids = tokenizer.encode(sentence * (target // 8 + 10), add_special_tokens=False)[:target]
    state = tokenizer.decode(ids)
    return state, len(tokenizer.encode(state, add_special_tokens=False))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def backend_metadata(info):
    """Keep reproducibility settings without saving backend API keys or paths to secrets."""
    allowed = {"version", "model_path", "revision", "quantization", "dtype", "context_length",
               "max_total_tokens", "max_running_requests", "chunked_prefill_size", "mem_fraction_static",
               "mamba_radix_cache_strategy", "disable_radix_cache", "disable_cuda_graph",
               "attention_backend", "moe_runner_backend", "tp_size", "json_model_override_args"}
    result = {}
    for key, value in info.items():
        if key in allowed:
            result[key] = value
        elif isinstance(value, dict):
            nested = backend_metadata(value)
            if nested:
                result[key] = nested
    return result


def trial(scorer, state, predicates, mode, gpu):
    scorer.backend.flush()  # Outside the timed region, for both modes.
    with MemorySampler(gpu) as memory:
        start = time.perf_counter()
        flush_seconds = 0.0
        if mode == "shared":
            responses = [scorer.semantic_score_many(state, predicates)]
        else:
            responses = []
            for index, predicate in enumerate(predicates):
                if index:
                    before = time.perf_counter()
                    scorer.backend.flush()
                    flush_seconds += time.perf_counter() - before
                # Exactly one full-context readout, no redundant warmup. Clearing
                # between calls prevents radix hits from flattering the baseline.
                responses.append(scorer.run(state, [predicate], ["False", "True"], warm=False))
        elapsed = time.perf_counter() - start
    hits = [x for response in responses for x in response["usage"]["branch_cached_tokens"]]
    return {"wall_ms": elapsed * 1000, "cache_reset_ms": flush_seconds * 1000,
            "wall_excluding_cache_reset_ms": (elapsed - flush_seconds) * 1000,
            "decisions_per_second": len(predicates) / elapsed,
            "amortized_ms_per_decision": elapsed * 1000 / len(predicates),
            "peak_device_memory_mib": max(memory.samples) if memory.samples else None,
            "memory_samples": len(memory.samples), "memory_error": memory.error,
            "branch_cached_tokens": hits,
            "cache_reuse_observed": any(x > 0 for x in hits if x is not None) if any(x is not None for x in hits) else None,
            "responses": responses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--lengths", type=int, nargs="+", default=[500, 2000, 8000, 16000])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--modes", nargs="+", choices=["shared", "independent"], default=["shared", "independent"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("semantic-benchmark.json"))
    parser.add_argument("--exclusive-backend", action="store_true", help="acknowledge cache resets; use a dedicated vLLM instance")
    args = parser.parse_args()
    if not args.exclusive_backend:
        parser.error("--exclusive-backend is required: this benchmark clears the backend cache")
    if args.repetitions < 1 or any(x < 1 for x in args.lengths) or any(not 1 <= x <= 128 for x in args.batches):
        parser.error("positive lengths/repetitions and batch sizes 1..128 required")
    scorer = from_args(args)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "settings": vars(args).copy(),
              "backend_info": scorer.backend.request("/v1/models"),
              "measurements": [], "comparisons": [], "warmups": []}
    report["settings"]["output"] = str(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exercise kernels before cold-cache trials; flushing removes KV, not JIT state.
    scorer.semantic_score("DNS lookup failed.", "The network failed.")
    base = ["The failure is environmental.", "The source code contains a confirmed bug.",
            "The test itself is broken.", "More information is required.",
            "The operation should be retried after network recovery."]
    for target in args.lengths:
        state, actual = state_at_length(scorer.tokenizer, target)
        for count in args.batches:
            # Distinct suffixes avoid accidentally benchmarking duplicate full prompts.
            predicates = [f"Decision {i + 1}: {base[i % len(base)]}" for i in range(count)]
            # GPU can specialize kernels for new state and batch shapes.
            # Warm both paths at this point before collecting cold-KV timings;
            # a tiny startup prompt alone does not exclude serving-time JIT.
            report["warmups"].append({
                "state_tokens_actual": actual, "predicates": count,
                "shared": trial(scorer, state, predicates, "shared", args.gpu),
                "independent_single": trial(scorer, state, predicates[:1], "independent", args.gpu),
            })
            trials = {mode: [] for mode in dict.fromkeys(args.modes)}
            for repetition in range(args.repetitions):
                order = list(trials) if repetition % 2 == 0 else list(reversed(trials))
                for mode in order:
                    trials[mode].append(trial(scorer, state, predicates, mode, args.gpu))
            for mode, samples in trials.items():
                walls = [v["wall_ms"] for v in samples]
                memory = [v["peak_device_memory_mib"] for v in samples if v["peak_device_memory_mib"] is not None]
                summary = {"state_tokens_target": target, "state_tokens_actual": actual,
                           "predicates": count, "mode": mode, "median_ms": statistics.median(walls),
                           "p95_ms": percentile(walls, .95),
                           "decisions_per_second": count * 1000 / statistics.median(walls),
                           "amortized_ms_per_decision": statistics.median(walls) / count,
                           "peak_device_memory_mib": max(memory) if memory else None,
                           "samples": samples}
                report["measurements"].append(summary)
                print(json.dumps({k: v for k, v in summary.items() if k != "samples"}), flush=True)
            if "shared" in trials and "independent" in trials:
                # Same prompts in both paths also expose cache-induced score drift.
                for shared, independent in zip(trials["shared"], trials["independent"]):
                    shared_scores = [r["scores"]["True"] for r in shared["responses"][0]["results"]]
                    independent_scores = [r["results"][0]["scores"]["True"] for r in independent["responses"]]
                    shared["max_absolute_score_delta_vs_independent"] = max(abs(a-b) for a, b in zip(shared_scores, independent_scores))
                shared_median = statistics.median(x["wall_ms"] for x in trials["shared"])
                report["comparisons"].append({
                    "state_tokens_actual": actual, "predicates": count,
                    "speedup_wall": statistics.median(x["wall_ms"] for x in trials["independent"]) / shared_median,
                    "speedup_excluding_cache_reset": statistics.median(x["wall_excluding_cache_reset_ms"] for x in trials["independent"]) / shared_median,
                    "max_absolute_score_delta": max(x["max_absolute_score_delta_vs_independent"] for x in trials["shared"]),
                })
            # Save after each pair so long experiments retain completed measurements.
            args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
