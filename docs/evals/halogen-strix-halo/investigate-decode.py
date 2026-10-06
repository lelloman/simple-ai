#!/usr/bin/env python3
"""Run one synthetic benchmark through the runner, with read-only host telemetry."""
import argparse
import concurrent.futures
import glob
import json
import pathlib
import socket
import subprocess
import threading
import time
import urllib.request

from benchmark import messages, request


def metrics():
    with urllib.request.urlopen("http://127.0.0.1:8731/metrics", timeout=3) as r:
        return {line.split()[0]: float(line.split()[1])
                for line in r.read().decode().splitlines()
                if line and not line.startswith("#")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--output-tokens", type=int, default=1024)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.url = "http://127.0.0.1:8080"
    a.model = "code:smart"
    a.timeout = 600
    name = f"long-{a.rows}-{a.repeat}"
    if a.concurrency > 1:
        name += f"-batch{a.concurrency}"
    fields = ["freq1_input", "power1_average", "power1_input", "temp1_input"]
    paths = [f for field in fields for f in glob.glob(
        f"/sys/class/drm/card*/device/hwmon/hwmon*/{field}")]
    paths += glob.glob("/sys/class/drm/card*/device/pp_dpm_mclk")
    paths += glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
    engine_pids = subprocess.check_output(["pgrep", "-x", "flash_serve"], text=True).split()
    start = time.monotonic()
    stop = threading.Event()
    samples = []

    def collect():
        while not stop.is_set():
            s = {"t": time.monotonic() - start, "wall_time": time.time()}
            for f in paths:
                try:
                    s[f] = pathlib.Path(f).read_text().strip()
                except OSError:
                    pass
            s["vmstat"] = {k: int(v) for k, v in (line.split() for line in
                pathlib.Path("/proc/vmstat").read_text().splitlines())
                if k in ("pgmajfault", "pswpin", "pswpout", "compact_stall")}
            s["io_pressure"] = pathlib.Path("/proc/pressure/io").read_text().strip()
            for pid in engine_pids:
                try:
                    s[f"engine_io_{pid}"] = {k.rstrip(":"): int(v) for k, v in
                        (line.split() for line in pathlib.Path(f"/proc/{pid}/io").read_text().splitlines())}
                except OSError:
                    pass
            try:
                s["metrics"] = metrics()
            except Exception as e:
                s["metrics_error"] = str(e)
            samples.append(s)
            stop.wait(1)

    # Observe a quiet admission point, but do not take a production host offline.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if metrics().get("llamacpp:requests_processing", 0) == 0:
            break
        time.sleep(2)
    before = metrics()
    start = time.monotonic()
    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    try:
        if a.concurrency == 1:
            results = [request(a, name, messages(a.rows, f"decode-investigation-v1-{name}"))]
        else:
            barrier = threading.Barrier(a.concurrency)
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                futures = [pool.submit(request, a, f"{name}-{i}", messages(a.rows,
                    f"decode-investigation-v1-{name}-{i}"), barrier)
                    for i in range(a.concurrency)]
                results = [future.result() for future in futures]
    finally:
        stop.set()
        thread.join(timeout=5)
        (a.out / f"{name}-telemetry.json").write_text(json.dumps({
            "host": socket.gethostname(), "kernel": subprocess.check_output(
                ["uname", "-r"], text=True).strip(),
            "before": before, "samples": samples,
        }, indent=2))
    print("RESULT", json.dumps({"host": socket.gethostname(), "name": name,
          "results": [{"timings": r["timings"], "error": r.get("error")} for r in results],
          "max_requests_processing": max((s.get("metrics", {}).get(
              "llamacpp:requests_processing", 0) for s in samples), default=0)}), flush=True)


if __name__ == "__main__":
    main()
