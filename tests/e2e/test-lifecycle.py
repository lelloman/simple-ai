"""Check real backend signals with the isolated Compose fixture running."""
import subprocess
import time


def compose(*args):
    return subprocess.check_output(["docker", "compose", *args], text=True).strip()


for signal in ("SIGTERM", "SIGINT"):
    compose("up", "-d", "backend")
    container = compose("ps", "-q", "backend")
    deadline = time.monotonic() + 30
    while subprocess.check_output(["docker", "inspect", "-f", "{{.State.Health.Status}}", container], text=True).strip() != "healthy":
        assert time.monotonic() < deadline, "backend readiness timed out"
        time.sleep(0.2)
    started = time.monotonic()
    subprocess.run(["docker", "kill", "--signal", signal, container], check=True, capture_output=True)
    exit_code = subprocess.check_output(["docker", "wait", container], text=True, timeout=10).strip()
    assert exit_code == "0", (signal, exit_code)
    assert "Graceful shutdown complete" in compose("logs", "backend")
    print(f"PASS backend {signal}: exited cleanly in {time.monotonic() - started:.2f}s")
