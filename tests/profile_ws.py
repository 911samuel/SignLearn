"""WebSocket round-trip latency profiler for the SignLearn backend.

Starts the server in a subprocess, streams N frames (default 300), and
measures the round-trip latency for every "ready" prediction event — i.e.
the time between sending the frame that completes a 30-frame window and
receiving the resulting prediction.

Usage:
    python scripts/profile_ws.py
    python scripts/profile_ws.py --frames 300 --url http://localhost:5001
    python scripts/profile_ws.py --no-server   # server already running
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import socketio as sio_lib

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "processed_mini" / "train"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "phase3_latency.json"
DEFAULT_URL = "http://localhost:5001"
P95_TARGET_MS = 500.0


def _load_frames(n: int) -> list[list[float]]:
    files = sorted(FIXTURE_DIR.glob("*.npy"))
    assert files, f"No fixture files in {FIXTURE_DIR}"
    seq = np.load(files[0])
    frames: list[list[float]] = []
    while len(frames) < n:
        frames.extend(f.tolist() for f in seq)
    return frames[:n]


def _wait_for_server(url: str, timeout: float = 20.0) -> None:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"Server at {url} did not respond within {timeout}s")


def run(url: str, total_frames: int, start_server: bool) -> list[float]:
    """Return list of round-trip latencies in milliseconds for ready predictions."""
    proc = None
    if start_server:
        print("Starting server...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_server.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_server(url)
        print(f"Server ready at {url}")

    sio = sio_lib.Client()
    latencies_ms: list[float] = []

    # Queue of send-timestamps for window-completing frames (every 30th frame).
    window_send_times: list[float] = []
    lock = threading.Lock()

    @sio.on("prediction")
    def on_prediction(data):
        recv_ts = time.perf_counter()
        if not data.get("ready"):
            return
        with lock:
            if window_send_times:
                send_ts = window_send_times.pop(0)
                latencies_ms.append((recv_ts - send_ts) * 1000)

    sio.connect(url)
    sio.emit("reset")
    time.sleep(0.1)

    frames = _load_frames(total_frames)
    seq_len = 30

    # Warmup: send one full window so the model is loaded and JIT-warmed
    # before we start timing. Discard any prediction received.
    print("Warming up (1 window)...", flush=True)
    warmup_frames = frames[:seq_len]
    for frame in warmup_frames:
        sio.emit("frame", {"landmarks": frame, "t": int(time.time() * 1000)})
        time.sleep(0.005)
    time.sleep(1.0)
    with lock:
        latencies_ms.clear()
        window_send_times.clear()
    sio.emit("reset")
    time.sleep(0.1)

    print(f"Streaming {total_frames} frames ({total_frames // seq_len} windows)...")

    for i, frame in enumerate(frames):
        frame_num = i + 1
        # Record timestamp just before emitting the window-completing frame.
        if frame_num % seq_len == 0:
            ts = time.perf_counter()
            with lock:
                window_send_times.append(ts)
        sio.emit("frame", {"landmarks": frame, "t": int(time.time() * 1000)})
        time.sleep(0.005)  # ~200 FPS — fast enough to stress without flooding

    time.sleep(2.0)  # drain in-flight responses
    sio.disconnect()

    if proc is not None:
        proc.terminate()
        proc.wait(timeout=5)

    return latencies_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--no-server", action="store_true",
                        help="Connect to an already-running server")
    args = parser.parse_args()

    latencies = run(args.url, args.frames, start_server=not args.no_server)

    if not latencies:
        print("No latency samples collected — check server logs.", file=sys.stderr)
        sys.exit(1)

    sorted_lat = sorted(latencies)
    p50 = statistics.median(latencies)
    p95 = sorted_lat[int(len(latencies) * 0.95)]
    p99 = sorted_lat[min(int(len(latencies) * 0.99), len(latencies) - 1)]
    mean = statistics.mean(latencies)

    report = {
        "frames_sent": args.frames,
        "predictions_received": len(latencies),
        "mean_ms": round(mean, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "target_p95_ms": P95_TARGET_MS,
        "passed": p95 <= P95_TARGET_MS,
        "async_mode": "threading",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*40}")
    print(f"  Samples  : {len(latencies)}")
    print(f"  Mean     : {mean:.1f} ms")
    print(f"  p50      : {p50:.1f} ms")
    print(f"  p95      : {p95:.1f} ms  (target < {P95_TARGET_MS} ms)")
    print(f"  p99      : {p99:.1f} ms")
    status = "PASS ✓" if report["passed"] else "FAIL ✗"
    print(f"  Result   : {status}")
    print(f"{'='*40}")
    print(f"\nReport saved to {REPORT_PATH}")

    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
