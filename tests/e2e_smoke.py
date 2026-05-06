"""End-to-end smoke test for the SignLearn backend.

Starts the server, streams one fixture sequence over WebSocket, verifies a
ready prediction is received, posts a speech entry, and asserts both show up
in GET /transcript.

Usage:
    python scripts/e2e_smoke.py               # starts its own server
    python scripts/e2e_smoke.py --no-server   # server already running
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import socketio as sio_lib

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "processed_mini" / "train"
DEFAULT_URL = "http://localhost:5001"


def _wait_for_server(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"Server did not start within {timeout}s")


def _load_frames() -> list[list[float]]:
    files = sorted(FIXTURE_DIR.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No fixture .npy files in {FIXTURE_DIR}")
    return [frame.tolist() for frame in np.load(files[0])]  # 30 frames


def run(url: str, start_server: bool) -> None:
    proc = None
    if start_server:
        print("Starting server...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_server.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_server(url)
        print(f"Server up at {url}")

    try:
        _smoke(url)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=5)


def _smoke(url: str) -> None:
    import threading
    import json

    sio = sio_lib.Client()
    ready_received = threading.Event()
    predictions: list[dict] = []

    @sio.on("prediction")
    def on_pred(data):
        predictions.append(data)
        if data.get("ready"):
            ready_received.set()

    # --- Step 1: clear any leftover transcript ---
    req = urllib.request.Request(
        f"{url}/transcript?confirm=1",
        method="DELETE",
    )
    urllib.request.urlopen(req)
    print("Transcript cleared.")

    # --- Step 2: connect and stream one window ---
    sio.connect(url)
    sio.emit("reset")
    time.sleep(0.05)

    frames = _load_frames()
    print(f"Streaming {len(frames)} frames...")
    for frame in frames:
        sio.emit("frame", {"landmarks": frame, "t": int(time.time() * 1000)})
        time.sleep(0.005)

    # --- Step 3: wait for a ready prediction ---
    if not ready_received.wait(timeout=10.0):
        sio.disconnect()
        raise AssertionError("Timed out waiting for a ready prediction")

    last_pred = [p for p in predictions if p.get("ready")][-1]
    label = last_pred.get("label")
    conf = last_pred.get("confidence")
    print(f"Prediction received: label={label!r}  confidence={conf}")
    sio.disconnect()

    # --- Step 4: post a speech entry ---
    body = json.dumps({"text": "e2e smoke test"}).encode()
    req = urllib.request.Request(
        f"{url}/speech-to-text",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    assert resp.status == 201, f"POST /speech-to-text returned {resp.status}"
    print("Speech entry posted.")

    # --- Step 5: verify transcript endpoint ---
    transcript_resp = urllib.request.urlopen(f"{url}/transcript")
    transcript = json.loads(transcript_resp.read())["messages"]

    speech_entries = [m for m in transcript if m["source"] == "speech"]
    assert speech_entries, "Speech entry not found in transcript"
    assert speech_entries[0]["text"] == "e2e smoke test"
    print(f"Transcript OK — {len(transcript)} message(s), speech entry confirmed.")

    print("\nAll checks passed.")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()

    try:
        run(args.url, start_server=not args.no_server)
    except (AssertionError, RuntimeError) as exc:
        print(f"\nSMOKE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
