"""End-to-end smoke test for the SignLearn backend.

Creates a room, joins as Signer, streams one fixture sequence, verifies a
ready prediction is received. Joins a second client as Hearing, emits a
speech caption, then asserts both messages show up in GET /transcript.

Usage:
    python tests/e2e_smoke.py               # starts its own server
    python tests/e2e_smoke.py --no-server   # server already running
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
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


def _create_room(url: str) -> str:
    req = urllib.request.Request(f"{url}/rooms", method="POST")
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())["room_id"]


def _connect_and_join(url: str, room_id: str, role: str, name: str) -> sio_lib.Client:
    sio = sio_lib.Client()
    joined = threading.Event()

    @sio.on("join_ok")
    def _on_join_ok(_data):
        joined.set()

    @sio.on("join_error")
    def _on_join_error(data):
        raise AssertionError(f"join_error for {role}: {data}")

    sio.connect(url)
    sio.emit("join_room", {"room_id": room_id, "role": role, "name": name})
    if not joined.wait(timeout=5.0):
        raise AssertionError(f"Timed out joining room as {role}")
    return sio


def run(url: str, start_server: bool) -> None:
    proc = None
    if start_server:
        print("Starting server...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "backend" / "scripts" / "run_server.py")],
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
    room_id = _create_room(url)
    print(f"Room created: {room_id}")

    # --- Signer streams a window ---
    ready_received = threading.Event()
    predictions: list[dict] = []

    signer = sio_lib.Client()

    @signer.on("prediction")
    def on_pred(data):
        predictions.append(data)
        if data.get("ready"):
            ready_received.set()

    join_ok = threading.Event()
    signer.on("join_ok", lambda *_: join_ok.set())
    signer.connect(url)
    signer.emit("join_room", {"room_id": room_id, "role": "signer", "name": "A"})
    if not join_ok.wait(timeout=5):
        raise AssertionError("Signer failed to join")

    signer.emit("reset")
    time.sleep(0.05)

    frames = _load_frames()
    print(f"Streaming {len(frames)} frames...")
    for frame in frames:
        signer.emit("frame", {"landmarks": frame, "t": int(time.time() * 1000)})
        time.sleep(0.005)

    if not ready_received.wait(timeout=10.0):
        signer.disconnect()
        raise AssertionError("Timed out waiting for a ready prediction")

    last_pred = [p for p in predictions if p.get("ready")][-1]
    print(f"Prediction received: label={last_pred.get('label')!r} confidence={last_pred.get('confidence')}")
    signer.disconnect()

    # --- Hearing user emits a speech caption ---
    hearing = _connect_and_join(url, room_id, "hearing", "B")
    hearing.emit("speech", {"text": "e2e smoke test"})
    time.sleep(0.3)
    hearing.disconnect()
    print("Speech caption emitted.")

    # --- Verify transcript ---
    transcript_resp = urllib.request.urlopen(f"{url}/transcript?room_id={room_id}")
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
