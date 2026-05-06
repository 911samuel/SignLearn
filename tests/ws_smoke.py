"""Manual WebSocket smoke test.

Streams one fixture .npy file to the running server and prints each prediction.

Usage (server must already be running):
    python tests/ws_smoke.py
    python tests/ws_smoke.py --fixture tests/fixtures/processed_mini/train/a_s01_0000.npy
    python tests/ws_smoke.py --url http://localhost:5001
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import socketio

DEFAULT_URL = "http://localhost:5001"
DEFAULT_FIXTURE = "tests/fixtures/processed_mini/train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--fixture", default=None, help="Path to a .npy file or directory")
    args = parser.parse_args()

    fixture_path = Path(args.fixture) if args.fixture else Path(DEFAULT_FIXTURE)
    if fixture_path.is_dir():
        npy_files = sorted(fixture_path.glob("*.npy"))
        if not npy_files:
            print(f"No .npy files found in {fixture_path}", file=sys.stderr)
            sys.exit(1)
        fixture_path = npy_files[0]
    seq = np.load(fixture_path)
    print(f"Streaming {fixture_path.name} ({seq.shape}) to {args.url}")

    sio = socketio.Client()
    received: list[dict] = []

    @sio.event
    def connect():
        print("Connected.")

    @sio.on("prediction")
    def on_prediction(data):
        received.append(data)
        status = f"ready={data['ready']}"
        if data["ready"] and data["label"]:
            status += f"  label={data['label']}  conf={data['confidence']:.3f}"
        print(f"  prediction: {status}")

    @sio.on("error")
    def on_error(data):
        print(f"  ERROR: {data['message']}", file=sys.stderr)

    sio.connect(args.url)
    sio.emit("reset")
    time.sleep(0.05)

    for i, frame in enumerate(seq):
        sio.emit("frame", {"landmarks": frame.tolist(), "t": int(time.time() * 1000)})
        time.sleep(0.033)  # ~30 FPS pacing

    time.sleep(0.5)
    sio.disconnect()

    ready_preds = [r for r in received if r["ready"]]
    if ready_preds:
        last = ready_preds[-1]
        print(f"\nFinal: label={last['label']!r}  confidence={last['confidence']}")
    else:
        print("\nNo ready prediction received.")


if __name__ == "__main__":
    main()
