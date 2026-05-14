"""Phase 5 integration acceptance tests (room-scoped).

Acceptance criteria:

1. /health returns model_loaded: true
2. Signer WebSocket round-trip completes for fixture frames
3. Hearing-user speech caption persists to /transcript
4. WS round-trip p95 < 2000 ms (from artifacts/reports/phase5_latency.json)
5. Malformed frame emits 'error' instead of crashing
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
import pytest
import socketio as sio_lib

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "processed_mini" / "train"
PHASE5_LATENCY_REPORT = REPO_ROOT / "artifacts" / "reports" / "phase5_latency.json"
BASE_URL = "http://localhost:5001"


def _wait_for_server(url: str, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"Server at {url} did not respond within {timeout}s")


@pytest.fixture(scope="session")
def server():
    import os
    env = {**os.environ, "SIGNLEARN_ASYNC_MODE": "threading", "FLASK_DEBUG": "1"}
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "backend" / "scripts" / "run_server.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    _wait_for_server(BASE_URL)
    yield BASE_URL
    proc.terminate()
    proc.wait(timeout=5)


def _create_room(url: str) -> str:
    req = urllib.request.Request(f"{url}/rooms", method="POST")
    return json.loads(urllib.request.urlopen(req).read())["room_id"]


def _join(url: str, room_id: str, role: str, name: str) -> sio_lib.Client:
    sio = sio_lib.Client()
    joined = threading.Event()
    sio.on("join_ok", lambda *_: joined.set())
    sio.connect(url)
    sio.emit("join_room", {"room_id": room_id, "role": role, "name": name})
    assert joined.wait(timeout=5.0), f"Failed to join as {role}"
    return sio


@pytest.fixture
def fresh_room(server):
    room_id = _create_room(server)
    yield room_id
    req = urllib.request.Request(
        f"{server}/transcript?room_id={room_id}&confirm=1", method="DELETE"
    )
    urllib.request.urlopen(req)


# 1. /health -------------------------------------------------------------

def test_health_model_loaded(server):
    resp = urllib.request.urlopen(f"{server}/health")
    data = json.loads(resp.read())
    assert data["status"] == "ok"
    assert data["model_loaded"] is True, f"load_error: {data.get('load_error')}"


# 2. WS prediction round-trip -------------------------------------------

def _load_fixture_frames() -> list[list[float]]:
    files = sorted(FIXTURE_DIR.glob("*.npy"))
    assert files
    return [frame.tolist() for frame in np.load(files[0])]


def test_ws_prediction_roundtrip(server, fresh_room):
    """Verify the full signer→inference→prediction socket pipeline.

    We only check that prediction events are emitted with the right structure
    (not that ready=True fires) because the default model may give low-confidence
    outputs on fixture frames that don't match its training distribution.
    """
    received: list[dict] = []
    got_prediction = threading.Event()
    signer = _join(server, fresh_room, "signer", "A")

    @signer.on("prediction")
    def on_pred(data):
        received.append(data)
        got_prediction.set()  # any prediction event is sufficient

    signer.emit("reset")
    time.sleep(0.05)
    for frame in _load_fixture_frames():
        signer.emit("frame", {"landmarks": frame, "t": int(time.time() * 1000)})
        time.sleep(0.005)

    got_event = got_prediction.wait(timeout=15.0)
    signer.disconnect()
    assert got_event, "Timed out — no prediction events received at all"
    # Verify structure of the last received prediction.
    last = received[-1]
    assert "ready" in last
    assert "label" in last
    assert "confidence" in last


# 3. Speech caption persists --------------------------------------------

def test_speech_caption_persists(server, fresh_room):
    # A signer must be present so the hearing role isn't the lone member, but
    # `speech` itself only requires the sender's role to be 'hearing'.
    hearing = _join(server, fresh_room, "hearing", "B")
    hearing.emit("speech", {"text": "phase5 acceptance test"})
    time.sleep(0.3)
    hearing.disconnect()

    resp = urllib.request.urlopen(f"{server}/transcript?room_id={fresh_room}")
    messages = json.loads(resp.read())["messages"]
    speech = [m for m in messages if m["source"] == "speech"]
    assert speech
    assert speech[0]["text"] == "phase5 acceptance test"


# 4. Latency p95 --------------------------------------------------------

def test_latency_report_passes():
    assert PHASE5_LATENCY_REPORT.exists(), (
        f"Latency report not found at {PHASE5_LATENCY_REPORT}. "
        "Run: python tests/profile_ws.py"
    )
    report = json.loads(PHASE5_LATENCY_REPORT.read_text())
    assert report.get("passed") is True, (
        f"p95 {report.get('p95_ms')} ms exceeds target {report.get('target_p95_ms')} ms"
    )
    assert report["p95_ms"] < 2000


# 5. Malformed frame ----------------------------------------------------

def test_bad_frame_emits_error(server, fresh_room):
    signer = _join(server, fresh_room, "signer", "A")
    got_error = threading.Event()
    errors: list[dict] = []

    @signer.on("error")
    def on_err(data):
        errors.append(data)
        got_error.set()

    signer.emit("frame", {"landmarks": [0.0] * 10, "t": int(time.time() * 1000)})
    received = got_error.wait(timeout=5.0)
    signer.disconnect()
    assert received and errors[0].get("message")
