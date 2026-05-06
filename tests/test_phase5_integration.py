"""Phase 5 integration acceptance tests.

Validates the four Phase 5 acceptance criteria from docs/sign_learn.md:

1. /health returns model_loaded: true
2. WebSocket prediction round-trip completes for fixture frames
3. Speech POST persists to /transcript
4. WS round-trip p95 < 2000 ms (from artifacts/reports/phase5_latency.json)

Usage:
    pytest tests/test_phase5_integration.py -v
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


# ---------------------------------------------------------------------------
# Session-scoped server fixture
# ---------------------------------------------------------------------------

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
    """Start the Flask backend for the test session, stop it when done."""
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "backend" / "scripts" / "run_server.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_server(BASE_URL)
    yield BASE_URL
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def clear_transcript(server):
    """Wipe the transcript before each test to prevent cross-test pollution."""
    req = urllib.request.Request(
        f"{server}/transcript?confirm=1", method="DELETE"
    )
    urllib.request.urlopen(req)


# ---------------------------------------------------------------------------
# Acceptance test 1: /health reports model loaded
# ---------------------------------------------------------------------------

def test_health_model_loaded(server):
    """GET /health must return model_loaded: true and include model_path."""
    resp = urllib.request.urlopen(f"{server}/health")
    data = json.loads(resp.read())

    assert data["status"] == "ok", f"Unexpected status: {data['status']}"
    assert data["model_loaded"] is True, (
        f"model_loaded is False — load_error: {data.get('load_error')}"
    )
    assert "model_path" in data, "/health missing model_path field"
    assert data.get("load_error") is None, (
        f"load_error is set: {data['load_error']}"
    )


# ---------------------------------------------------------------------------
# Acceptance test 2: WebSocket prediction round-trip
# ---------------------------------------------------------------------------

def _load_fixture_frames() -> list[list[float]]:
    files = sorted(FIXTURE_DIR.glob("*.npy"))
    assert files, f"No fixture .npy files in {FIXTURE_DIR}"
    return [frame.tolist() for frame in np.load(files[0])]  # 30 frames


def test_ws_prediction_roundtrip(server):
    """Streaming 30 fixture frames must produce a ready prediction."""
    sio = sio_lib.Client()
    ready_event = threading.Event()
    received: list[dict] = []

    @sio.on("prediction")
    def on_pred(data):
        received.append(data)
        if data.get("ready"):
            ready_event.set()

    sio.connect(server)
    sio.emit("reset")
    time.sleep(0.05)

    for frame in _load_fixture_frames():
        sio.emit("frame", {"landmarks": frame, "t": int(time.time() * 1000)})
        time.sleep(0.005)

    got_ready = ready_event.wait(timeout=10.0)
    sio.disconnect()

    assert got_ready, "Timed out waiting for a ready prediction after 30 frames"

    ready_preds = [p for p in received if p.get("ready")]
    assert ready_preds, "No ready=true prediction in received events"

    pred = ready_preds[-1]
    # label may be None if confidence < threshold — that's fine, the pipeline ran
    assert "label" in pred, "Prediction missing 'label' field"
    assert "confidence" in pred, "Prediction missing 'confidence' field"


# ---------------------------------------------------------------------------
# Acceptance test 3: Speech POST persists to transcript
# ---------------------------------------------------------------------------

def test_speech_post_persists(server):
    """POST /speech-to-text must persist and appear in GET /transcript."""
    payload = json.dumps({"text": "phase5 acceptance test"}).encode()
    req = urllib.request.Request(
        f"{server}/speech-to-text",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    assert resp.status == 201, f"POST /speech-to-text returned HTTP {resp.status}"

    transcript_resp = urllib.request.urlopen(f"{server}/transcript")
    messages = json.loads(transcript_resp.read())["messages"]

    speech = [m for m in messages if m["source"] == "speech"]
    assert speech, "No speech entry found in /transcript after POST"
    assert speech[0]["text"] == "phase5 acceptance test"


# ---------------------------------------------------------------------------
# Acceptance test 4: WS latency p95 < 2000 ms
# ---------------------------------------------------------------------------

def test_latency_report_passes():
    """phase5_latency.json must exist and report p95 < 2000 ms."""
    assert PHASE5_LATENCY_REPORT.exists(), (
        f"Latency report not found at {PHASE5_LATENCY_REPORT}. "
        "Run: python tests/profile_ws.py"
    )

    report = json.loads(PHASE5_LATENCY_REPORT.read_text())

    assert report.get("passed") is True, (
        f"p95 latency {report.get('p95_ms')} ms exceeds "
        f"target {report.get('target_p95_ms')} ms"
    )
    assert report["p95_ms"] < 2000, (
        f"p95 = {report['p95_ms']} ms — must be < 2000 ms"
    )


# ---------------------------------------------------------------------------
# Acceptance test 5: model-not-loaded resilience
# ---------------------------------------------------------------------------

def test_model_not_loaded_emits_gracefully(server):
    """When model is loaded, frame with bad shape must emit error not crash."""
    sio = sio_lib.Client()
    error_received = threading.Event()
    error_data: list[dict] = []

    @sio.on("error")
    def on_error(data):
        error_data.append(data)
        error_received.set()

    sio.connect(server)
    sio.emit("reset")
    time.sleep(0.05)

    # Send a frame with wrong feature dimension — should get an error event, not a crash
    bad_frame = [0.0] * 10  # wrong size (expect 126)
    sio.emit("frame", {"landmarks": bad_frame, "t": int(time.time() * 1000)})
    got_error = error_received.wait(timeout=5.0)
    sio.disconnect()

    assert got_error, "Server did not emit 'error' event for malformed frame"
    assert error_data[0].get("message"), "Error event has no message"
