"""Subtask 3: WebSocket frame/prediction flow tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.api.app import create_app
from backend.api.model_loader import load_model
from backend.api.config import CONFIG

FIXTURE_DIR = Path("tests/fixtures/processed_mini/train")


@pytest.fixture(scope="module")
def client():
    load_model()
    app, socketio = create_app()
    app.config["TESTING"] = True
    test_client = socketio.test_client(app)
    yield test_client
    test_client.disconnect()


def _load_fixture_frames(n: int = CONFIG.sequence_len) -> list[list[float]]:
    """Return *n* frames from the first fixture file."""
    npy_files = sorted(FIXTURE_DIR.glob("*.npy"))
    assert npy_files, f"No fixture files in {FIXTURE_DIR}"
    seq = np.load(npy_files[0])             # (30, 126)
    # Repeat the sequence if more frames than available are requested.
    frames = list(seq)
    while len(frames) < n:
        frames += list(seq)
    return [f.tolist() for f in frames[:n]]


def test_connect(client):
    assert client.is_connected()


def test_frame_not_ready_before_window_full(client):
    """First 29 frames should return ready=False."""
    client.emit("reset")
    client.get_received()  # flush reset_ack

    frames = _load_fixture_frames(CONFIG.sequence_len - 1)
    for frame in frames:
        client.emit("frame", {"landmarks": frame, "t": 0})

    events = client.get_received()
    predictions = [e for e in events if e["name"] == "prediction"]
    assert all(not p["args"][0]["ready"] for p in predictions), (
        "Expected ready=False while buffer is warming up"
    )


def test_frame_ready_after_window_full(client):
    """Sending the 30th frame must emit ready=True."""
    from backend.api.model_loader import get_class_names
    if not get_class_names():
        pytest.skip("No class names available — model + label map mismatch (empty data/processed/)")

    client.emit("reset")
    client.get_received()

    frames = _load_fixture_frames(CONFIG.sequence_len)
    for frame in frames:
        client.emit("frame", {"landmarks": frame, "t": 0})

    events = client.get_received()
    ready_events = [
        e for e in events
        if e["name"] == "prediction" and e["args"][0]["ready"]
    ]
    assert len(ready_events) >= 1, "Expected at least one ready=True prediction"

    last = ready_events[-1]["args"][0]
    if last["label"] is not None:
        assert last["label"] in get_class_names()
        assert 0.0 <= last["confidence"] <= 1.0


def test_reset_clears_buffer(client):
    """After reset the next 29 frames should again be ready=False."""
    client.emit("reset")
    client.get_received()

    frames = _load_fixture_frames(CONFIG.sequence_len - 1)
    for frame in frames:
        client.emit("frame", {"landmarks": frame, "t": 0})

    events = client.get_received()
    predictions = [e for e in events if e["name"] == "prediction"]
    assert all(not p["args"][0]["ready"] for p in predictions)


def test_bad_frame_emits_error(client):
    client.emit("frame", {"landmarks": [0.0] * 10, "t": 0})  # wrong length
    events = client.get_received()
    errors = [e for e in events if e["name"] == "error"]
    assert errors, "Expected an 'error' event for malformed landmarks"
