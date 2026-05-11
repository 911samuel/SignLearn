"""WebSocket frame/prediction flow tests (room-scoped)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.api.app import create_app
from backend.api.model_loader import load_model
from backend.api.config import CONFIG
from backend.api.rooms import STORE

FIXTURE_DIR = Path("tests/fixtures/processed_mini/train")


def _join_as_signer(client) -> str:
    room = STORE.create()
    client.emit("join_room", {"room_id": room.id, "role": "signer", "name": "tester"})
    # Drain join_ok + room_state events.
    client.get_received()
    return room.id


@pytest.fixture(scope="module")
def client():
    load_model()
    app, socketio = create_app()
    app.config["TESTING"] = True
    test_client = socketio.test_client(app)
    _join_as_signer(test_client)
    yield test_client
    test_client.disconnect()


def _load_fixture_frames(n: int = CONFIG.sequence_len) -> list[list[float]]:
    npy_files = sorted(FIXTURE_DIR.glob("*.npy"))
    assert npy_files, f"No fixture files in {FIXTURE_DIR}"
    seq = np.load(npy_files[0])
    frames = list(seq)
    while len(frames) < n:
        frames += list(seq)
    return [f.tolist() for f in frames[:n]]


def test_connect(client):
    assert client.is_connected()


def test_frame_not_ready_before_window_full(client):
    client.emit("reset")
    client.get_received()

    frames = _load_fixture_frames(CONFIG.sequence_len - 1)
    for frame in frames:
        client.emit("frame", {"landmarks": frame, "t": 0})

    events = client.get_received()
    predictions = [e for e in events if e["name"] == "prediction"]
    assert predictions, "Expected at least one prediction event"
    assert all(not p["args"][0]["ready"] for p in predictions)


def test_frame_ready_after_window_full(client):
    from backend.api.model_loader import get_class_names
    if not get_class_names():
        pytest.skip("No class names available — model + label map mismatch")

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
    assert len(ready_events) >= 1

    last = ready_events[-1]["args"][0]
    if last["label"] is not None:
        assert last["label"] in get_class_names()
        assert 0.0 <= last["confidence"] <= 1.0


def test_reset_clears_buffer(client):
    client.emit("reset")
    client.get_received()

    frames = _load_fixture_frames(CONFIG.sequence_len - 1)
    for frame in frames:
        client.emit("frame", {"landmarks": frame, "t": 0})

    events = client.get_received()
    predictions = [e for e in events if e["name"] == "prediction"]
    assert all(not p["args"][0]["ready"] for p in predictions)


def test_bad_frame_emits_error(client):
    client.emit("frame", {"landmarks": [0.0] * 10, "t": 0})
    events = client.get_received()
    errors = [e for e in events if e["name"] == "error"]
    assert errors


def test_frame_rejected_without_room(client):
    """A connection with no room membership should not receive predictions."""
    app, socketio = create_app()
    app.config["TESTING"] = True
    standalone = socketio.test_client(app)
    standalone.emit("frame", {"landmarks": [0.0] * CONFIG.feature_dim, "t": 0})
    events = standalone.get_received()
    assert not [e for e in events if e["name"] == "prediction"]
    standalone.disconnect()
