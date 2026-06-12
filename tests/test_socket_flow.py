"""WebSocket frame/prediction flow tests (room-scoped)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.api.app import create_app
from backend.api import model_loader
from backend.api.config import CONFIG
from backend.api.rooms import STORE
from backend.model.architectures.lstm import build_lstm
from backend.model.config import TrainConfig, compact_label_map

FIXTURE_DIR = Path("tests/fixtures/processed_mini/train")


def _join_as_signer(client) -> str:
    room = STORE.create()
    client.emit("join_room", {"room_id": room.id, "role": "signer", "name": "tester"})
    # Drain join_ok + room_state events.
    client.get_received()
    return room.id


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Load a tiny model whose output dim matches the actual class count."""
    tmp = tmp_path_factory.mktemp("model_socket")
    cmap = compact_label_map()
    num_classes = len(cmap) if cmap else 2
    cfg = TrainConfig(epochs=0, num_classes=num_classes)
    m = build_lstm(cfg)
    model_path = tmp / "tiny_socket.keras"
    m.save(str(model_path))

    model_loader._reset_for_testing()
    model_loader.load_model(model_path)

    app, socketio = create_app()
    app.config["TESTING"] = True
    test_client = socketio.test_client(app)
    _join_as_signer(test_client)
    yield test_client
    test_client.disconnect()
    model_loader._reset_for_testing()


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
    """After filling the window, inference runs and a prediction event is emitted.

    The test uses a tiny randomly-initialized model, so confidence may be below
    the smoother's gate (conf_threshold=0.75) — we only verify that the inference
    PIPELINE runs (prediction event emitted with expected keys), not that the
    specific ready/label values are non-null.
    """
    from backend.api.model_loader import get_class_names
    if not get_class_names():
        pytest.skip("No class names available — model + label map mismatch")

    client.emit("reset")
    client.get_received()

    frames = _load_fixture_frames(CONFIG.sequence_len)
    for frame in frames:
        client.emit("frame", {"landmarks": frame, "t": 0})

    events = client.get_received()
    pred_events = [e for e in events if e["name"] == "prediction"]
    # At least one prediction event must be emitted once the buffer fills.
    assert len(pred_events) >= 1, "Expected at least one prediction event after filling the window"

    # The last prediction must have the required keys with valid types.
    last = pred_events[-1]["args"][0]
    assert "ready" in last
    assert "label" in last
    assert "confidence" in last
    # If the label is non-null (confidence gate passed), it must be a valid class.
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
