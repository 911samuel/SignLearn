"""REST route tests for the room-scoped API."""

from __future__ import annotations

import pytest

from backend.api import storage
from backend.api.app import create_app


@pytest.fixture
def client(tmp_path):
    from backend.api import model_loader
    from backend.api.config import CONFIG
    model_loader._reset_for_testing()
    storage.set_db_path(tmp_path / "test.sqlite")
    app, _socketio = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c
    storage.set_db_path(CONFIG.db_path)


@pytest.fixture
def room_id(client) -> str:
    resp = client.post("/rooms")
    assert resp.status_code == 201
    return resp.get_json()["room_id"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["sequence_len"] == 30
    assert body["feature_dim"] == 126
    assert body["model_loaded"] is False


# ---------------------------------------------------------------------------
# /rooms
# ---------------------------------------------------------------------------

def test_create_room_returns_code(client):
    resp = client.post("/rooms")
    assert resp.status_code == 201
    rid = resp.get_json()["room_id"]
    assert isinstance(rid, str) and len(rid) == 6


def test_get_room_missing_returns_404(client):
    resp = client.get("/rooms/ZZZZZZ")
    assert resp.status_code == 404
    assert resp.get_json()["exists"] is False


def test_get_room_exists(client, room_id):
    resp = client.get(f"/rooms/{room_id}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["exists"] is True
    assert body["room_id"] == room_id
    assert body["members"] == []


# ---------------------------------------------------------------------------
# /transcript (room-scoped)
# ---------------------------------------------------------------------------

def test_transcript_requires_room_id(client):
    resp = client.get("/transcript")
    assert resp.status_code == 400


def test_transcript_empty_for_new_room(client, room_id):
    resp = client.get(f"/transcript?room_id={room_id}")
    assert resp.status_code == 200
    assert resp.get_json()["messages"] == []


def test_transcript_returns_appended_messages(client, room_id):
    storage.append(room_id, "speech", "hello")
    storage.append(room_id, "sign", "thank_you", confidence=0.91)

    msgs = client.get(f"/transcript?room_id={room_id}").get_json()["messages"]
    assert [m["text"] for m in msgs] == ["hello", "thank_you"]
    assert msgs[0]["source"] == "speech"
    assert msgs[1]["confidence"] == pytest.approx(0.91)


def test_transcript_isolated_per_room(client):
    a = client.post("/rooms").get_json()["room_id"]
    b = client.post("/rooms").get_json()["room_id"]
    storage.append(a, "speech", "in A")
    storage.append(b, "speech", "in B")

    in_a = client.get(f"/transcript?room_id={a}").get_json()["messages"]
    in_b = client.get(f"/transcript?room_id={b}").get_json()["messages"]
    assert [m["text"] for m in in_a] == ["in A"]
    assert [m["text"] for m in in_b] == ["in B"]


def test_delete_transcript_requires_confirm(client, room_id):
    resp = client.delete(f"/transcript?room_id={room_id}")
    assert resp.status_code == 400


def test_delete_transcript_scoped_to_room(client):
    a = client.post("/rooms").get_json()["room_id"]
    b = client.post("/rooms").get_json()["room_id"]
    storage.append(a, "speech", "in A")
    storage.append(b, "speech", "in B")

    resp = client.delete(f"/transcript?room_id={a}&confirm=1")
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] == 1
    assert client.get(f"/transcript?room_id={b}").get_json()["messages"]
