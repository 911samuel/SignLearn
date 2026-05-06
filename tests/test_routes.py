"""REST route tests — Subtasks 1 and 4."""

from __future__ import annotations

import pytest

from backend.api import storage
from backend.api.app import create_app


@pytest.fixture
def client(tmp_path):
    storage.set_db_path(tmp_path / "test.sqlite")
    app, _socketio = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c
    # Reset to default so other tests are unaffected.
    from backend.api.config import CONFIG
    storage.set_db_path(CONFIG.db_path)


# ---------------------------------------------------------------------------
# Subtask 1
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
# Subtask 4 — /speech-to-text
# ---------------------------------------------------------------------------

def test_speech_to_text_stores_message(client):
    resp = client.post("/speech-to-text", json={"text": "hello world"})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["status"] == "ok"
    assert isinstance(body["id"], int)


def test_speech_to_text_empty_body_returns_400(client):
    resp = client.post("/speech-to-text", json={})
    assert resp.status_code == 400


def test_speech_to_text_missing_json_returns_400(client):
    resp = client.post("/speech-to-text", data="not json", content_type="text/plain")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Subtask 4 — /transcript
# ---------------------------------------------------------------------------

def test_transcript_empty_on_fresh_db(client):
    resp = client.get("/transcript")
    assert resp.status_code == 200
    assert resp.get_json()["messages"] == []


def test_transcript_contains_posted_speech(client):
    client.post("/speech-to-text", json={"text": "nice to meet you"})
    client.post("/speech-to-text", json={"text": "goodbye"})

    resp = client.get("/transcript")
    messages = resp.get_json()["messages"]
    assert len(messages) == 2
    assert messages[0]["source"] == "speech"
    assert messages[0]["text"] == "nice to meet you"
    assert messages[1]["text"] == "goodbye"


def test_transcript_limit_param(client):
    for i in range(5):
        client.post("/speech-to-text", json={"text": f"msg {i}"})

    resp = client.get("/transcript?limit=3")
    messages = resp.get_json()["messages"]
    assert len(messages) == 3


def test_transcript_schema(client):
    client.post("/speech-to-text", json={"text": "test"})
    msg = client.get("/transcript").get_json()["messages"][0]
    assert set(msg.keys()) == {"id", "ts", "source", "text", "confidence"}
    assert msg["source"] == "speech"
    assert msg["confidence"] is None


# ---------------------------------------------------------------------------
# Subtask 4 — DELETE /transcript
# ---------------------------------------------------------------------------

def test_delete_transcript_requires_confirm(client):
    resp = client.delete("/transcript")
    assert resp.status_code == 400


def test_delete_transcript_clears_messages(client):
    client.post("/speech-to-text", json={"text": "to be deleted"})
    resp = client.delete("/transcript?confirm=1")
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] == 1

    remaining = client.get("/transcript").get_json()["messages"]
    assert remaining == []
