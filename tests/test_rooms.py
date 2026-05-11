"""Room registry + socket-level scoping tests."""

from __future__ import annotations

import pytest

from backend.api import storage
from backend.api.app import create_app
from backend.api.rooms import STORE, RoomStore


# ---------------------------------------------------------------------------
# RoomStore unit tests
# ---------------------------------------------------------------------------

def test_room_create_is_unique_6char_code():
    store = RoomStore()
    a = store.create()
    b = store.create()
    assert a.id != b.id
    assert len(a.id) == 6
    assert a.id.isupper() or a.id.isdigit() or any(c.isdigit() for c in a.id)


def test_add_member_rejects_duplicate_role():
    store = RoomStore()
    room = store.create()
    store.add_member(room.id, "sid1", "signer", "A")
    with pytest.raises(ValueError):
        store.add_member(room.id, "sid2", "signer", "B")


def test_add_member_rejects_unknown_role():
    store = RoomStore()
    room = store.create()
    with pytest.raises(ValueError):
        store.add_member(room.id, "sid1", "spectator", "A")  # type: ignore[arg-type]


def test_room_full_after_two_members():
    store = RoomStore()
    room = store.create()
    store.add_member(room.id, "sid1", "signer", "A")
    store.add_member(room.id, "sid2", "hearing", "B")
    with pytest.raises(ValueError):
        store.add_member(room.id, "sid3", "signer", "C")


def test_peer_sid_returns_other_member():
    store = RoomStore()
    room = store.create()
    store.add_member(room.id, "sid1", "signer", "A")
    store.add_member(room.id, "sid2", "hearing", "B")
    assert room.peer_sid("sid1") == "sid2"
    assert room.peer_sid("sid2") == "sid1"


def test_remove_last_member_drops_room():
    store = RoomStore()
    room = store.create()
    store.add_member(room.id, "sid1", "signer", "A")
    assert store.remove_member("sid1") is None
    assert store.get(room.id) is None


# ---------------------------------------------------------------------------
# Socket-level: prediction is scoped to the room
# ---------------------------------------------------------------------------

@pytest.fixture
def app_and_socketio(tmp_path):
    storage.set_db_path(tmp_path / "test.sqlite")
    app, socketio = create_app()
    app.config["TESTING"] = True
    yield app, socketio
    from backend.api.config import CONFIG
    storage.set_db_path(CONFIG.db_path)


def _join(socketio, app, room_id: str, role: str, name: str):
    client = socketio.test_client(app)
    client.emit("join_room", {"room_id": room_id, "role": role, "name": name})
    client.get_received()  # drain join_ok + room_state
    return client


def test_signer_speech_event_is_rejected(app_and_socketio):
    app, socketio = app_and_socketio
    room = STORE.create()
    signer = _join(socketio, app, room.id, "signer", "A")
    signer.emit("speech", {"text": "this should be ignored"})
    msgs = storage.fetch(room.id)
    assert msgs == []
    signer.disconnect()


def test_hearing_speech_persists_and_broadcasts(app_and_socketio):
    app, socketio = app_and_socketio
    room = STORE.create()
    signer = _join(socketio, app, room.id, "signer", "A")
    hearing = _join(socketio, app, room.id, "hearing", "B")

    hearing.emit("speech", {"text": "hello"})
    signer_events = signer.get_received()
    captions = [e for e in signer_events if e["name"] == "caption"]
    assert captions, "Signer should receive a 'caption' broadcast"
    assert captions[-1]["args"][0]["text"] == "hello"
    assert captions[-1]["args"][0]["source"] == "speech"

    assert [m["text"] for m in storage.fetch(room.id)] == ["hello"]
    signer.disconnect()
    hearing.disconnect()


def test_webrtc_offer_relays_to_peer(app_and_socketio):
    app, socketio = app_and_socketio
    room = STORE.create()
    signer = _join(socketio, app, room.id, "signer", "A")
    hearing = _join(socketio, app, room.id, "hearing", "B")

    signer.emit("webrtc_offer", {"sdp": "fake-sdp"})
    events = hearing.get_received()
    offers = [e for e in events if e["name"] == "webrtc_offer"]
    assert offers and offers[0]["args"][0]["sdp"] == "fake-sdp"
    signer.disconnect()
    hearing.disconnect()


def test_room_state_broadcast_on_join(app_and_socketio):
    app, socketio = app_and_socketio
    room = STORE.create()
    signer = _join(socketio, app, room.id, "signer", "A")
    signer.get_received()  # ignore initial state
    _ = _join(socketio, app, room.id, "hearing", "B")

    events = signer.get_received()
    states = [e for e in events if e["name"] == "room_state"]
    assert states
    roster = states[-1]["args"][0]["members"]
    roles = {m["role"] for m in roster}
    assert roles == {"signer", "hearing"}
