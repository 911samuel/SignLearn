"""Flask-SocketIO event handlers for SignLearn rooms.

Wire format (browser ↔ server)::

    join_room  {room_id, role, name}          → server validates, joins room
    room_state {members: [{role, name}], you: {role, name}}  ← broadcast on change

    frame      {landmarks: [126 floats], t}   → only accepted from the Signer
    prediction {label, confidence, ready}     ← broadcast to room (peer + self)

    speech     {text}                         → only accepted from the Hearing user
    caption    {source: 'sign'|'speech', text, name, confidence?, ts}
                                              ← broadcast to room

    webrtc_offer  {sdp}     ┐
    webrtc_answer {sdp}     │ relayed to the peer sid in the same room
    webrtc_ice    {candidate} ┘

    reset                                     → clear Signer's sliding window
    reset_ack                                 ← acknowledgement to caller
"""

from __future__ import annotations

import time

from flask import request
from flask_socketio import SocketIO, emit, join_room as sio_join_room, leave_room as sio_leave_room

from backend.api import storage
from backend.api.inference import FrameBuffer
from backend.api.rooms import STORE

# One FrameBuffer per connected Signer, keyed by session ID.
_buffers: dict[str, FrameBuffer] = {}

# Debounce state: last (label, timestamp) written to DB per connection.
_last_written: dict[str, tuple[str, float]] = {}
_DEBOUNCE_SECS = 1.0


def _maybe_log(sid: str, room_id: str, label: str, confidence: float) -> None:
    now = time.monotonic()
    prev_label, prev_ts = _last_written.get(sid, ("", 0.0))
    if label == prev_label and (now - prev_ts) < _DEBOUNCE_SECS:
        return
    _last_written[sid] = (label, now)
    storage.append(room_id, "sign", label, confidence)


def _broadcast_room_state(socketio: SocketIO, room_id: str) -> None:
    room = STORE.get(room_id)
    if room is None:
        return
    socketio.emit("room_state", {"members": room.roster()}, to=room_id)


def register(socketio: SocketIO) -> None:
    """Attach all socket event handlers to *socketio*."""

    @socketio.on("connect")
    def on_connect():
        # Buffers are allocated lazily on join; nothing to do here.
        pass

    @socketio.on("disconnect")
    def on_disconnect():
        sid = request.sid
        _buffers.pop(sid, None)
        _last_written.pop(sid, None)
        room = STORE.remove_member(sid)
        if room is not None:
            _broadcast_room_state(socketio, room.id)

    # ------------------------------------------------------------------
    # Room lifecycle
    # ------------------------------------------------------------------

    @socketio.on("join_room")
    def on_join(data: dict):
        room_id = (data.get("room_id") or "").strip().upper()
        role = (data.get("role") or "").strip().lower()
        name = (data.get("name") or "").strip()[:40] or "Guest"

        if not room_id:
            emit("join_error", {"message": "room_id is required"})
            return
        if STORE.get(room_id) is None:
            emit("join_error", {"message": f"Room {room_id} does not exist"})
            return

        try:
            member = STORE.add_member(room_id, request.sid, role, name)  # type: ignore[arg-type]
        except (KeyError, ValueError) as exc:
            emit("join_error", {"message": str(exc)})
            return

        sio_join_room(room_id)
        if role == "signer":
            _buffers[request.sid] = FrameBuffer()

        emit("join_ok", {
            "room_id": room_id,
            "you": {"role": member.role, "name": member.name},
        })
        _broadcast_room_state(socketio, room_id)

    @socketio.on("leave_room")
    def on_leave():
        room = STORE.remove_member(request.sid)
        _buffers.pop(request.sid, None)
        _last_written.pop(request.sid, None)
        if room is not None:
            sio_leave_room(room.id)
            _broadcast_room_state(socketio, room.id)

    # ------------------------------------------------------------------
    # Sign frames
    # ------------------------------------------------------------------

    @socketio.on("frame")
    def on_frame(data: dict):
        sid = request.sid
        room = STORE.room_for_sid(sid)
        if room is None:
            return
        member = room.members.get(sid)
        if member is None or member.role != "signer":
            return

        buf = _buffers.get(sid)
        if buf is None:
            buf = _buffers[sid] = FrameBuffer()

        try:
            result = buf.push(data["landmarks"])
        except RuntimeError:
            emit("prediction", {
                "label": None, "confidence": None, "ready": False,
                "error": "model_not_loaded",
            })
            return
        except (KeyError, ValueError) as exc:
            emit("error", {"message": str(exc)})
            return

        if result is None:
            emit("prediction", {"label": None, "confidence": None, "ready": False})
            return

        if result.get("label") is not None:
            _maybe_log(sid, room.id, result["label"], result["confidence"])
            socketio.emit("caption", {
                "source": "sign",
                "text": result["label"],
                "confidence": result["confidence"],
                "name": member.name,
                "ts": int(time.time() * 1000),
            }, to=room.id)

        socketio.emit("prediction", result, to=room.id)

    @socketio.on("reset")
    def on_reset():
        buf = _buffers.get(request.sid)
        if buf is not None:
            buf.reset()
        emit("reset_ack", {})

    # ------------------------------------------------------------------
    # Speech captions
    # ------------------------------------------------------------------

    @socketio.on("speech")
    def on_speech(data: dict):
        sid = request.sid
        room = STORE.room_for_sid(sid)
        if room is None:
            return
        member = room.members.get(sid)
        if member is None or member.role != "hearing":
            return

        text = (data.get("text") or "").strip()
        if not text:
            return

        storage.append(room.id, "speech", text)
        socketio.emit("caption", {
            "source": "speech",
            "text": text,
            "name": member.name,
            "ts": int(time.time() * 1000),
        }, to=room.id)

    # ------------------------------------------------------------------
    # WebRTC signaling (server is a pure relay)
    # ------------------------------------------------------------------

    def _relay(event_name: str, data: dict):
        sid = request.sid
        room = STORE.room_for_sid(sid)
        if room is None:
            return
        peer = room.peer_sid(sid)
        if peer is None:
            return
        socketio.emit(event_name, {**data, "from": sid}, to=peer)

    @socketio.on("webrtc_offer")
    def on_offer(data: dict):
        _relay("webrtc_offer", data)

    @socketio.on("webrtc_answer")
    def on_answer(data: dict):
        _relay("webrtc_answer", data)

    @socketio.on("webrtc_ice")
    def on_ice(data: dict):
        _relay("webrtc_ice", data)
