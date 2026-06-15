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

import logging
import time

from flask import request
from flask_socketio import SocketIO, emit, join_room as sio_join_room, leave_room as sio_leave_room

from backend.api import storage
from backend.api.errors import LandmarkValidationError, ModelNotReadyError, SignLearnError
from backend.api.inference import FrameBuffer
from backend.api.rooms import STORE

_log = logging.getLogger(__name__)

# One FrameBuffer per connected Signer, keyed by session ID.
_buffers: dict[str, FrameBuffer] = {}

# Last-active timestamp per session for idle TTL eviction.
_last_active: dict[str, float] = {}
_IDLE_TTL_SECS = 300.0  # evict FrameBuffers idle for 5 min (belt-and-suspenders)

# Debounce state: last (label, timestamp) written to DB per connection.
_last_written: dict[str, tuple[str, float]] = {}
_DEBOUNCE_SECS = 1.0


def _evict_idle_buffers() -> None:
    """Remove FrameBuffers that haven't received a frame for _IDLE_TTL_SECS.

    Called on every incoming frame to amortize the cost.  Without this,
    clients that close the tab without a clean WS close would leak memory.
    """
    now = time.monotonic()
    stale = [sid for sid, ts in _last_active.items() if (now - ts) > _IDLE_TTL_SECS]
    for sid in stale:
        _buffers.pop(sid, None)
        _last_active.pop(sid, None)
        _last_written.pop(sid, None)


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
        _last_active.pop(sid, None)
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

        # Update last-active and opportunistically evict idle sessions.
        _last_active[sid] = time.monotonic()
        _evict_idle_buffers()

        try:
            result = buf.push(data["landmarks"])
        except ModelNotReadyError as exc:
            emit("prediction", {
                "label": None, "confidence": None, "ready": False,
                "error": exc.error_code,
            })
            return
        except (LandmarkValidationError, SignLearnError) as exc:
            emit("error", {"message": exc.message, "code": exc.error_code})
            return
        except RuntimeError as exc:
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
    # Word capture (hold-to-sign)
    #
    # Client buffers 80 frames locally during a hold gesture and sends the
    # complete (80, 126) sequence in a single 'word_predict' event. Server
    # runs the focused word model and returns top-3 candidates so the UI can
    # surface a pick-from-suggestions affordance (top-5 acc of the word model
    # is 98.8% so this is the high-confidence path).
    # ------------------------------------------------------------------

    @socketio.on("word_predict")
    def on_word_predict(data: dict):
        import numpy as np
        from backend.api.model_loader import run_word_inference_probs, get_word_class_names
        from backend.data.normalize import normalize_sequence

        sid = request.sid
        room = STORE.room_for_sid(sid)
        if room is None:
            return
        member = room.members.get(sid)
        if member is None or member.role != "signer":
            return

        try:
            frames = data.get("frames") or []
            arr = np.asarray(frames, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 126:
                emit("word_prediction", {"error": "expected (T, 126) frames"})
                return
            # Pad or center-crop to 80 frames
            T = arr.shape[0]
            if T < 10:
                emit("word_prediction", {"error": "too few frames (need ≥10)"})
                return
            if T >= 80:
                start = (T - 80) // 2
                arr = arr[start:start + 80]
            else:
                pad = np.zeros((80 - T, 126), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            normalized = normalize_sequence(arr)
            probs = run_word_inference_probs(normalized)
            names = get_word_class_names()
            top_idx = np.argsort(probs)[::-1][:3]
            top3 = [
                {"label": names[int(i)], "confidence": float(probs[int(i)])}
                for i in top_idx if int(i) < len(names)
            ]
        except Exception as exc:  # noqa: BLE001
            # Log the real error server-side, return a user-safe message.
            _log.exception("word_predict failed for sid=%s", sid)
            emit("word_prediction", {"error": "prediction_unavailable"})
            return

        if top3:
            best = top3[0]
            socketio.emit("caption", {
                "source": "sign",
                "text": best["label"],
                "confidence": best["confidence"],
                "name": member.name,
                "ts": int(time.time() * 1000),
                "mode": "word",
                "candidates": top3,
            }, to=room.id)
        emit("word_prediction", {"top3": top3})

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

    @socketio.on("webrtc_ready")
    def on_ready(data: dict | None = None):
        _relay("webrtc_ready", data or {})
