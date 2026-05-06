"""Flask-SocketIO event handlers for real-time landmark streaming."""

from __future__ import annotations

import time

from flask import request
from flask_socketio import SocketIO, emit

from backend.api import storage
from backend.api.inference import FrameBuffer

# One FrameBuffer per connected client, keyed by session ID.
_buffers: dict[str, FrameBuffer] = {}

# Debounce state: last (label, timestamp) written to DB per connection.
# Same label within 1 second is collapsed into a single transcript row.
_last_written: dict[str, tuple[str, float]] = {}
_DEBOUNCE_SECS = 1.0


def _maybe_log(sid: str, label: str, confidence: float) -> None:
    """Write *label* to the transcript, deduplicated within the debounce window."""
    now = time.monotonic()
    prev_label, prev_ts = _last_written.get(sid, ("", 0.0))
    if label == prev_label and (now - prev_ts) < _DEBOUNCE_SECS:
        return
    _last_written[sid] = (label, now)
    storage.append("sign", label, confidence)


def register(socketio: SocketIO) -> None:
    """Attach all socket event handlers to *socketio*."""

    @socketio.on("connect")
    def on_connect():
        _buffers[request.sid] = FrameBuffer()
        _last_written.pop(request.sid, None)

    @socketio.on("disconnect")
    def on_disconnect():
        _buffers.pop(request.sid, None)
        _last_written.pop(request.sid, None)

    @socketio.on("frame")
    def on_frame(data: dict):
        """Receive one landmark frame from the client.

        Expected payload::

            {"landmarks": [<126 floats>], "t": <unix ms>}

        Emits ``prediction`` back to the sender::

            {"label": str | null, "confidence": float | null, "ready": bool}
        """
        buf = _buffers.get(request.sid)
        if buf is None:
            return

        try:
            result = buf.push(data["landmarks"])
        except RuntimeError:
            # Model checkpoint missing or failed to load — stay alive, tell the client
            emit("prediction", {"label": None, "confidence": None, "ready": False,
                                "error": "model_not_loaded"})
            return
        except (KeyError, ValueError) as exc:
            emit("error", {"message": str(exc)})
            return

        if result is None:
            emit("prediction", {"label": None, "confidence": None, "ready": False})
        else:
            if result.get("label") is not None:
                _maybe_log(request.sid, result["label"], result["confidence"])
            emit("prediction", result)

    @socketio.on("reset")
    def on_reset():
        """Clear the sliding window for this connection."""
        buf = _buffers.get(request.sid)
        if buf is not None:
            buf.reset()
        emit("reset_ack", {})
