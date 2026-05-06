"""REST routes for the SignLearn backend."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.api.config import CONFIG
from backend.api import model_loader, storage

bp = Blueprint("api", __name__)


def _model_status() -> dict:
    loaded = model_loader.is_loaded()
    return {
        "model_loaded": loaded,
        "num_classes": len(model_loader.get_class_names()) if loaded else None,
        "model_path": str(CONFIG.model_path),
        "load_error": model_loader.get_load_error(),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@bp.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "sequence_len": CONFIG.sequence_len,
        "feature_dim": CONFIG.feature_dim,
        **_model_status(),
    })


# ---------------------------------------------------------------------------
# Speech-to-text log
# ---------------------------------------------------------------------------

@bp.post("/speech-to-text")
def speech_to_text():
    """Log a speech transcription from the browser's Web Speech API.

    Body::

        {"text": "<transcribed string>"}

    Returns::

        {"id": <int>, "status": "ok"}
    """
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text field is required"}), 400

    row_id = storage.append("speech", text)
    return jsonify({"id": row_id, "status": "ok"}), 201


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

@bp.get("/transcript")
def get_transcript():
    """Return the current session transcript.

    Query params:
        limit (int, default 100) — max number of messages to return.

    Returns::

        {"messages": [{id, ts, source, text, confidence}, ...]}
    """
    try:
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(limit, 1000))
    except ValueError:
        limit = 100

    messages = storage.fetch(limit)
    return jsonify({"messages": messages})


@bp.delete("/transcript")
def delete_transcript():
    """Clear all messages from the transcript (dev convenience).

    Requires ``?confirm=1`` to prevent accidental deletion.
    """
    if request.args.get("confirm") != "1":
        return jsonify({"error": "Pass ?confirm=1 to clear the transcript"}), 400

    count = storage.clear()
    return jsonify({"deleted": count, "status": "ok"})
