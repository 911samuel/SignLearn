"""REST routes for the SignLearn backend."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.api.config import CONFIG
from backend.api import model_loader, storage
from backend.api.rooms import STORE

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
# Rooms
# ---------------------------------------------------------------------------

@bp.post("/rooms")
def create_room():
    """Allocate a new room code.

    Returns::

        {"room_id": "ABC123"}
    """
    room = STORE.create()
    return jsonify({"room_id": room.id}), 201


@bp.get("/rooms/<room_id>")
def get_room(room_id: str):
    """Return room state, used by the Join page to disable taken roles."""
    room = STORE.get(room_id.upper())
    if room is None:
        return jsonify({"exists": False, "members": []}), 404
    return jsonify({
        "exists": True,
        "room_id": room.id,
        "members": [{"role": m.role, "name": m.name} for m in room.members.values()],
    })


# ---------------------------------------------------------------------------
# Transcript (per room)
# ---------------------------------------------------------------------------

@bp.get("/transcript")
def get_transcript():
    """Return the transcript for the given room.

    Query params:
        room_id (required) — 6-char room code.
        limit (int, default 100) — max number of messages to return.
    """
    room_id = (request.args.get("room_id") or "").strip().upper()
    if not room_id:
        return jsonify({"error": "room_id query param is required"}), 400

    try:
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(limit, 1000))
    except ValueError:
        limit = 100

    messages = storage.fetch(room_id, limit)
    return jsonify({"messages": messages})


@bp.post("/feedback")
def post_feedback():
    """Record a user feedback message.

    Body (JSON)::

        {
            "category": "bug" | "praise" | "idea" | "accessibility",
            "text":     "What the user wrote",
            "room_id":  "ABC123"   // optional
        }
    """
    data = request.get_json(silent=True) or {}

    category = str(data.get("category", "")).strip()
    text = str(data.get("text", "")).strip()
    room_id = str(data.get("room_id") or "").strip().upper() or None

    if not category:
        return jsonify({"error": "category is required"}), 400
    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) > 4000:
        return jsonify({"error": "text is too long (max 4000 chars)"}), 400

    row_id = storage.append_feedback(category, text, room_id)
    return jsonify({"id": row_id, "status": "ok"}), 201


@bp.post("/corrections")
def post_correction():
    """Record a signer's correction of a model prediction.

    Body (JSON)::

        {
            "room_id":        "ABC123",
            "original_text":  "hello",    // what the model predicted
            "corrected_text": "thank you", // what the signer intended
            "confidence":     0.72         // optional
        }
    """
    data = request.get_json(silent=True) or {}

    room_id = str(data.get("room_id", "")).strip().upper()
    original = str(data.get("original_text", "")).strip()
    corrected = str(data.get("corrected_text", "")).strip()
    confidence = data.get("confidence")

    if not room_id:
        return jsonify({"error": "room_id is required"}), 400
    if not original:
        return jsonify({"error": "original_text is required"}), 400
    if not corrected:
        return jsonify({"error": "corrected_text is required"}), 400
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None

    row_id = storage.append_correction(room_id, original, corrected, confidence)
    return jsonify({"id": row_id, "status": "ok"}), 201


@bp.delete("/transcript")
def delete_transcript():
    """Clear messages for a room (dev convenience).

    Requires ``?confirm=1``. If ``room_id`` is omitted, clears everything.
    """
    if request.args.get("confirm") != "1":
        return jsonify({"error": "Pass ?confirm=1 to clear the transcript"}), 400

    room_id = (request.args.get("room_id") or "").strip().upper() or None
    count = storage.clear(room_id)
    return jsonify({"deleted": count, "status": "ok"})
