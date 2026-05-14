"""Typed exceptions and Flask error handlers for SignLearn.

All application-layer errors subclass :class:`SignLearnError`. Flask
error handlers are registered via :func:`register_error_handlers` which
is called from the app factory.

Error → HTTP mapping
--------------------
``LandmarkValidationError``   → 422  (malformed landmarks payload)
``ModelNotReadyError``        → 503  (model failed to load at startup)
``RoomNotFoundError``         → 404  (unknown room_id)
``RateLimitError``            → 429  (client sending too fast)
``SignLearnError`` (generic)  → 500  (catch-all)
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class SignLearnError(Exception):
    """Base class for all application-layer errors."""

    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        d: dict = {"error": self.error_code, "message": self.message}
        if self.detail:
            d["detail"] = self.detail
        return d


class LandmarkValidationError(SignLearnError):
    """Raised when the incoming landmark payload is malformed or out of range."""

    http_status = 422
    error_code = "landmark_validation_error"


class ModelNotReadyError(SignLearnError):
    """Raised when inference is attempted but the model has not loaded."""

    http_status = 503
    error_code = "model_not_ready"


class RoomNotFoundError(SignLearnError):
    """Raised when a room_id is not found in the store."""

    http_status = 404
    error_code = "room_not_found"


class RateLimitError(SignLearnError):
    """Raised when a client is sending frames faster than the pipeline can handle."""

    http_status = 429
    error_code = "rate_limit_exceeded"


# ---------------------------------------------------------------------------
# Flask error handlers
# ---------------------------------------------------------------------------


def register_error_handlers(app: Flask) -> None:
    """Wire up JSON error responses for all :class:`SignLearnError` subtypes.

    Also registers a generic 500 handler so unhandled exceptions return JSON
    rather than HTML tracebacks in production.
    """

    @app.errorhandler(SignLearnError)
    def handle_signlearn_error(exc: SignLearnError):
        _log.error(
            "%s: %s%s",
            type(exc).__name__,
            exc.message,
            f" — {exc.detail}" if exc.detail else "",
        )
        return jsonify(exc.to_dict()), exc.http_status

    @app.errorhandler(422)
    def handle_422(exc):
        return jsonify({"error": "unprocessable_entity", "message": str(exc)}), 422

    @app.errorhandler(404)
    def handle_404(exc):
        return jsonify({"error": "not_found", "message": str(exc)}), 404

    @app.errorhandler(405)
    def handle_405(exc):
        return jsonify({"error": "method_not_allowed", "message": str(exc)}), 405

    @app.errorhandler(500)
    def handle_500(exc):
        _log.exception("Unhandled 500: %s", exc)
        return jsonify({"error": "internal_error", "message": "An unexpected error occurred."}), 500
