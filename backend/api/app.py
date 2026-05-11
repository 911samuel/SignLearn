"""Flask + Flask-SocketIO application factory."""

from __future__ import annotations

import logging
import os

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from backend.api.config import CONFIG
from backend.api.routes import bp as api_bp
from backend.api import socket_handlers

_log = logging.getLogger(__name__)

_INSECURE_DEFAULT = "signlearn-dev"


def create_app() -> tuple[Flask, SocketIO]:
    app = Flask(__name__)
    secret = os.environ.get("SIGNLEARN_SECRET_KEY", "")
    if not secret:
        if os.environ.get("FLASK_DEBUG", "0") == "1":
            secret = _INSECURE_DEFAULT
            _log.warning(
                "SIGNLEARN_SECRET_KEY is not set; using insecure default. "
                "Set the env var before deploying to production."
            )
        else:
            raise RuntimeError(
                "SIGNLEARN_SECRET_KEY environment variable must be set in production. "
                "Run: export SIGNLEARN_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')"
            )
    app.config["SECRET_KEY"] = secret
    CORS(app, origins=list(CONFIG.cors_origins))
    app.register_blueprint(api_bp)

    socketio = SocketIO(
        app,
        cors_allowed_origins=list(CONFIG.cors_origins),
        async_mode=CONFIG.async_mode,
    )
    socket_handlers.register(socketio)
    return app, socketio
