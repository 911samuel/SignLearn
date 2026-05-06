"""Flask + Flask-SocketIO application factory."""

from __future__ import annotations

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from backend.api.config import CONFIG
from backend.api.routes import bp as api_bp
from backend.api import socket_handlers


def create_app() -> tuple[Flask, SocketIO]:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "signlearn-dev"
    CORS(app, origins=list(CONFIG.cors_origins))
    app.register_blueprint(api_bp)

    socketio = SocketIO(
        app,
        cors_allowed_origins=list(CONFIG.cors_origins),
        async_mode=CONFIG.async_mode,
    )
    socket_handlers.register(socketio)
    return app, socketio
