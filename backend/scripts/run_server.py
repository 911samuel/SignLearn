"""Dev entrypoint for the SignLearn backend.

Usage:
    python backend/scripts/run_server.py

Async mode:
    Defaults to "threading" (Werkzeug dev server + simple-websocket).
    Set SIGNLEARN_ASYNC_MODE=eventlet to use eventlet instead (not
    recommended — eventlet cannot green ONNX RLocks and will hang).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when the script is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.api.app import create_app
from backend.api.config import CONFIG
from backend.api import model_loader


def main() -> None:
    model_loader.load_model()
    app, socketio = create_app()
    socketio.run(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        debug=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
