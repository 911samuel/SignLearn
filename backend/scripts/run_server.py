"""Dev entrypoint for the SignLearn backend.

Usage:
    python backend/scripts/run_server.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# eventlet monkey-patch must happen before any other network imports,
# but only when actually using eventlet (not threading mode used in tests).
if os.environ.get("SIGNLEARN_ASYNC_MODE", "eventlet") == "eventlet":
    import eventlet
    eventlet.monkey_patch()

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
