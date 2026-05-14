"""Runtime config for the SignLearn backend API.

Single source of truth for ports, paths, sliding-window size, and the
confidence threshold used to gate predictions written to the transcript.

Environment variable overrides
-------------------------------
SIGNLEARN_MODEL_PATH   Path to the .keras or .onnx checkpoint to serve.
                       Default: artifacts/checkpoints/lstm_best.keras
SIGNLEARN_SECRET_KEY   Flask secret key (required for session cookies in tests).
SIGNLEARN_ADMIN_TOKEN  Token required by POST /admin/reload.  If unset the
                       endpoint returns 403 (disabled).
SIGNLEARN_ASYNC_MODE   "threading" (tests) or "eventlet" (production default).
"""

import os
from dataclasses import dataclass
from pathlib import Path

from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Tests set SIGNLEARN_ASYNC_MODE=threading because eventlet monkey-patching
# conflicts with pytest's threading model and socketio.test_client.
_ASYNC_MODE = os.environ.get("SIGNLEARN_ASYNC_MODE", "eventlet")

# Default model path — override with SIGNLEARN_MODEL_PATH env var.
# Set to a .onnx file for ~8× faster CPU inference (make export-onnx first).
_DEFAULT_MODEL = _REPO_ROOT / "artifacts" / "checkpoints" / "lstm_best.keras"
_MODEL_PATH = Path(os.environ["SIGNLEARN_MODEL_PATH"]) if os.environ.get("SIGNLEARN_MODEL_PATH") else _DEFAULT_MODEL


@dataclass(frozen=True)
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 5001
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    )
    model_path: Path = _MODEL_PATH
    db_path: Path = _REPO_ROOT / "artifacts" / "signlearn.sqlite"
    sequence_len: int = SEQUENCE_LEN
    feature_dim: int = FEATURE_DIM
    conf_threshold: float = 0.6
    async_mode: str = _ASYNC_MODE


CONFIG = APIConfig()
