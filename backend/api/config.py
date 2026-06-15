"""Runtime config for the SignLearn backend API.

Single source of truth for ports, paths, sliding-window size, and the
confidence threshold used to gate predictions written to the transcript.

Environment variable overrides
-------------------------------
SIGNLEARN_MODEL_PATH   Path to the .keras or .onnx checkpoint to serve.
                       Default: artifacts/checkpoints/tcn_best.onnx
SIGNLEARN_WORD_MODEL_PATH
                       Optional path to a separate word-recognition checkpoint
                       (.keras or .onnx).  When set, callers may load this
                       additionally via WORD_MODEL_PATH for a parallel word
                       prediction stream.  Default: unset (word model not
                       served; letter/digit model remains the only stream).
SIGNLEARN_SECRET_KEY   Flask secret key (required for session cookies in tests).
SIGNLEARN_ADMIN_TOKEN  Token required by POST /admin/reload.  If unset the
                       endpoint returns 403 (disabled).
SIGNLEARN_ASYNC_MODE   "threading" (tests) or "gevent" (production default).
SIGNLEARN_CORS_ORIGINS Comma-separated allowlist of browser origins.  Defaults
                       to localhost ports for dev; set to your Vercel URL in
                       production (e.g. https://signlearn.vercel.app).
"""

import os
from dataclasses import dataclass
from pathlib import Path

from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Tests set SIGNLEARN_ASYNC_MODE=threading because eventlet monkey-patching
# conflicts with pytest's threading model and socketio.test_client.
_ASYNC_MODE = os.environ.get("SIGNLEARN_ASYNC_MODE", "threading")

# Default model path — override with SIGNLEARN_MODEL_PATH env var.
# Winner: TCN · raw · lr=5e-4 → 97.53% val acc, p95=0.23ms (4889 fps).
_DEFAULT_MODEL = _REPO_ROOT / "artifacts" / "checkpoints" / "tcn_best.onnx"
_MODEL_PATH = Path(os.environ["SIGNLEARN_MODEL_PATH"]) if os.environ.get("SIGNLEARN_MODEL_PATH") else _DEFAULT_MODEL

# Optional word-recognition checkpoint, opt-in via env var.  Unset by default
# so the existing letter/digit serving path is unaffected.
WORD_MODEL_PATH = (
    Path(os.environ["SIGNLEARN_WORD_MODEL_PATH"])
    if os.environ.get("SIGNLEARN_WORD_MODEL_PATH")
    else None
)

_DEFAULT_CORS = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001"
_CORS_ORIGINS = tuple(
    o.strip() for o in os.environ.get("SIGNLEARN_CORS_ORIGINS", _DEFAULT_CORS).split(",") if o.strip()
)


@dataclass(frozen=True)
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 5001
    cors_origins: tuple[str, ...] = _CORS_ORIGINS
    model_path: Path = _MODEL_PATH
    db_path: Path = _REPO_ROOT / "artifacts" / "signlearn.sqlite"
    sequence_len: int = SEQUENCE_LEN
    feature_dim: int = FEATURE_DIM
    conf_threshold: float = 0.6
    async_mode: str = _ASYNC_MODE


CONFIG = APIConfig()
