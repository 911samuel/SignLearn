"""Runtime config for the SignLearn backend API.

Single source of truth for ports, paths, sliding-window size, and the
confidence threshold used to gate predictions written to the transcript.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Tests set SIGNLEARN_ASYNC_MODE=threading because eventlet monkey-patching
# conflicts with pytest's threading model and socketio.test_client.
_ASYNC_MODE = os.environ.get("SIGNLEARN_ASYNC_MODE", "eventlet")


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
    model_path: Path = _REPO_ROOT / "artifacts" / "checkpoints" / "lstm_best.keras"
    db_path: Path = _REPO_ROOT / "artifacts" / "signlearn.sqlite"
    sequence_len: int = SEQUENCE_LEN
    feature_dim: int = FEATURE_DIM
    conf_threshold: float = 0.6
    async_mode: str = _ASYNC_MODE


CONFIG = APIConfig()
