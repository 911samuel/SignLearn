"""Runtime config for the SignLearn backend API.

Single source of truth for ports, paths, sliding-window size, and the
confidence threshold used to gate predictions written to the transcript.
"""

from dataclasses import dataclass
from pathlib import Path

from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 5001
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173")
    model_path: Path = _REPO_ROOT / "artifacts" / "checkpoints" / "lstm_best.keras"
    db_path: Path = _REPO_ROOT / "artifacts" / "signlearn.sqlite"
    sequence_len: int = SEQUENCE_LEN
    feature_dim: int = FEATURE_DIM
    conf_threshold: float = 0.6
    async_mode: str = "threading"


CONFIG = APIConfig()
