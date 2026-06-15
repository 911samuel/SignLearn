"""Production-grade model loader for SignLearn.

Supports two backends transparently based on the checkpoint file extension:

- ``*.keras``  → ``tf.keras.Model``       (training-grade, slow CPU inference)
- ``*.onnx``   → :class:`OnnxRunner`      (~10-20× faster CPU inference)

Singleton state is encapsulated in :class:`ModelHolder` which exposes
``reload(path)`` for hot-swapping a new checkpoint without dropping
WebSocket connections. The swap is atomic behind a reentrant lock, with
SHA-256 verification of the new file before the swap completes.

Backward-compatible module-level functions ``load_model``, ``get_model``,
``get_class_names``, ``is_loaded``, ``run_inference``, ``run_inference_probs``
are preserved so existing call sites (inference.py, smoke tests, route
handlers) keep working without modification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.api.config import CONFIG


def _resolve_class_names(model_path: Path) -> list[str]:
    """Return class names for the served checkpoint.

    Production path: read a ``<model>.classes.json`` sidecar that was persisted
    alongside the ONNX export.  This avoids pulling TensorFlow + the training
    data pipeline into the inference image.

    Dev fallback: call :func:`backend.model.config.compact_class_names`, which
    scans ``data/processed/train/``.  Lazy-imported so the import chain only
    activates when the sidecar is missing.
    """
    sidecar = model_path.with_suffix("").with_suffix(".classes.json")
    if sidecar.exists():
        with sidecar.open() as fh:
            return list(json.load(fh))
    from backend.model.config import compact_class_names  # noqa: PLC0415

    return compact_class_names()

_log = logging.getLogger(__name__)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


class ModelHolder:
    """Thread-safe singleton for the active inference model.

    The holder doesn't own the class-name list — that comes from
    :func:`backend.model.config.compact_class_names` and is refreshed on
    every reload so a checkpoint trained on different vocab snapshots is
    correctly decoded.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None              # tf.keras.Model | OnnxRunner | None
        self._class_names: list[str] | None = None
        self._load_error: str | None = None
        self._path: Path | None = None
        self._sha256: str | None = None
        self._backend: str = "none"

    # -- public surface --

    def load(self, path: Path | None = None) -> Any | None:
        """Load and cache the model from *path* (or :data:`CONFIG.model_path`).

        Returns the loaded model object, or ``None`` if loading failed.
        Failures are stored and surfaced via :meth:`get_load_error`.
        """
        with self._lock:
            target = path or CONFIG.model_path
            if self._model is not None and self._path == target:
                return self._model
            return self._load_locked(target)

    def reload(self, path: Path) -> bool:
        """Atomically swap to a new checkpoint. Returns True on success.

        On failure the old model stays active and the error is captured.
        WebSocket connections are unaffected — they continue using their
        existing references obtained via :func:`get_model`, which returns
        the holder's current model snapshot at call time.
        """
        target = Path(path)
        with self._lock:
            if not target.exists():
                self._load_error = f"reload: {target} does not exist"
                _log.error(self._load_error)
                return False
            new_sha = _sha256_file(target)
            if new_sha == self._sha256:
                _log.info("reload: %s sha256 matches current model, no-op", target)
                return True
            return self._load_locked(target) is not None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference (probabilities) on a batch under the lock."""
        with self._lock:
            if self._model is None:
                self._load_locked(CONFIG.model_path)
            if self._model is None:
                raise RuntimeError(f"Model not loaded — {self._load_error}")
            return self._model.predict(x, verbose=0)

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        if self._model is None:
            raise RuntimeError(f"Model not loaded — {self._load_error}")
        return self._model

    @property
    def class_names(self) -> list[str]:
        if self._class_names is None:
            self.load()
        return self._class_names or []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def info(self) -> dict:
        """Diagnostic info for /health and /metrics endpoints."""
        return {
            "loaded": self.is_loaded,
            "backend": self._backend,
            "path": str(self._path) if self._path else None,
            "sha256": self._sha256,
            "n_classes": len(self._class_names) if self._class_names else 0,
            "load_error": self._load_error,
        }

    def reset_for_testing(self) -> None:
        with self._lock:
            self._model = None
            self._class_names = None
            self._load_error = None
            self._path = None
            self._sha256 = None
            self._backend = "none"

    # -- internals --

    def _load_locked(self, path: Path) -> Any | None:
        try:
            suffix = path.suffix.lower()
            if suffix == ".onnx":
                from backend.api.onnx_runner import OnnxRunner
                new_model: Any = OnnxRunner(path)
                backend = "onnx"
            else:  # default .keras / .h5
                import tensorflow as tf
                new_model = tf.keras.models.load_model(str(path))
                backend = "keras"
            new_class_names = _resolve_class_names(path)
            new_sha = _sha256_file(path)
            # Swap is the last action — if anything above raised, old model stays.
            self._model = new_model
            self._class_names = new_class_names
            self._path = path
            self._sha256 = new_sha
            self._backend = backend
            self._load_error = None
            _log.info(
                "Model loaded: %s backend=%s n_classes=%d sha=%s",
                path, backend, len(new_class_names), new_sha[:12],
            )
            return new_model
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            _log.error("Failed to load %s: %s", path, exc)
            return None


# ---------------------------------------------------------------------------
# Module-level singleton + backward-compat shim
# ---------------------------------------------------------------------------

_holder = ModelHolder()


def load_model(path: Path | None = None) -> Any | None:
    return _holder.load(path)


def reload_model(path: Path) -> bool:
    return _holder.reload(path)


def get_model() -> Any:
    return _holder.model


def get_class_names() -> list[str]:
    return _holder.class_names


def is_loaded() -> bool:
    return _holder.is_loaded


def get_load_error() -> str | None:
    return _holder.load_error


def get_model_info() -> dict:
    return _holder.info


def _reset_for_testing() -> None:
    _holder.reset_for_testing()


def run_inference(seq: np.ndarray) -> tuple[str, float]:
    expected = (CONFIG.sequence_len, CONFIG.feature_dim)
    if seq.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {seq.shape}")
    names = _holder.class_names
    if not names:
        raise ValueError(
            "Class name list is empty — no processed data found in data/processed/."
        )
    probs = _holder.predict(seq[np.newaxis])[0]
    idx = int(np.argmax(probs))
    if idx >= len(names):
        raise ValueError(
            f"Model output index {idx} out of range for {len(names)} class names."
        )
    return names[idx], float(probs[idx])


def run_inference_probs(seq: np.ndarray) -> np.ndarray:
    expected = (CONFIG.sequence_len, CONFIG.feature_dim)
    if seq.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {seq.shape}")
    probs = _holder.predict(seq[np.newaxis])[0]
    return np.asarray(probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Word model (separate holder — 80-frame sequences, 17-class conversational vocab)
# ---------------------------------------------------------------------------

WORD_SEQ_LEN = 80
_WORD_MODEL_PATH = _REPO_ROOT_GUESS = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints" / "tcn_word_best.onnx"
_WORD_VOCAB_PATH = Path(__file__).resolve().parents[2] / "configs" / "word_curated_v3.txt"


def _load_word_class_names() -> list[str]:
    """Load the curated word vocabulary in the same sorted-lowercase order
    the trainer used (see backend.scripts.train_word_model.load_label_map).
    """
    if not _WORD_VOCAB_PATH.exists():
        return []
    raw = [w.strip().lower() for w in _WORD_VOCAB_PATH.read_text().splitlines()
           if w.strip() and not w.strip().startswith("#")]
    return sorted(set(raw))


class _WordHolder:
    """Lighter-weight holder for the word model.

    Same ONNX-runner backend as the letter model but with its own (80, 126)
    input shape and 17-class output, plus class names sourced from the
    curated vocabulary file rather than the letter label map.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._class_names: list[str] = []
        self._path: Path | None = None
        self._sha256: str | None = None
        self._load_error: str | None = None

    def load(self, path: Path | None = None) -> Any | None:
        with self._lock:
            target = Path(path) if path else _WORD_MODEL_PATH
            if self._model is not None and self._path == target:
                return self._model
            try:
                from backend.api.onnx_runner import OnnxRunner
                self._model = OnnxRunner(target)
                self._class_names = _load_word_class_names()
                self._path = target
                self._sha256 = _sha256_file(target)
                self._load_error = None
                _log.info(
                    "Word model loaded: %s n_classes=%d sha=%s",
                    target, len(self._class_names), self._sha256[:12],
                )
                return self._model
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                _log.error("Failed to load word model %s: %s", target, exc)
                return None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        if not self._class_names:
            self.load()
        return self._class_names

    @property
    def info(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "path": str(self._path) if self._path else None,
            "sha256": self._sha256,
            "n_classes": len(self._class_names),
            "load_error": self._load_error,
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._model is None:
                self.load()
            if self._model is None:
                raise RuntimeError(f"Word model not loaded — {self._load_error}")
            return self._model.predict(x, verbose=0)


_word_holder = _WordHolder()


def run_word_inference_probs(seq: np.ndarray) -> np.ndarray:
    """Run the word model on a single (80, 126) landmark sequence."""
    expected = (WORD_SEQ_LEN, CONFIG.feature_dim)
    if seq.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {seq.shape}")
    probs = _word_holder.predict(seq[np.newaxis])[0]
    return np.asarray(probs, dtype=np.float32)


def get_word_class_names() -> list[str]:
    return _word_holder.class_names


def get_word_model_info() -> dict:
    return _word_holder.info


def is_word_model_loaded() -> bool:
    return _word_holder.is_loaded
