"""Singleton LSTM model + label decoder.

Loaded once at first call to :func:`get_model`; subsequent calls return the
cached instance.  A threading lock guards ``model.predict`` so concurrent
SocketIO greenlets don't race on the TensorFlow session.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.api.config import CONFIG
from backend.model.config import compact_class_names

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_model: tf.keras.Model | None = None
_class_names: list[str] | None = None
_load_error: str | None = None


def load_model(path: Path | None = None) -> tf.keras.Model | None:
    """Load and cache the LSTM model from *path* (default: ``CONFIG.model_path``).

    Returns the model on success, ``None`` on failure (error stored in
    :func:`get_load_error`).  Does not raise — the server stays alive even
    when the checkpoint is missing.
    """
    global _model, _class_names, _load_error
    model_path = path or CONFIG.model_path
    with _lock:
        if _model is None and _load_error is None:
            try:
                _model = tf.keras.models.load_model(str(model_path))
                _class_names = compact_class_names()
                _log.info("Model loaded from %s (%d classes)", model_path, len(_class_names))
            except Exception as exc:  # noqa: BLE001
                _load_error = f"{type(exc).__name__}: {exc}"
                _log.error("Failed to load model from %s: %s", model_path, exc)
    return _model


def get_model() -> tf.keras.Model:
    """Return the cached model, loading it on first call.

    Raises
    ------
    RuntimeError:
        If the model checkpoint could not be loaded.
    """
    if _model is None:
        load_model()
    if _model is None:
        raise RuntimeError(
            f"Model not loaded — {_load_error or 'call load_model() first'}"
        )
    return _model


def get_class_names() -> list[str]:
    """Return the vocabulary list in compact-index order."""
    if _class_names is None:
        load_model()
    return _class_names or []


def is_loaded() -> bool:
    return _model is not None


def get_load_error() -> str | None:
    """Return the error message from the last failed load attempt, or ``None``."""
    return _load_error


def run_inference(seq: np.ndarray) -> tuple[str, float]:
    """Run a single sequence through the model under the global lock.

    Parameters
    ----------
    seq:
        Float32 array of shape ``(SEQUENCE_LEN, FEATURE_DIM)``.

    Returns
    -------
    label:
        Predicted class name.
    confidence:
        Softmax probability of the top class, in ``[0, 1]``.

    Raises
    ------
    ValueError:
        If *seq* has the wrong shape.
    """
    expected = (CONFIG.sequence_len, CONFIG.feature_dim)
    if seq.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {seq.shape}")

    model = get_model()
    names = get_class_names()
    if not names:
        raise ValueError(
            "Class name list is empty — no processed data found in data/processed/. "
            "Run generate_test_fixtures.py and train_model.py to populate it."
        )

    with _lock:
        probs = model.predict(seq[np.newaxis], verbose=0)[0]

    idx = int(np.argmax(probs))
    if idx >= len(names):
        raise ValueError(
            f"Model output index {idx} is out of range for class names "
            f"(got {len(names)} names). Model and label map are mismatched."
        )
    return names[idx], float(probs[idx])
