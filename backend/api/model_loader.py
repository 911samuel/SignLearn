"""Singleton LSTM model + label decoder.

Loaded once at first call to :func:`get_model`; subsequent calls return the
cached instance.  A threading lock guards ``model.predict`` so concurrent
SocketIO greenlets don't race on the TensorFlow session.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.api.config import CONFIG
from backend.model.config import compact_class_names

_lock = threading.Lock()
_model: tf.keras.Model | None = None
_class_names: list[str] | None = None


def load_model(path: Path | None = None) -> tf.keras.Model:
    """Load and cache the LSTM model from *path* (default: ``CONFIG.model_path``)."""
    global _model, _class_names
    model_path = path or CONFIG.model_path
    with _lock:
        if _model is None:
            _model = tf.keras.models.load_model(str(model_path))
            _class_names = compact_class_names()
    return _model


def get_model() -> tf.keras.Model:
    """Return the cached model, loading it on first call."""
    if _model is None:
        load_model()
    return _model  # type: ignore[return-value]


def get_class_names() -> list[str]:
    """Return the vocabulary list in compact-index order."""
    if _class_names is None:
        load_model()
    return _class_names  # type: ignore[return-value]


def is_loaded() -> bool:
    return _model is not None


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

    with _lock:
        probs = model.predict(seq[np.newaxis], verbose=0)[0]

    idx = int(np.argmax(probs))
    return names[idx], float(probs[idx])
