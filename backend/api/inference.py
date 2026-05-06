"""Inference helpers for the SignLearn backend."""

from __future__ import annotations

from collections import deque

import numpy as np

from backend.api.config import CONFIG
from backend.api.model_loader import run_inference
from backend.data.normalize import normalize_frame


def predict(seq: np.ndarray) -> tuple[str, float]:
    """Normalize *seq* and return ``(label, confidence)``.

    Normalization is applied here (backend is the single source of truth,
    matching the Phase 2 training pipeline) so callers can pass raw landmark
    arrays straight from the WebSocket payload.

    Parameters
    ----------
    seq:
        Float32 array shaped ``(SEQUENCE_LEN, FEATURE_DIM)``.
    """
    normalized = np.stack([normalize_frame(frame) for frame in seq]).astype(np.float32)
    return run_inference(normalized)


class FrameBuffer:
    """Per-connection sliding window of landmark frames.

    Usage::

        buf = FrameBuffer()
        result = buf.push(raw_landmarks_126_floats)
        # result is None until the buffer is full; then it is a dict:
        # {"label": str | None, "confidence": float | None, "ready": True}
    """

    def __init__(self) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=CONFIG.sequence_len)

    @property
    def full(self) -> bool:
        return len(self._buf) == CONFIG.sequence_len

    def push(self, frame: list[float] | np.ndarray) -> dict | None:
        """Append one normalized frame; run inference when the window is full.

        Returns ``None`` while warming up, then a prediction dict once the
        buffer has reached ``SEQUENCE_LEN`` frames.
        """
        arr = np.asarray(frame, dtype=np.float32)
        if arr.shape != (CONFIG.feature_dim,):
            raise ValueError(
                f"Expected frame with {CONFIG.feature_dim} values, got {arr.shape}"
            )
        self._buf.append(normalize_frame(arr))

        if not self.full:
            return None

        seq = np.stack(list(self._buf), axis=0)
        label, confidence = run_inference(seq)

        if confidence < CONFIG.conf_threshold:
            return {"label": None, "confidence": None, "ready": True}
        return {"label": label, "confidence": round(confidence, 4), "ready": True}

    def reset(self) -> None:
        self._buf.clear()
