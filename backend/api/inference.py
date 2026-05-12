"""Inference helpers for the SignLearn backend.

Phase 7 added per-connection prediction smoothing on top of the existing
30-frame sliding window. The wire format emitted by :class:`FrameBuffer`
matches the documented protocol:

  ``{"label": str | None, "confidence": float | None, "ready": bool}``

so frontend code keeps working unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from backend.api.config import CONFIG
from backend.api.model_loader import (
    get_class_names,
    run_inference,
    run_inference_probs,
)
from backend.api.smoothing import PredictionSmoother, SmoothingConfig
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
    """Per-connection sliding window of landmark frames with smoothed predictions.

    Usage::

        buf = FrameBuffer()
        result = buf.push(raw_landmarks_126_floats)
        # result is None until the buffer is full or while smoothing is
        # suppressing emission; otherwise a wire-format dict:
        # {"label": str | None, "confidence": float | None, "ready": True}
    """

    def __init__(self, smoothing: SmoothingConfig | None = None) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=CONFIG.sequence_len)
        cfg = smoothing or SmoothingConfig(conf_threshold=CONFIG.conf_threshold)
        # Class names may not be known until the model has loaded; build the
        # smoother lazily on first prediction.
        self._smoothing_cfg = cfg
        self._smoother: PredictionSmoother | None = None

    @property
    def full(self) -> bool:
        return len(self._buf) == CONFIG.sequence_len

    def _ensure_smoother(self) -> PredictionSmoother:
        if self._smoother is None:
            self._smoother = PredictionSmoother(get_class_names(), self._smoothing_cfg)
        return self._smoother

    def push(self, frame: list[float] | np.ndarray) -> dict | None:
        """Append one normalized frame; run inference + smoothing when ready.

        Returns ``None`` while warming up (buffer not yet full) or while the
        smoother decides to suppress emission (off-stride frame, repeat within
        cooldown). Otherwise returns a prediction dict.
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
        probs = run_inference_probs(seq)
        return self._ensure_smoother().update(probs)

    def reset(self) -> None:
        self._buf.clear()
        if self._smoother is not None:
            self._smoother.reset()
