"""Inference helpers for the SignLearn backend.

Phase 7 added per-connection prediction smoothing on top of the existing
30-frame sliding window. The wire format emitted by :class:`FrameBuffer`
matches the documented protocol:

  ``{"label": str | None, "confidence": float | None, "ready": bool}``

so frontend code keeps working unchanged.

Phase 5 adds:
- Landmark validation via :exc:`backend.api.errors.LandmarkValidationError`
- Telemetry emission on every prediction via :data:`backend.api.telemetry.METRICS`
- No-hand-frame tracking (all-zero landmark arrays)
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from backend.api.config import CONFIG
from backend.api.errors import LandmarkValidationError, ModelNotReadyError
from backend.api.model_loader import (
    get_class_names,
    get_load_error,
    is_loaded,
    run_inference,
    run_inference_probs,
)
from backend.api.smoothing import PredictionSmoother, SmoothingConfig
from backend.data.normalize import normalize_frame

# Threshold below which a landmark frame is considered "no-hand detected".
_NO_HAND_THRESHOLD = 1e-6


def _validate_frame(arr: np.ndarray) -> None:
    """Raise :exc:`LandmarkValidationError` for clearly invalid frames."""
    if arr.ndim != 1 or arr.shape[0] != CONFIG.feature_dim:
        raise LandmarkValidationError(
            f"Expected {CONFIG.feature_dim} landmark values, got shape {arr.shape}",
        )
    if not np.all(np.isfinite(arr)):
        raise LandmarkValidationError(
            "Landmark array contains NaN or Inf values",
        )


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
    if not is_loaded():
        raise ModelNotReadyError(
            "Model has not loaded yet",
            detail=get_load_error(),
        )
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

    Raises
    ------
    LandmarkValidationError
        When the incoming frame has the wrong shape or contains non-finite values.
    ModelNotReadyError
        When inference is attempted before the model has loaded.
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
        """Append one landmark frame; run inference + smoothing when ready.

        Validates the frame, tracks no-hand frames, runs inference once the
        30-frame window is full, and emits telemetry on every call.

        Returns ``None`` while warming up (buffer not yet full) or while the
        smoother decides to suppress emission (off-stride frame, repeat within
        cooldown). Otherwise returns a prediction dict.

        Raises
        ------
        LandmarkValidationError
            Malformed or non-finite landmark payload.
        ModelNotReadyError
            Model not yet loaded.
        """
        from backend.api.telemetry import METRICS

        arr = np.asarray(frame, dtype=np.float32)
        _validate_frame(arr)

        # Detect no-hand frames (all zeros from the frontend when MediaPipe
        # finds no hands) — record the metric but still buffer them so the
        # window advances naturally.
        if float(np.max(np.abs(arr))) < _NO_HAND_THRESHOLD:
            METRICS.record_no_hand_frame()
            self._buf.append(arr)
            return {"label": None, "confidence": None, "ready": False}

        self._buf.append(normalize_frame(arr))

        if not self.full:
            return None

        if not is_loaded():
            raise ModelNotReadyError(
                "Model has not loaded yet",
                detail=get_load_error(),
            )

        t0 = time.perf_counter()
        seq = np.stack(list(self._buf), axis=0)
        probs = run_inference_probs(seq)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = self._ensure_smoother().update(probs)

        # Emit telemetry for every completed inference cycle.
        label = result.get("label") if result else None
        confidence = result.get("confidence") if result else None
        METRICS.record_prediction(label=label, confidence=confidence, latency_ms=latency_ms)

        return result

    def reset(self) -> None:
        self._buf.clear()
        if self._smoother is not None:
            self._smoother.reset()
