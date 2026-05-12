"""Phase 7 — real-time prediction smoothing.

Maintains a per-connection rolling state to:

- Apply an exponential moving average (EMA) over softmax probability vectors,
  damping jitter that comes from frame-to-frame noise.
- Gate emission on a confidence threshold so low-confidence frames don't
  produce caption flicker.
- Suppress repeated identical labels emitted within a short cooldown window
  unless confidence dips and re-rises.

Designed to be a drop-in for :class:`backend.api.inference.FrameBuffer` — the
public ``update(probs)`` returns either ``None`` (suppressed) or a
``{"label", "confidence", "ready"}`` dict matching the existing wire format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SmoothingConfig:
    ema_alpha: float = 0.6            # weight on the new observation
    conf_threshold: float = 0.75      # min smoothed top-1 prob to emit a label
    repeat_cooldown_frames: int = 15  # don't re-emit same label within N frames
    stride: int = 1                   # emit every K frames (1 = every frame)


class PredictionSmoother:
    """Rolling smoother over softmax probability vectors.

    Parameters
    ----------
    class_names:
        Ordered list of class names indexed by the model's output dimension.
    cfg:
        Optional :class:`SmoothingConfig` override.
    """

    def __init__(self, class_names: Sequence[str], cfg: SmoothingConfig | None = None) -> None:
        self.class_names = list(class_names)
        self.cfg = cfg or SmoothingConfig()
        self._ema: np.ndarray | None = None
        self._frames_since_emit: int = self.cfg.repeat_cooldown_frames + 1
        self._frames_seen: int = 0
        self._last_label: str | None = None

    def reset(self) -> None:
        self._ema = None
        self._frames_since_emit = self.cfg.repeat_cooldown_frames + 1
        self._frames_seen = 0
        self._last_label = None

    def update(self, probs: np.ndarray) -> dict | None:
        """Ingest a softmax vector and return a wire-format dict or ``None``.

        Returns ``None`` when the smoother decides to suppress emission
        (low confidence, repeat within cooldown, or off-stride frame).
        """
        probs = np.asarray(probs, dtype=np.float32).ravel()
        if probs.shape[0] != len(self.class_names):
            raise ValueError(
                f"probs has {probs.shape[0]} dims but smoother was built for "
                f"{len(self.class_names)} classes"
            )

        # EMA over probability vectors.
        if self._ema is None:
            self._ema = probs.copy()
        else:
            a = self.cfg.ema_alpha
            self._ema = a * probs + (1.0 - a) * self._ema

        self._frames_seen += 1
        self._frames_since_emit += 1

        # Honor stride.
        if self._frames_seen % max(1, self.cfg.stride) != 0:
            return None

        top_idx = int(np.argmax(self._ema))
        top_prob = float(self._ema[top_idx])
        label = self.class_names[top_idx]

        # Confidence gate.
        if top_prob < self.cfg.conf_threshold:
            return {"label": None, "confidence": None, "ready": True}

        # Repeat-suppression.
        if (label == self._last_label
                and self._frames_since_emit <= self.cfg.repeat_cooldown_frames):
            return None

        self._last_label = label
        self._frames_since_emit = 0
        return {"label": label, "confidence": round(top_prob, 4), "ready": True}
