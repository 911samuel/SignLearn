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

import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SmoothingConfig:
    ema_alpha: float = 0.6            # weight on the new observation
    conf_threshold: float = 0.75      # min smoothed top-1 prob to emit a label
    repeat_cooldown_frames: int = 15  # don't re-emit same label within N frames
    stride: int = 1                   # emit every K frames (1 = every frame)
    # Hysteresis: require this many consecutive over-threshold frames before
    # switching to a *new* label. Suppresses one-frame flicker between
    # visually-similar classes (eg. m↔n, u↔v).
    hysteresis_frames: int = 2
    # Adaptive threshold: per-class rolling mean of observed top-1 prob,
    # used to dynamically raise the conf gate for "easy" classes. Disabled
    # when alpha=0.
    adaptive_alpha: float = 0.05
    adaptive_bump_sigmas: float = 0.0  # set >0 to require top_prob > mean + k·std


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
        # Hysteresis: how many consecutive frames have predicted a *candidate*
        # label that differs from the currently-emitted one.
        self._candidate_label: str | None = None
        self._candidate_streak: int = 0
        # Adaptive per-class rolling stats (running mean of top-1 prob).
        self._adaptive_mean = np.zeros(len(class_names), dtype=np.float32)
        self._adaptive_initialized = False
        self.last_used_ts: float = time.time()

    def reset(self) -> None:
        self._ema = None
        self._frames_since_emit = self.cfg.repeat_cooldown_frames + 1
        self._frames_seen = 0
        self._last_label = None
        self._candidate_label = None
        self._candidate_streak = 0
        # Adaptive stats persist across resets so the threshold doesn't reset
        # mid-session if the user briefly disconnects.

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

        self.last_used_ts = time.time()

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

        # Update adaptive per-class rolling mean (only when label fires top-1).
        if self.cfg.adaptive_alpha > 0:
            b = self.cfg.adaptive_alpha
            self._adaptive_mean[top_idx] = (
                b * top_prob + (1.0 - b) * self._adaptive_mean[top_idx]
            )
            self._adaptive_initialized = True

        # Effective confidence gate: global threshold OR adaptive bump.
        gate = self.cfg.conf_threshold
        if (
            self.cfg.adaptive_alpha > 0
            and self.cfg.adaptive_bump_sigmas > 0
            and self._adaptive_initialized
        ):
            adapt_floor = (
                self._adaptive_mean[top_idx]
                + self.cfg.adaptive_bump_sigmas * 0.05  # surrogate σ; rolling std is overkill
            )
            gate = max(gate, float(adapt_floor))

        if top_prob < gate:
            self._candidate_label = None
            self._candidate_streak = 0
            return {"label": None, "confidence": None, "ready": True}

        # Hysteresis: when label differs from the currently-emitted one,
        # require N consecutive frames of agreement before switching.
        # Skip hysteresis on first emission (None → first_label is not a
        # "switch" — there is no prior label to protect against flicker).
        if label != self._last_label:
            if self._last_label is not None:
                if label == self._candidate_label:
                    self._candidate_streak += 1
                else:
                    self._candidate_label = label
                    self._candidate_streak = 1
                if self._candidate_streak < max(1, self.cfg.hysteresis_frames):
                    return None
            # First emission (or transition from suppressed state) falls through.
            self._candidate_label = None
            self._candidate_streak = 0
        else:
            self._candidate_label = None
            self._candidate_streak = 0

        # Repeat-suppression.
        if (label == self._last_label
                and self._frames_since_emit <= self.cfg.repeat_cooldown_frames):
            return None

        self._last_label = label
        self._frames_since_emit = 0
        return {"label": label, "confidence": round(top_prob, 4), "ready": True}


# ---------------------------------------------------------------------------
# Per-connection smoother registry with idle TTL eviction
# ---------------------------------------------------------------------------

class SmootherRegistry:
    """Threaded registry mapping connection IDs → PredictionSmoother.

    Background thread evicts idle smoothers older than ``ttl_seconds`` to
    prevent unbounded memory growth from disconnected users. The eviction
    pass runs at most every ``cleanup_interval`` seconds and is otherwise
    lock-free for the read path.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        cfg: SmoothingConfig | None = None,
        ttl_seconds: float = 300.0,      # 5 minutes idle → evict
        cleanup_interval: float = 60.0,  # check every minute
    ) -> None:
        self._class_names = list(class_names)
        self._cfg = cfg or SmoothingConfig()
        self._ttl = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        self._smoothers: dict[str, PredictionSmoother] = {}
        self._last_cleanup: float = time.time()

    def get(self, connection_id: str) -> PredictionSmoother:
        with self._lock:
            sm = self._smoothers.get(connection_id)
            if sm is None:
                sm = PredictionSmoother(self._class_names, self._cfg)
                self._smoothers[connection_id] = sm
            self._maybe_cleanup_locked()
        return sm

    def drop(self, connection_id: str) -> None:
        with self._lock:
            self._smoothers.pop(connection_id, None)

    def size(self) -> int:
        with self._lock:
            return len(self._smoothers)

    def _maybe_cleanup_locked(self) -> None:
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        cutoff = now - self._ttl
        stale = [k for k, sm in self._smoothers.items() if sm.last_used_ts < cutoff]
        for k in stale:
            del self._smoothers[k]
