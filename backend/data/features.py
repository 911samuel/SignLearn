"""Phase 3 — engineered features on top of normalized landmark sequences.

The existing :func:`backend.data.normalize.normalize_sequence` produces the
canonical ``(T, 126)`` representation consumed by the WebSocket backend. This
module layers *optional* engineered channels on top — velocity, acceleration,
joint angles — selected via ``TrainConfig.feature_mode``.

Public surface
--------------
- ``FEATURE_MODES``: tuple of supported mode strings.
- ``output_dim(mode)``: feature dimension for a given mode.
- ``apply_feature_mode(seq, mode)``: transform a normalized ``(T, 126)`` to
  the engineered ``(T, D)`` shape for ``mode``.
- ``savgol_smooth(seq, window, polyorder)``: temporal jitter reduction.

All functions are pure numpy and safe to call from inside ``tf.data`` maps via
``tf.numpy_function``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from backend.data.constants import COORDS, FEATURE_DIM, HAND_DIM, N_LANDMARKS

FEATURE_MODES: tuple[str, ...] = (
    "raw",
    "raw+velocity",
    "raw+velocity+angles",
)

# Five finger MCP-PIP-TIP triplets per hand for joint-angle features.
# MediaPipe hand landmark indexing: 0=wrist; thumb 1-4; index 5-8;
# middle 9-12; ring 13-16; pinky 17-20. We pick MCP-PIP-DIP triples.
_FINGER_TRIPLES: list[tuple[int, int, int]] = [
    (1, 2, 3),     # thumb
    (5, 6, 7),     # index
    (9, 10, 11),   # middle
    (13, 14, 15),  # ring
    (17, 18, 19),  # pinky
]
N_ANGLES_PER_HAND = len(_FINGER_TRIPLES)


def output_dim(mode: str) -> int:
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode {mode!r}; expected one of {FEATURE_MODES}")
    if mode == "raw":
        return FEATURE_DIM
    if mode == "raw+velocity":
        return FEATURE_DIM * 2
    # raw + velocity + angles per hand
    return FEATURE_DIM * 2 + N_ANGLES_PER_HAND * 2


def _split_hands(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split (T, 126) into two (T, 21, 3) arrays."""
    T = seq.shape[0]
    left  = seq[:, :HAND_DIM].reshape(T, N_LANDMARKS, COORDS)
    right = seq[:, HAND_DIM:].reshape(T, N_LANDMARKS, COORDS)
    return left, right


def _velocity(seq: np.ndarray) -> np.ndarray:
    """First-order temporal difference, same shape, zero-padded at t=0."""
    v = np.zeros_like(seq)
    v[1:] = seq[1:] - seq[:-1]
    return v


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex *b* of triangle (a, b, c), in radians.

    Shapes: a, b, c each (T, 3). Returns (T,).
    """
    ba = a - b
    bc = c - b
    num = np.einsum("tk,tk->t", ba, bc)
    den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
    cos = np.clip(num / den, -1.0, 1.0)
    return np.arccos(cos).astype(np.float32)


def _hand_angles(hand: np.ndarray) -> np.ndarray:
    """Compute joint angles for one hand. (T, 21, 3) → (T, N_ANGLES_PER_HAND)."""
    cols = []
    for i, j, k in _FINGER_TRIPLES:
        cols.append(_angle(hand[:, i], hand[:, j], hand[:, k]))
    return np.stack(cols, axis=1).astype(np.float32)


def apply_feature_mode(seq: np.ndarray, mode: str) -> np.ndarray:
    """Build engineered features for one *already normalized* sequence.

    Parameters
    ----------
    seq:
        Float32 array shaped ``(T, 126)``.
    mode:
        One of :data:`FEATURE_MODES`.

    Returns
    -------
    Float32 array shaped ``(T, output_dim(mode))``.
    """
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected (T, {FEATURE_DIM}), got {seq.shape}")
    if mode == "raw":
        return seq.astype(np.float32)

    parts = [seq.astype(np.float32)]
    if mode in ("raw+velocity", "raw+velocity+angles"):
        parts.append(_velocity(seq).astype(np.float32))

    if mode == "raw+velocity+angles":
        left, right = _split_hands(seq)
        parts.append(_hand_angles(left))
        parts.append(_hand_angles(right))

    return np.concatenate(parts, axis=1).astype(np.float32)


def savgol_smooth(seq: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along the time axis.

    Used to reduce frame-to-frame landmark jitter prior to feature engineering.
    Falls back to the input unchanged if the sequence is shorter than ``window``.
    """
    if seq.shape[0] < window:
        return seq.astype(np.float32)
    return savgol_filter(seq, window_length=window, polyorder=polyorder, axis=0).astype(np.float32)
