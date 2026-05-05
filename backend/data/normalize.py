"""Subtask 4: wrist-centering, unit scaling, temporal interpolation for landmark sequences."""

import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Layout constants (must match extract.py)
# ---------------------------------------------------------------------------

N_LANDMARKS   = 21
COORDS        = 3         # x, y, z
HAND_DIM      = N_LANDMARKS * COORDS   # 63
TWO_HAND_DIM  = HAND_DIM * 2           # 126
WRIST_IDX     = 0                      # landmark index of the wrist


def _split_hands(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a (126,) frame into two (21, 3) arrays (left, right)."""
    assert frame.shape == (TWO_HAND_DIM,), f"Expected ({TWO_HAND_DIM},), got {frame.shape}"
    left  = frame[:HAND_DIM].reshape(N_LANDMARKS, COORDS)
    right = frame[HAND_DIM:].reshape(N_LANDMARKS, COORDS)
    return left, right


def _merge_hands(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Merge two (21, 3) arrays back into a (126,) frame."""
    return np.concatenate([left.flatten(), right.flatten()], dtype=np.float32)


def _hand_is_empty(hand: np.ndarray) -> bool:
    """True when a hand slot is all zeros (absent / padded)."""
    return not np.any(hand)


# ---------------------------------------------------------------------------
# Per-hand normalization primitives
# ---------------------------------------------------------------------------

def _wrist_center_hand(hand: np.ndarray) -> np.ndarray:
    """Translate so wrist (landmark 0) is at the origin.

    Args:
        hand: (21, 3) float32
    Returns:
        (21, 3) float32 with hand[0] == [0, 0, 0]
    """
    if _hand_is_empty(hand):
        return hand.copy()
    return (hand - hand[WRIST_IDX]).astype(np.float32)


def _scale_unit_hand(hand: np.ndarray) -> np.ndarray:
    """Scale so the max pairwise Euclidean distance among landmarks equals 1.

    Args:
        hand: (21, 3) float32, wrist-centred
    Returns:
        (21, 3) float32 with max inter-landmark distance ≈ 1
    """
    if _hand_is_empty(hand):
        return hand.copy()

    # Compute all pairwise distances efficiently: (N, 1, 3) - (1, N, 3)
    diff = hand[:, None, :] - hand[None, :, :]       # (21, 21, 3)
    dists = np.linalg.norm(diff, axis=-1)             # (21, 21)
    max_dist = dists.max()

    if max_dist < 1e-9:
        return hand.copy()

    return (hand / max_dist).astype(np.float32)


# ---------------------------------------------------------------------------
# Public frame / sequence API
# ---------------------------------------------------------------------------

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Wrist-centre and unit-scale each hand in a (126,) frame independently.

    Zero-padded hand slots are left as zeros.
    """
    left, right = _split_hands(frame)
    left  = _scale_unit_hand(_wrist_center_hand(left))
    right = _scale_unit_hand(_wrist_center_hand(right))
    return _merge_hands(left, right)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply normalize_frame to every frame in a (T, 126) sequence."""
    assert seq.ndim == 2 and seq.shape[1] == TWO_HAND_DIM, (
        f"Expected (T, {TWO_HAND_DIM}), got {seq.shape}"
    )
    return np.stack([normalize_frame(f) for f in seq], axis=0).astype(np.float32)


def interpolate_to_length(seq: np.ndarray, target_len: int = 30) -> np.ndarray:
    """Resample a variable-length (T, 126) sequence to (target_len, 126).

    Uses linear interpolation along the time axis. For static images where
    T == target_len this is a no-op (returns a copy). Implemented for the
    future video path where T may differ.
    """
    assert seq.ndim == 2 and seq.shape[1] == TWO_HAND_DIM, (
        f"Expected (T, {TWO_HAND_DIM}), got {seq.shape}"
    )
    T = seq.shape[0]
    if T == target_len:
        return seq.astype(np.float32)

    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, target_len)
    fn = interp1d(old_t, seq, axis=0, kind="linear", fill_value="extrapolate")
    return fn(new_t).astype(np.float32)
