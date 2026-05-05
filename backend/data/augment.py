"""Subtask 6: on-the-fly landmark-level augmentation (rotate, scale, translate, noise, drop).

All transforms operate on (30, 126) float32 sequences (post-normalization).
They are never persisted — applied inside the tf.data pipeline at training time only.
"""

import numpy as np

from backend.data.normalize import (
    HAND_DIM,
    N_LANDMARKS,
    TWO_HAND_DIM,
    _hand_is_empty,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(seq: np.ndarray):
    """Split (T, 126) into two (T, 21, 3) arrays: left and right."""
    left  = seq[:, :HAND_DIM].reshape(-1, N_LANDMARKS, 3)
    right = seq[:, HAND_DIM:].reshape(-1, N_LANDMARKS, 3)
    return left, right


def _merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Merge two (T, 21, 3) arrays back into (T, 126)."""
    T = left.shape[0]
    return np.concatenate(
        [left.reshape(T, -1), right.reshape(T, -1)], axis=1
    ).astype(np.float32)


def _assert_seq(seq: np.ndarray) -> None:
    assert seq.ndim == 2 and seq.shape[1] == TWO_HAND_DIM, (
        f"Expected (T, {TWO_HAND_DIM}), got {seq.shape}"
    )


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def rotate(seq: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the x-y plane of each hand around its wrist by angle_deg.

    After normalization the wrist sits at the origin, so this is a pure
    rotation with no translation component.  z is left unchanged.

    Args:
        seq:       (T, 126) float32
        angle_deg: rotation angle in degrees (positive = counter-clockwise)

    Returns:
        (T, 126) float32
    """
    _assert_seq(seq)
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)   # (2, 2)

    left, right = _split(seq)

    def _rot_hand(hand: np.ndarray) -> np.ndarray:
        # hand: (T, 21, 3)
        xy = hand[..., :2]   # (T, 21, 2)
        xy_rot = (R @ xy.reshape(-1, 2).T).T.reshape(hand.shape[0], N_LANDMARKS, 2)
        return np.concatenate([xy_rot, hand[..., 2:3]], axis=-1).astype(np.float32)

    return _merge(_rot_hand(left), _rot_hand(right))


def scale(seq: np.ndarray, factor: float) -> np.ndarray:
    """Scale all landmarks by factor relative to the wrist origin.

    After normalization the wrist is at zero, so multiply is sufficient.

    Args:
        seq:    (T, 126) float32
        factor: multiplicative scale (e.g. 0.9 to 1.1)

    Returns:
        (T, 126) float32
    """
    _assert_seq(seq)
    return (seq * factor).astype(np.float32)


def translate(seq: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Add a (2,) or (3,) shift to every landmark in every frame.

    Zero-padded hand slots (absent hands) are left unchanged so they stay zero.

    Args:
        seq:   (T, 126) float32
        shift: (2,) shift [dx, dy] or (3,) [dx, dy, dz] applied to xy(z)

    Returns:
        (T, 126) float32
    """
    _assert_seq(seq)
    assert shift.shape in ((2,), (3,)), f"shift must be (2,) or (3,), got {shift.shape}"

    left, right = _split(seq)   # each (T, 21, 3)

    def _shift_hand(hand: np.ndarray) -> np.ndarray:
        out = hand.copy()
        # skip empty hand slots (all-zero frames)
        for t in range(out.shape[0]):
            if not _hand_is_empty(out[t]):
                out[t, :, :len(shift)] += shift
        return out.astype(np.float32)

    return _merge(_shift_hand(left), _shift_hand(right))


def gaussian_noise(seq: np.ndarray, sigma: float = 0.01, rng=None) -> np.ndarray:
    """Add independent Gaussian noise to every coordinate.

    Args:
        seq:   (T, 126) float32
        sigma: std-dev of noise (default 0.01 — ~1% of unit-scale range)
        rng:   numpy Generator for reproducibility (default: global random state)

    Returns:
        (T, 126) float32
    """
    _assert_seq(seq)
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, size=seq.shape).astype(np.float32)
    return (seq + noise).astype(np.float32)


def drop_frames(seq: np.ndarray, p: float = 0.1, rng=None) -> np.ndarray:
    """Zero out each frame independently with probability p.

    Shape is preserved — frames are replaced with zeros, not removed.
    This simulates momentary occlusion or dropped detections.

    Args:
        seq: (T, 126) float32
        p:   probability of zeroing each frame (default 0.1)
        rng: numpy Generator

    Returns:
        (T, 126) float32
    """
    _assert_seq(seq)
    rng = rng or np.random.default_rng()
    mask = rng.random(seq.shape[0]) >= p   # True = keep, False = zero
    return (seq * mask[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

# Default probability that each transform is applied
_DEFAULT_PROBS = {
    "rotate":    0.5,
    "scale":     0.5,
    "translate": 0.5,
    "noise":     0.8,
    "drop":      0.3,
}

# Param ranges — sampled uniformly within each range
_DEFAULT_RANGES = {
    "rotate_deg":  (-10.0, 10.0),
    "scale_factor": (0.9, 1.1),
    "translate_xy": (-0.05, 0.05),
    "noise_sigma":  0.01,
    "drop_p":       0.1,
}


def random_augment(
    seq: np.ndarray,
    rng=None,
    probs: dict | None = None,
    ranges: dict | None = None,
) -> np.ndarray:
    """Apply a randomised combination of all augmentations to one sequence.

    Each transform is applied independently with its configured probability.
    Designed to be called inside a tf.data map function.

    Args:
        seq:    (T, 126) float32, normalised
        rng:    numpy Generator (pass a seeded one for reproducibility)
        probs:  override default application probabilities (partial dict OK)
        ranges: override default parameter ranges (partial dict OK)

    Returns:
        (T, 126) float32 — always same shape as input
    """
    _assert_seq(seq)
    rng    = rng or np.random.default_rng()
    probs  = {**_DEFAULT_PROBS,  **(probs  or {})}
    ranges = {**_DEFAULT_RANGES, **(ranges or {})}

    out = seq.copy()

    if rng.random() < probs["rotate"]:
        lo, hi = ranges["rotate_deg"]
        out = rotate(out, float(rng.uniform(lo, hi)))

    if rng.random() < probs["scale"]:
        lo, hi = ranges["scale_factor"]
        out = scale(out, float(rng.uniform(lo, hi)))

    if rng.random() < probs["translate"]:
        lo, hi = ranges["translate_xy"]
        shift = rng.uniform(lo, hi, size=2).astype(np.float32)
        out = translate(out, shift)

    if rng.random() < probs["noise"]:
        out = gaussian_noise(out, sigma=ranges["noise_sigma"], rng=rng)

    if rng.random() < probs["drop"]:
        out = drop_frames(out, p=ranges["drop_p"], rng=rng)

    return out
