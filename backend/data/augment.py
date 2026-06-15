"""Subtask 6: on-the-fly landmark-level augmentation (rotate, scale, translate, noise, drop).

All transforms operate on (30, 126) float32 sequences (post-normalization).
They are never persisted — applied inside the tf.data pipeline at training time only.
"""

import numpy as np
from scipy.interpolate import interp1d

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
    if seq.ndim != 2 or seq.shape[1] != TWO_HAND_DIM:
        raise ValueError(f"Expected (T, {TWO_HAND_DIM}), got {seq.shape}")


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
    if shift.shape not in ((2,), (3,)):
        raise ValueError(f"shift must be (2,) or (3,), got {shift.shape}")

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


def flip(seq: np.ndarray) -> np.ndarray:
    """Mirror the sequence horizontally to simulate a left-handed signer.

    Negates the x-coordinate of every landmark and swaps the left and right
    hand channels so the model sees a physically consistent mirrored signing.

    Args:
        seq: (T, 126) float32 — normalised, layout [left(63) | right(63)]

    Returns:
        (T, 126) float32
    """
    _assert_seq(seq)
    left, right = _split(seq)   # each (T, 21, 3)
    # Mirror x (index 0) for each hand.
    left_flipped  = left.copy();  left_flipped[..., 0]  *= -1
    right_flipped = right.copy(); right_flipped[..., 0] *= -1
    # Swap channels: what was the right hand becomes the left and vice versa.
    return _merge(right_flipped, left_flipped)


def rotate3d(seq: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Apply a small 3D rotation around the wrist (origin after normalization).

    Yaw  rotates around the y-axis (left/right of camera plane),
    pitch rotates around the x-axis (up/down),
    roll  rotates around the z-axis (in-plane).

    Args:
        seq:        (T, 126) float32
        yaw_deg:    rotation around y-axis, degrees
        pitch_deg:  rotation around x-axis, degrees
        roll_deg:   rotation around z-axis, degrees

    Returns:
        (T, 126) float32
    """
    _assert_seq(seq)
    a, b, c = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(a), np.sin(a)
    cp, sp = np.cos(b), np.sin(b)
    cr, sr = np.cos(c), np.sin(c)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    R = (Rz @ Rx @ Ry).astype(np.float32)

    left, right = _split(seq)

    def _rot_hand(hand: np.ndarray) -> np.ndarray:
        flat = hand.reshape(-1, 3)
        rotated = flat @ R.T
        return rotated.reshape(hand.shape).astype(np.float32)

    return _merge(_rot_hand(left), _rot_hand(right))


def speed_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Resample the time axis to simulate signing at a different speed.

    factor < 1 stretches the gesture (slower); factor > 1 compresses (faster).
    The output is always re-interpolated back to the original sequence length so
    downstream consumers see a fixed (T, 126) shape.

    Args:
        seq:    (T, 126) float32
        factor: speed multiplier (0.8 = 25% slower, 1.25 = 25% faster)

    Returns:
        (T, 126) float32
    """
    _assert_seq(seq)
    T = seq.shape[0]
    if factor <= 0 or T < 2:
        return seq.astype(np.float32)

    # Sample T points from a stretched/compressed time axis, then resample back to T.
    src_len = max(2, int(round(T / factor)))
    src_t = np.linspace(0, 1, src_len)
    new_t = np.linspace(0, 1, T)

    # First, resample the original T frames to src_len frames…
    fn1 = interp1d(np.linspace(0, 1, T), seq, axis=0, kind="linear", fill_value="extrapolate")
    warped = fn1(src_t)
    # …then resample back to T frames.
    fn2 = interp1d(src_t, warped, axis=0, kind="linear", fill_value="extrapolate")
    return fn2(new_t).astype(np.float32)


def time_warp(seq: np.ndarray, factor_range=(0.8, 1.25), n_knots: int = 4, rng=None) -> np.ndarray:
    """Non-uniform temporal warping.

    Generates ``n_knots`` random local speed multipliers in ``factor_range``,
    builds a monotonic cumulative time map from them, and resamples the
    sequence onto that map. Unlike :func:`speed_warp` (uniform stretching),
    this stretches some intervals and compresses others within the same
    clip — simulating prosody/rhythm variation across signers.

    Output shape is always (T, 126).
    """
    _assert_seq(seq)
    T = seq.shape[0]
    if T < 2:
        return seq.astype(np.float32)
    rng = rng or np.random.default_rng()
    lo, hi = factor_range
    # Sample n_knots local speeds, then build a piecewise-linear cumulative
    # warp by integrating their reciprocal (slow speed = stretched time).
    speeds = rng.uniform(lo, hi, size=max(2, n_knots))
    cum = np.concatenate([[0.0], np.cumsum(1.0 / speeds)])
    cum = cum / cum[-1]  # normalize to [0, 1]
    # Resample cum to length T (knots are sparse anchors over the timeline).
    knot_pos = np.linspace(0.0, 1.0, len(cum))
    new_t = np.interp(np.linspace(0.0, 1.0, T), knot_pos, cum)
    fn = interp1d(
        np.linspace(0.0, 1.0, T), seq, axis=0,
        kind="linear", fill_value="extrapolate",
    )
    return fn(new_t).astype(np.float32)


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

# Default probability that each transform is applied.
# Flip defaults to 0 because handedness is linguistically meaningful for many ASL signs.
_DEFAULT_PROBS = {
    "rotate":     0.5,
    "rotate3d":   0.0,  # opt-in via TRAINING_PROBS
    "scale":      0.5,
    "translate":  0.5,
    "noise":      0.8,
    "drop":       0.3,
    "speed_warp": 0.0,  # opt-in via TRAINING_PROBS
    "flip":       0.0,
}

# Aggressive probability profile activated for training runs (Phase 4).
# Opt-in by passing ``probs=TRAINING_PROBS`` to ``random_augment``.
TRAINING_PROBS = {
    "rotate":     0.6,
    "rotate3d":   0.5,
    "scale":      0.5,
    "translate":  0.5,
    "noise":      0.8,
    "drop":       0.3,
    "speed_warp": 0.4,
    "flip":       0.0,
}

# Param ranges — sampled uniformly within each range
_DEFAULT_RANGES = {
    "rotate_deg":   (-10.0, 10.0),
    "rotate3d_deg": (-15.0, 15.0),
    "scale_factor": (0.9, 1.1),
    "translate_xy": (-0.05, 0.05),
    "noise_sigma":   0.01,
    "drop_p":        0.1,
    "speed_factor": (0.8, 1.25),
    "time_warp_factor": (0.8, 1.25),
}


# Named augmentation profiles consumed by train_word_model.py --aug-profile.
# Keep TRAINING_PROBS as the baseline; profiles only diverge from it.
AUG_PROFILES: dict[str, dict] = {
    "baseline":   dict(TRAINING_PROBS),
    "timewarp":   {**TRAINING_PROBS, "time_warp": 0.5},
    # mixup is applied at batch level, not inside random_augment — the
    # per-sample probs here stay equal to baseline.
    "mixup_sameclass": dict(TRAINING_PROBS),
    "timewarp+mixup":  {**TRAINING_PROBS, "time_warp": 0.5},
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

    if rng.random() < probs.get("rotate3d", 0.0):
        lo, hi = ranges["rotate3d_deg"]
        yaw, pitch, roll = (float(rng.uniform(lo, hi)) for _ in range(3))
        out = rotate3d(out, yaw, pitch, roll)

    if rng.random() < probs.get("speed_warp", 0.0):
        lo, hi = ranges["speed_factor"]
        out = speed_warp(out, float(rng.uniform(lo, hi)))

    if rng.random() < probs.get("time_warp", 0.0):
        out = time_warp(out, factor_range=ranges["time_warp_factor"], rng=rng)

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

    if rng.random() < probs["flip"]:
        out = flip(out)

    return out
