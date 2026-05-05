"""Tests for Subtask 4: per-frame normalization."""

import numpy as np
import pytest

from backend.data.normalize import (
    HAND_DIM,
    N_LANDMARKS,
    TWO_HAND_DIM,
    _hand_is_empty,
    _scale_unit_hand,
    _wrist_center_hand,
    interpolate_to_length,
    normalize_frame,
    normalize_sequence,
)

RNG = np.random.default_rng(42)


def _random_hand() -> np.ndarray:
    return RNG.random((N_LANDMARKS, 3)).astype(np.float32)


def _random_frame() -> np.ndarray:
    return RNG.random(TWO_HAND_DIM).astype(np.float32)


def _zero_hand() -> np.ndarray:
    return np.zeros((N_LANDMARKS, 3), dtype=np.float32)


def _zero_right_frame() -> np.ndarray:
    """Frame with a real left hand and a zero-padded right hand."""
    f = np.zeros(TWO_HAND_DIM, dtype=np.float32)
    f[:HAND_DIM] = RNG.random(HAND_DIM).astype(np.float32)
    return f


# ---------------------------------------------------------------------------
# _hand_is_empty
# ---------------------------------------------------------------------------

class TestHandIsEmpty:
    def test_zero_hand(self):
        assert _hand_is_empty(_zero_hand())

    def test_nonzero_hand(self):
        assert not _hand_is_empty(_random_hand())

    def test_single_nonzero(self):
        h = _zero_hand()
        h[5, 1] = 0.001
        assert not _hand_is_empty(h)


# ---------------------------------------------------------------------------
# _wrist_center_hand
# ---------------------------------------------------------------------------

class TestWristCenter:
    def test_wrist_at_origin(self):
        hand = _random_hand()
        centered = _wrist_center_hand(hand)
        np.testing.assert_allclose(centered[0], [0, 0, 0], atol=1e-6)

    def test_shape_preserved(self):
        hand = _random_hand()
        assert _wrist_center_hand(hand).shape == (N_LANDMARKS, 3)

    def test_dtype_preserved(self):
        hand = _random_hand()
        assert _wrist_center_hand(hand).dtype == np.float32

    def test_translation_invariance(self):
        """Translating the raw hand does not change the centred result."""
        hand = _random_hand()
        shift = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        c1 = _wrist_center_hand(hand)
        c2 = _wrist_center_hand(hand + shift)
        np.testing.assert_allclose(c1, c2, atol=1e-5)

    def test_zero_hand_unchanged(self):
        h = _zero_hand()
        np.testing.assert_array_equal(_wrist_center_hand(h), h)

    def test_relative_positions_preserved(self):
        """Differences between landmarks are unchanged after centering."""
        hand = _random_hand()
        centered = _wrist_center_hand(hand)
        original_diff = hand[5] - hand[3]
        centred_diff  = centered[5] - centered[3]
        np.testing.assert_allclose(original_diff, centred_diff, atol=1e-5)


# ---------------------------------------------------------------------------
# _scale_unit_hand
# ---------------------------------------------------------------------------

class TestScaleUnit:
    def test_max_dist_is_one(self):
        hand = _wrist_center_hand(_random_hand())
        scaled = _scale_unit_hand(hand)
        diff = scaled[:, None, :] - scaled[None, :, :]
        max_dist = np.linalg.norm(diff, axis=-1).max()
        assert abs(max_dist - 1.0) < 1e-5, f"Max dist should be ~1, got {max_dist}"

    def test_scale_invariance(self):
        """Scaling the raw hand does not change the unit-scaled result."""
        hand = _wrist_center_hand(_random_hand())
        s1 = _scale_unit_hand(hand)
        s2 = _scale_unit_hand(hand * 3.7)
        np.testing.assert_allclose(s1, s2, atol=1e-5)

    def test_shape_preserved(self):
        hand = _random_hand()
        assert _scale_unit_hand(hand).shape == (N_LANDMARKS, 3)

    def test_dtype_preserved(self):
        hand = _random_hand()
        assert _scale_unit_hand(hand).dtype == np.float32

    def test_zero_hand_unchanged(self):
        h = _zero_hand()
        np.testing.assert_array_equal(_scale_unit_hand(h), h)


# ---------------------------------------------------------------------------
# normalize_frame
# ---------------------------------------------------------------------------

class TestNormalizeFrame:
    def test_output_shape(self):
        assert normalize_frame(_random_frame()).shape == (TWO_HAND_DIM,)

    def test_output_dtype(self):
        assert normalize_frame(_random_frame()).dtype == np.float32

    def test_no_nan(self):
        assert not np.isnan(normalize_frame(_random_frame())).any()

    def test_zero_right_hand_stays_zero(self):
        frame = _zero_right_frame()
        normed = normalize_frame(frame)
        np.testing.assert_array_equal(normed[HAND_DIM:], np.zeros(HAND_DIM))

    def test_wrist_at_origin_left(self):
        frame = _random_frame()
        normed = normalize_frame(frame)
        left_wrist = normed[:3]  # first 3 coords = left wrist x,y,z
        np.testing.assert_allclose(left_wrist, [0, 0, 0], atol=1e-5)

    def test_translation_invariance(self):
        frame = _random_frame()
        shift = np.zeros(TWO_HAND_DIM, dtype=np.float32)
        shift[:HAND_DIM] = 0.4          # shift left hand
        shift[HAND_DIM:] = -0.2         # shift right hand
        np.testing.assert_allclose(
            normalize_frame(frame),
            normalize_frame(frame + shift),
            atol=1e-5,
        )

    def test_scale_invariance(self):
        frame = _random_frame()
        np.testing.assert_allclose(
            normalize_frame(frame),
            normalize_frame(frame * 2.5),
            atol=1e-5,
        )

    def test_wrong_shape_raises(self):
        with pytest.raises(AssertionError):
            normalize_frame(np.zeros(63, dtype=np.float32))


# ---------------------------------------------------------------------------
# normalize_sequence
# ---------------------------------------------------------------------------

class TestNormalizeSequence:
    def _make_seq(self, T=30) -> np.ndarray:
        return RNG.random((T, TWO_HAND_DIM)).astype(np.float32)

    def test_output_shape(self):
        seq = self._make_seq()
        assert normalize_sequence(seq).shape == (30, TWO_HAND_DIM)

    def test_output_dtype(self):
        assert normalize_sequence(self._make_seq()).dtype == np.float32

    def test_no_nan(self):
        assert not np.isnan(normalize_sequence(self._make_seq())).any()

    def test_wrong_shape_raises(self):
        with pytest.raises(AssertionError):
            normalize_sequence(np.zeros((30, 63), dtype=np.float32))

    def test_per_frame_independence(self):
        """Each frame is normalized independently (no cross-frame coupling)."""
        seq = self._make_seq()
        normed_seq = normalize_sequence(seq)
        for i in range(seq.shape[0]):
            expected = normalize_frame(seq[i])
            np.testing.assert_allclose(normed_seq[i], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# interpolate_to_length
# ---------------------------------------------------------------------------

class TestInterpolate:
    def test_noop_when_same_length(self):
        seq = RNG.random((30, TWO_HAND_DIM)).astype(np.float32)
        result = interpolate_to_length(seq, 30)
        np.testing.assert_allclose(result, seq, atol=1e-5)

    def test_upsample_shape(self):
        seq = RNG.random((15, TWO_HAND_DIM)).astype(np.float32)
        assert interpolate_to_length(seq, 30).shape == (30, TWO_HAND_DIM)

    def test_downsample_shape(self):
        seq = RNG.random((60, TWO_HAND_DIM)).astype(np.float32)
        assert interpolate_to_length(seq, 30).shape == (30, TWO_HAND_DIM)

    def test_output_dtype(self):
        seq = RNG.random((20, TWO_HAND_DIM)).astype(np.float32)
        assert interpolate_to_length(seq, 30).dtype == np.float32

    def test_endpoints_preserved(self):
        """First and last frames should be preserved after interpolation."""
        seq = RNG.random((10, TWO_HAND_DIM)).astype(np.float32)
        result = interpolate_to_length(seq, 30)
        np.testing.assert_allclose(result[0], seq[0], atol=1e-5)
        np.testing.assert_allclose(result[-1], seq[-1], atol=1e-5)

    def test_wrong_shape_raises(self):
        with pytest.raises(AssertionError):
            interpolate_to_length(np.zeros((30, 63)), 30)


# ---------------------------------------------------------------------------
# Integration: normalize a real extracted .npy (skipped if absent)
# ---------------------------------------------------------------------------

import os
from pathlib import Path

PROCESSED = Path("data/processed")

class TestNormalizeRealData:
    def test_real_npy_range(self):
        npy_files = list(PROCESSED.rglob("*.npy"))
        if not npy_files:
            pytest.skip("No .npy files in data/processed")

        rng = np.random.default_rng(0)
        sample = rng.choice(npy_files, size=min(10, len(npy_files)), replace=False)

        for path in sample:
            seq = np.load(str(path))
            normed = normalize_sequence(seq)
            assert normed.shape == (30, TWO_HAND_DIM)
            assert not np.isnan(normed).any(), f"NaN in {path}"
            # After normalization values should be roughly in [-2, 2]
            assert normed.min() >= -2.0 and normed.max() <= 2.0, (
                f"{path}: range [{normed.min():.3f}, {normed.max():.3f}] outside [-2, 2]"
            )
