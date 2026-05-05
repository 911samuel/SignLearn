"""Tests for Subtask 6: on-the-fly landmark augmentation."""

import numpy as np
import pytest

from backend.data.normalize import TWO_HAND_DIM, HAND_DIM
from backend.data.augment import (
    drop_frames,
    gaussian_noise,
    random_augment,
    rotate,
    scale,
    translate,
)

T = 30
RNG = np.random.default_rng(42)


def _seq() -> np.ndarray:
    return RNG.random((T, TWO_HAND_DIM)).astype(np.float32)


def _one_hand_seq() -> np.ndarray:
    """Left hand only; right slot zero."""
    s = np.zeros((T, TWO_HAND_DIM), dtype=np.float32)
    s[:, :HAND_DIM] = RNG.random((T, HAND_DIM)).astype(np.float32)
    return s


# ---------------------------------------------------------------------------
# Shape / dtype contract — every transform must preserve (T, 126) float32
# ---------------------------------------------------------------------------

class TestShapeContract:
    transforms = [
        ("rotate",          lambda s: rotate(s, 10.0)),
        ("scale",           lambda s: scale(s, 1.1)),
        ("translate",       lambda s: translate(s, np.array([0.02, -0.01], np.float32))),
        ("gaussian_noise",  lambda s: gaussian_noise(s, sigma=0.01)),
        ("drop_frames",     lambda s: drop_frames(s, p=0.1)),
        ("random_augment",  lambda s: random_augment(s, rng=np.random.default_rng(0))),
    ]

    @pytest.mark.parametrize("name,fn", transforms)
    def test_shape_preserved(self, name, fn):
        s = _seq()
        assert fn(s).shape == (T, TWO_HAND_DIM), f"{name} changed shape"

    @pytest.mark.parametrize("name,fn", transforms)
    def test_dtype_float32(self, name, fn):
        s = _seq()
        assert fn(s).dtype == np.float32, f"{name} changed dtype"

    @pytest.mark.parametrize("name,fn", transforms)
    def test_wrong_shape_raises(self, name, fn):
        with pytest.raises(AssertionError):
            fn(np.zeros((T, 63), dtype=np.float32))


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------

class TestRotate:
    def test_identity_at_zero(self):
        s = _seq()
        np.testing.assert_allclose(rotate(s, 0.0), s, atol=1e-5)

    def test_inverse(self):
        """rotate(rotate(x, a), -a) ≈ x"""
        s = _seq()
        np.testing.assert_allclose(rotate(rotate(s, 15.0), -15.0), s, atol=1e-4)

    def test_360_is_identity(self):
        s = _seq()
        np.testing.assert_allclose(rotate(s, 360.0), s, atol=1e-4)

    def test_norm_preserved(self):
        """Rotation should not change the vector norms (Euclidean distances)."""
        s = _seq()
        r = rotate(s, 37.0)
        # compare norms frame-by-frame per landmark
        s_norms = np.linalg.norm(s.reshape(T, -1, 3), axis=-1)
        r_norms = np.linalg.norm(r.reshape(T, -1, 3), axis=-1)
        np.testing.assert_allclose(s_norms, r_norms, atol=1e-4)

    def test_mutates_xy(self):
        """A non-zero rotation must actually change the xy values."""
        s = _seq()
        r = rotate(s, 5.0)
        assert not np.allclose(s[:, :2], r[:, :2], atol=1e-5)

    def test_z_unchanged(self):
        """z coordinate must not be affected."""
        s = _seq()
        r = rotate(s, 30.0)
        # z coords are every 3rd element starting at index 2
        z_orig = s.reshape(T, -1, 3)[:, :, 2]
        z_rot  = r.reshape(T, -1, 3)[:, :, 2]
        np.testing.assert_allclose(z_orig, z_rot, atol=1e-5)


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------

class TestScale:
    def test_identity_at_one(self):
        s = _seq()
        np.testing.assert_allclose(scale(s, 1.0), s, atol=1e-6)

    def test_inverse(self):
        s = _seq()
        np.testing.assert_allclose(scale(scale(s, 2.0), 0.5), s, atol=1e-5)

    def test_magnifies(self):
        s = _seq()
        assert scale(s, 2.0).max() > s.max()

    def test_shrinks(self):
        s = _seq()
        assert scale(s, 0.5).max() < s.max()


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------

class TestTranslate:
    def test_zero_shift_identity(self):
        s = _seq()
        shift = np.zeros(2, dtype=np.float32)
        np.testing.assert_allclose(translate(s, shift), s, atol=1e-6)

    def test_inverse(self):
        s = _seq()
        shift = np.array([0.1, -0.05], np.float32)
        np.testing.assert_allclose(
            translate(translate(s, shift), -shift), s, atol=1e-5
        )

    def test_3d_shift_accepted(self):
        s = _seq()
        shift = np.array([0.01, 0.02, 0.0], np.float32)
        result = translate(s, shift)
        assert result.shape == (T, TWO_HAND_DIM)

    def test_zero_padded_hand_unchanged(self):
        """The right (zero-padded) hand slot must remain zero after translation."""
        s = _one_hand_seq()
        shift = np.array([0.05, 0.05], np.float32)
        result = translate(s, shift)
        np.testing.assert_array_equal(result[:, HAND_DIM:], np.zeros((T, HAND_DIM)))


# ---------------------------------------------------------------------------
# gaussian_noise
# ---------------------------------------------------------------------------

class TestGaussianNoise:
    def test_values_changed(self):
        s = _seq()
        n = gaussian_noise(s, sigma=0.01, rng=np.random.default_rng(1))
        assert not np.allclose(s, n)

    def test_no_nan(self):
        s = _seq()
        n = gaussian_noise(s, sigma=0.01, rng=np.random.default_rng(1))
        assert not np.isnan(n).any()

    def test_sigma_zero_is_identity(self):
        s = _seq()
        n = gaussian_noise(s, sigma=0.0, rng=np.random.default_rng(0))
        np.testing.assert_allclose(s, n, atol=1e-6)

    def test_reproducibility(self):
        s = _seq()
        n1 = gaussian_noise(s, rng=np.random.default_rng(99))
        n2 = gaussian_noise(s, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(n1, n2)

    def test_noise_magnitude(self):
        """Mean absolute noise should be close to sigma * sqrt(2/pi)."""
        s = np.zeros((T, TWO_HAND_DIM), dtype=np.float32)
        sigma = 0.05
        n = gaussian_noise(s, sigma=sigma, rng=np.random.default_rng(7))
        mae = np.abs(n).mean()
        expected = sigma * np.sqrt(2 / np.pi)
        assert abs(mae - expected) < 0.01, f"Noise MAE {mae:.4f} far from {expected:.4f}"


# ---------------------------------------------------------------------------
# drop_frames
# ---------------------------------------------------------------------------

class TestDropFrames:
    def test_p_zero_is_identity(self):
        s = _seq()
        np.testing.assert_array_equal(drop_frames(s, p=0.0), s)

    def test_p_one_zeros_all(self):
        s = _seq()
        dropped = drop_frames(s, p=1.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(dropped, np.zeros_like(s))

    def test_some_frames_zero(self):
        rng = np.random.default_rng(5)
        s = _seq()
        dropped = drop_frames(s, p=0.5, rng=rng)
        zero_rows = (dropped == 0).all(axis=1).sum()
        assert 0 < zero_rows < T, f"Expected some but not all zeros, got {zero_rows}"

    def test_non_dropped_frames_unchanged(self):
        s = _seq()
        dropped = drop_frames(s, p=0.3, rng=np.random.default_rng(3))
        kept = (dropped != 0).any(axis=1)
        np.testing.assert_allclose(dropped[kept], s[kept])


# ---------------------------------------------------------------------------
# random_augment
# ---------------------------------------------------------------------------

class TestRandomAugment:
    def test_no_nan(self):
        s = _seq()
        result = random_augment(s, rng=np.random.default_rng(0))
        assert not np.isnan(result).any()

    def test_reproducibility(self):
        s = _seq()
        r1 = random_augment(s, rng=np.random.default_rng(42))
        r2 = random_augment(s, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        s = _seq()
        r1 = random_augment(s, rng=np.random.default_rng(1))
        r2 = random_augment(s, rng=np.random.default_rng(2))
        assert not np.allclose(r1, r2)

    def test_prob_zero_is_identity(self):
        """All probs set to 0 → output equals input."""
        s = _seq()
        probs = {k: 0.0 for k in ("rotate", "scale", "translate", "noise", "drop")}
        result = random_augment(s, rng=np.random.default_rng(0), probs=probs)
        np.testing.assert_array_equal(result, s)

    def test_all_transforms_applied(self):
        """All probs set to 1.0 → output must differ from input."""
        s = _seq()
        probs = {k: 1.0 for k in ("rotate", "scale", "translate", "noise", "drop")}
        result = random_augment(s, rng=np.random.default_rng(0), probs=probs)
        assert not np.allclose(result, s)
