"""Tests for Subtask 5: skeleton visualization (non-interactive, no window opened)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.data.normalize import TWO_HAND_DIM, HAND_DIM
from backend.data.visualize import draw_skeleton, render_sequence, save_gif

RNG = np.random.default_rng(0)
PROCESSED = Path("data/processed")


def _random_frame() -> np.ndarray:
    return RNG.random(TWO_HAND_DIM).astype(np.float32)


def _random_seq(T: int = 30) -> np.ndarray:
    return RNG.random((T, TWO_HAND_DIM)).astype(np.float32)


def _one_hand_frame() -> np.ndarray:
    """Left hand present, right hand zero-padded."""
    f = np.zeros(TWO_HAND_DIM, dtype=np.float32)
    f[:HAND_DIM] = RNG.random(HAND_DIM).astype(np.float32)
    return f


class TestDrawSkeleton:
    def test_output_shape(self):
        img = draw_skeleton(_random_frame(), canvas_size=240)
        assert img.shape == (240, 240, 3)

    def test_output_dtype(self):
        img = draw_skeleton(_random_frame(), canvas_size=240)
        assert img.dtype == np.uint8

    def test_non_black_pixels_exist(self):
        """A frame with a real hand should produce some coloured pixels."""
        img = draw_skeleton(_random_frame(), canvas_size=240)
        assert img.sum() > 0, "Canvas is entirely black — no skeleton drawn"

    def test_empty_frame_is_black(self):
        """An all-zero frame (no hands) should produce a black canvas."""
        frame = np.zeros(TWO_HAND_DIM, dtype=np.float32)
        img = draw_skeleton(frame, canvas_size=240)
        assert img.sum() == 0

    def test_one_hand_frame(self):
        """One-hand frame should still produce a visible skeleton."""
        img = draw_skeleton(_one_hand_frame(), canvas_size=240)
        assert img.sum() > 0

    def test_wrong_shape_raises(self):
        with pytest.raises(AssertionError):
            draw_skeleton(np.zeros(63, dtype=np.float32))

    def test_canvas_size_respected(self):
        for size in [128, 256, 480]:
            img = draw_skeleton(_random_frame(), canvas_size=size)
            assert img.shape == (size, size, 3)

    def test_unnormalized_flag(self):
        """normalized=False should still produce a valid image."""
        img = draw_skeleton(_random_frame(), canvas_size=240, normalized=False)
        assert img.shape == (240, 240, 3)
        assert img.dtype == np.uint8


class TestRenderSequence:
    def test_output_length(self):
        frames = render_sequence(_random_seq(30))
        assert len(frames) == 30

    def test_frame_shapes(self):
        frames = render_sequence(_random_seq(30), canvas_size=200)
        for f in frames:
            assert f.shape == (200, 200, 3)

    def test_static_sign_identical_frames(self):
        """A replicated static-image sequence should produce identical frames."""
        base = _random_frame()
        seq = np.tile(base, (30, 1)).astype(np.float32)
        frames = render_sequence(seq)
        for i in range(1, len(frames)):
            np.testing.assert_array_equal(frames[0], frames[i])


class TestSaveGif:
    def test_gif_created(self):
        seq = _random_seq()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.gif"
            save_gif(seq, out, fps=10)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_gif_is_valid(self):
        from PIL import Image
        seq = _random_seq()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.gif"
            save_gif(seq, out, fps=10)
            img = Image.open(str(out))
            assert img.format == "GIF"


class TestRealData:
    def test_render_real_npy(self):
        npy_files = list(PROCESSED.rglob("*.npy"))
        if not npy_files:
            pytest.skip("No .npy files in data/processed")

        rng = np.random.default_rng(1)
        sample = rng.choice(npy_files, size=min(5, len(npy_files)), replace=False)

        for path in sample:
            seq = np.load(str(path)).astype(np.float32)
            frames = render_sequence(seq, canvas_size=240)
            assert len(frames) == 30
            for img in frames:
                assert img.shape == (240, 240, 3)
                assert img.dtype == np.uint8
