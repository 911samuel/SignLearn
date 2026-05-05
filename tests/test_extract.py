"""Tests for Subtask 3: landmark extraction."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.data.extract import (
    LANDMARK_DIM,
    SEQUENCE_LENGTH,
    TWO_HAND_DIM,
    canonical_name,
    extract_two_hands,
    to_sequence,
    process_dataset,
)

DIGITS_DIR = Path("data/raw/digits")
SAMPLE_IMAGE = DIGITS_DIR / "0" / "IMG_1118.JPG"


# ---------------------------------------------------------------------------
# to_sequence — no MediaPipe needed
# ---------------------------------------------------------------------------

class TestToSequence:
    def test_shape(self):
        frame = np.random.rand(TWO_HAND_DIM).astype(np.float32)
        seq = to_sequence(frame)
        assert seq.shape == (SEQUENCE_LENGTH, TWO_HAND_DIM)

    def test_dtype(self):
        frame = np.zeros(TWO_HAND_DIM, dtype=np.float32)
        assert to_sequence(frame).dtype == np.float32

    def test_all_frames_identical(self):
        frame = np.random.rand(TWO_HAND_DIM).astype(np.float32)
        seq = to_sequence(frame)
        for i in range(SEQUENCE_LENGTH):
            np.testing.assert_array_equal(seq[i], frame)

    def test_custom_target_len(self):
        frame = np.zeros(TWO_HAND_DIM, dtype=np.float32)
        seq = to_sequence(frame, target_len=10)
        assert seq.shape == (10, TWO_HAND_DIM)

    def test_wrong_shape_raises(self):
        with pytest.raises(AssertionError):
            to_sequence(np.zeros(63, dtype=np.float32))


# ---------------------------------------------------------------------------
# extract_two_hands — uses real MediaPipe (skipped if model absent)
# ---------------------------------------------------------------------------

MODEL_ABSENT = not Path("models/hand_landmarker.task").exists()


@pytest.mark.skipif(MODEL_ABSENT, reason="hand_landmarker.task not present")
class TestExtractTwoHands:
    @pytest.fixture(scope="class")
    def landmarker(self):
        from backend.data.extract import _build_landmarker
        lm = _build_landmarker()
        yield lm
        lm.close()

    @pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="digits dataset not present")
    def test_output_shape(self, landmarker):
        frame = extract_two_hands(SAMPLE_IMAGE, landmarker)
        assert frame.shape == (TWO_HAND_DIM,)

    @pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="digits dataset not present")
    def test_output_dtype(self, landmarker):
        frame = extract_two_hands(SAMPLE_IMAGE, landmarker)
        assert frame.dtype == np.float32

    @pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="digits dataset not present")
    def test_no_nan(self, landmarker):
        frame = extract_two_hands(SAMPLE_IMAGE, landmarker)
        assert not np.isnan(frame).any()

    @pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="digits dataset not present")
    def test_left_right_slots(self, landmarker):
        """Left and right slots are each exactly 63 floats."""
        frame = extract_two_hands(SAMPLE_IMAGE, landmarker)
        assert frame[:LANDMARK_DIM].shape == (LANDMARK_DIM,)
        assert frame[LANDMARK_DIM:].shape == (LANDMARK_DIM,)

    def test_bad_path_raises(self, landmarker):
        with pytest.raises(ValueError, match="Cannot read image"):
            extract_two_hands(Path("nonexistent.jpg"), landmarker)


# ---------------------------------------------------------------------------
# process_dataset integration (skipped unless digits dataset present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DIGITS_DIR.exists(), reason="digits dataset not present")
@pytest.mark.skipif(MODEL_ABSENT, reason="hand_landmarker.task not present")
class TestProcessDataset:
    def test_output_shape_and_dtype(self, tmp_path):
        """Process a small sample and verify every .npy has correct shape/dtype."""
        # Only process class "0", restrict to 5 images for speed
        cls_dir = DIGITS_DIR / "0"
        images = sorted(cls_dir.glob("*.[Jj][Pp][Gg]"))[:5]

        # Build a mini raw dir with symlinks
        mini_raw = tmp_path / "raw" / "0"
        mini_raw.mkdir(parents=True)
        for img in images:
            (mini_raw / img.name).symlink_to(img.resolve())

        out_dir = tmp_path / "processed"
        process_dataset(raw_dir=tmp_path / "raw", out_dir=out_dir, workers=1)

        npy_files = list(out_dir.rglob("*.npy"))
        assert len(npy_files) > 0, "No .npy files produced"

        for f in npy_files:
            arr = np.load(str(f))
            assert arr.shape == (SEQUENCE_LENGTH, TWO_HAND_DIM), (
                f"{f.name}: expected ({SEQUENCE_LENGTH}, {TWO_HAND_DIM}), got {arr.shape}"
            )
            assert arr.dtype == np.float32
            assert not np.isnan(arr).any()

    def test_idempotent(self, tmp_path):
        """Re-running process_dataset skips already-existing files."""
        cls_dir = DIGITS_DIR / "0"
        images = sorted(cls_dir.glob("*.[Jj][Pp][Gg]"))[:2]

        mini_raw = tmp_path / "raw" / "0"
        mini_raw.mkdir(parents=True)
        for img in images:
            (mini_raw / img.name).symlink_to(img.resolve())

        out_dir = tmp_path / "processed"
        process_dataset(raw_dir=tmp_path / "raw", out_dir=out_dir, workers=1)
        first_run_files = {f.stat().st_mtime for f in out_dir.rglob("*.npy")}

        # Second run — mtimes must not change
        process_dataset(raw_dir=tmp_path / "raw", out_dir=out_dir, workers=1)
        second_run_files = {f.stat().st_mtime for f in out_dir.rglob("*.npy")}
        assert first_run_files == second_run_files, "Files were re-written on second run"

    def test_split_dirs_created(self, tmp_path):
        """Output lands in train/, val/, or test/ subdirectories."""
        cls_dir = DIGITS_DIR / "0"
        images = sorted(cls_dir.glob("*.[Jj][Pp][Gg]"))[:20]

        mini_raw = tmp_path / "raw" / "0"
        mini_raw.mkdir(parents=True)
        for img in images:
            (mini_raw / img.name).symlink_to(img.resolve())

        out_dir = tmp_path / "processed"
        process_dataset(raw_dir=tmp_path / "raw", out_dir=out_dir, workers=1)

        splits_used = {f.parent.name for f in out_dir.rglob("*.npy")}
        assert splits_used <= {"train", "val", "test"}
        assert len(splits_used) > 0
