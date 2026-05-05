"""Tests for Subtask 2: pseudo-subject assignment and canonical naming."""

import os
from pathlib import Path
from scipy import stats

from backend.data.extract import assign_subject, canonical_name, subject_to_split

N_SUBJECTS = 11


class TestAssignSubject:
    def test_deterministic(self):
        """Same filename always gives the same subject."""
        assert assign_subject("IMG_1118.JPG") == assign_subject("IMG_1118.JPG")

    def test_range(self):
        """Subject id is always in [1, N_SUBJECTS]."""
        filenames = [f"IMG_{i:04d}.JPG" for i in range(500)]
        ids = [assign_subject(f) for f in filenames]
        assert all(1 <= s <= N_SUBJECTS for s in ids), f"Out-of-range ids: {set(ids)}"

    def test_uniform_distribution(self):
        """Subject assignment should be roughly uniform (chi-square p > 0.05)."""
        filenames = [f"IMG_{i:04d}.JPG" for i in range(1100)]
        ids = [assign_subject(f) for f in filenames]
        counts = [ids.count(s) for s in range(1, N_SUBJECTS + 1)]
        _, p = stats.chisquare(counts)
        assert p > 0.05, f"Uneven subject distribution (p={p:.4f}): {counts}"

    def test_extension_stripped(self):
        """Extension variant must not change the subject id."""
        assert assign_subject("IMG_1118.JPG") == assign_subject("IMG_1118.jpg")
        assert assign_subject("IMG_1118.JPG") == assign_subject("IMG_1118.PNG")

    def test_different_filenames_vary(self):
        """At least two different filenames should produce different subject ids."""
        ids = {assign_subject(f"IMG_{i:04d}.JPG") for i in range(50)}
        assert len(ids) > 1


class TestSubjectToSplit:
    def test_train_subjects(self):
        for s in range(1, 8):
            assert subject_to_split(s) == "train"

    def test_val_subjects(self):
        assert subject_to_split(8) == "val"
        assert subject_to_split(9) == "val"

    def test_test_subjects(self):
        assert subject_to_split(10) == "test"
        assert subject_to_split(11) == "test"

    def test_all_subjects_covered(self):
        splits = {subject_to_split(s) for s in range(1, N_SUBJECTS + 1)}
        assert splits == {"train", "val", "test"}


class TestCanonicalName:
    def test_format(self):
        assert canonical_name("hello", 3, 42) == "hello_s03_0042"

    def test_zero_padding(self):
        name = canonical_name("a", 1, 1)
        assert name == "a_s01_0001"

    def test_large_ids(self):
        name = canonical_name("thank_you", 11, 9999)
        assert name == "thank_you_s11_9999"

    def test_no_spaces(self):
        name = canonical_name("good_morning", 5, 100)
        assert " " not in name


class TestDistributionOnRealData:
    """Integration test: distribution over actual digits dataset (skipped if absent)."""

    DIGITS_DIR = Path("data/raw/digits")

    def test_digits_subject_distribution(self):
        if not self.DIGITS_DIR.exists():
            import pytest
            pytest.skip("data/raw/digits not present")

        filenames = []
        for cls_dir in sorted(self.DIGITS_DIR.iterdir()):
            if cls_dir.is_dir():
                filenames += [f.name for f in cls_dir.iterdir() if f.is_file()]

        ids = [assign_subject(f) for f in filenames]
        counts = [ids.count(s) for s in range(1, N_SUBJECTS + 1)]
        _, p = stats.chisquare(counts)
        print(f"\nDigits subject distribution: {counts}  p={p:.4f}")
        assert p > 0.05, f"Poor distribution (p={p:.4f}): {counts}"
