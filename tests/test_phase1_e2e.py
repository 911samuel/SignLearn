"""Subtask 9: end-to-end Phase 1 pipeline smoke test using the mini fixture dataset.

Runs the full chain:
  label_map → extract (fixture images) → validate → dataset loader

Skipped when hand_landmarker.task is absent (CI downloads it; local devs need
to run the curl once as documented in CLAUDE.md).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

FIXTURE_RAW  = Path("tests/fixtures/raw_mini")
MODEL_ABSENT = not Path("models/hand_landmarker.task").exists()


@pytest.mark.skipif(MODEL_ABSENT, reason="hand_landmarker.task not present")
class TestPhase1EndToEnd:

    @pytest.fixture(scope="class")
    def pipeline_output(self, tmp_path_factory):
        """Run the full Phase 1 pipeline once and return output paths."""
        tmp = tmp_path_factory.mktemp("phase1_e2e")
        processed = tmp / "processed"
        artifacts = tmp / "artifacts"

        # 1 — label_map (already built; load from committed file)
        from backend.data.label_map import load_label_map
        label_map = load_label_map()
        assert len(label_map) > 0

        # 2 — extract
        from backend.data.extract import process_dataset
        process_dataset(
            raw_dir=FIXTURE_RAW,
            out_dir=processed,
            workers=1,
        )

        # 3 — validate
        from backend.data.validate import validate
        errors = validate(processed_dir=processed, artifacts_dir=artifacts)

        return {
            "processed": processed,
            "artifacts": artifacts,
            "errors": errors,
            "label_map": label_map,
        }

    # ------------------------------------------------------------------ #
    # Extraction                                                           #
    # ------------------------------------------------------------------ #

    def test_pipeline_did_not_crash(self, pipeline_output):
        """The pipeline must complete (even with 0 hands detected) without raising."""
        # errors is [] when validation passes; the fixture may produce 0 files
        # because synthetic images lack detectable hands — that is expected.
        assert isinstance(pipeline_output["errors"], list)

    def test_failure_log_written_on_no_hand(self, pipeline_output):
        """When MediaPipe finds no hands the failure log must be written."""
        npy_files = list(pipeline_output["processed"].rglob("*.npy"))
        if npy_files:
            pass   # some hands detected — no failure log required
        else:
            log = Path("artifacts/extract_failures.log")
            assert log.exists(), "Failure log missing despite 0 extracted files"

    def test_npy_shapes_when_present(self, pipeline_output):
        """If any .npy files were produced they must have the correct shape."""
        for npy in pipeline_output["processed"].rglob("*.npy"):
            arr = np.load(str(npy))
            assert arr.shape == (30, 126), f"{npy.name}: bad shape {arr.shape}"
            assert arr.dtype == np.float32
            assert not np.isnan(arr).any()

    def test_filenames_canonical_when_present(self, pipeline_output):
        """Any produced files must follow the canonical naming convention."""
        import re
        stem_re = re.compile(r"^[a-z_]+_s\d{2}_\d{4}$")
        for npy in pipeline_output["processed"].rglob("*.npy"):
            assert stem_re.match(npy.stem), f"Non-canonical filename: {npy.name}"

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def test_validation_passes(self, pipeline_output):
        """validate() must exit cleanly regardless of extraction yield."""
        assert pipeline_output["errors"] == [], (
            f"Validation errors: {pipeline_output['errors']}"
        )

    def test_report_exists(self, pipeline_output):
        assert (pipeline_output["artifacts"] / "validation_report.md").exists()

    def test_stats_exist_when_train_populated(self, pipeline_output):
        """feature_stats.json is only written when train split has data."""
        train_npy = list((pipeline_output["processed"] / "train").rglob("*.npy")) \
            if (pipeline_output["processed"] / "train").exists() else []
        stats_path = pipeline_output["artifacts"] / "feature_stats.json"
        if train_npy:
            assert stats_path.exists()
            stats = json.loads(stats_path.read_text())
            assert len(stats["mean"]) == 126
        else:
            pytest.skip("No train files produced (synthetic fixture has no detectable hands)")

    # ------------------------------------------------------------------ #
    # Dataset loader (uses real data/processed if fixture produced nothing)
    # ------------------------------------------------------------------ #

    def test_dataset_loads(self, pipeline_output):
        npy_files = list(pipeline_output["processed"].rglob("*.npy"))
        if not npy_files:
            # Fall back to real data/processed which is always populated locally
            real = Path("data/processed")
            if not (real / "train").exists() or not any((real / "train").glob("*.npy")):
                pytest.skip("No extracted data available for dataset loader test")
            processed = real
        else:
            processed = pipeline_output["processed"]

        from backend.data.dataset import build_dataset
        import tensorflow as tf

        for split in ("train", "val", "test"):
            split_dir = processed / split
            if split_dir.exists() and any(split_dir.glob("*.npy")):
                ds = build_dataset(split, batch_size=4, processed_dir=processed)
                seq, lab = next(iter(ds))
                assert seq.shape[-2:] == (30, 126)
                assert seq.dtype == tf.float32
                assert lab.dtype == tf.int32
                break

    # ------------------------------------------------------------------ #
    # Normalization + augmentation round-trip                              #
    # ------------------------------------------------------------------ #

    def test_normalize_then_augment(self, pipeline_output):
        npy_files = list(pipeline_output["processed"].rglob("*.npy"))
        if not npy_files:
            # Use a real processed file if available
            real_files = list(Path("data/processed").rglob("*.npy"))
            if not real_files:
                pytest.skip("No .npy files available for normalize/augment test")
            npy_files = real_files[:1]

        from backend.data.normalize import normalize_sequence
        from backend.data.augment import random_augment

        seq  = np.load(str(npy_files[0])).astype(np.float32)
        norm = normalize_sequence(seq)
        aug  = random_augment(norm, rng=np.random.default_rng(0))

        assert aug.shape == (30, 126)
        assert not np.isnan(aug).any()
