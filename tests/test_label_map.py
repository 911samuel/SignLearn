"""Tests for Subtask 1: label_map."""
import json
import tempfile
from pathlib import Path

import pytest

from backend.data.label_map import (
    _LABEL_MAP_PATH,
    _parse_vocabulary,
    _VOCAB_PATH,
    build_label_map,
    inverse_label_map,
    load_label_map,
)

EXPECTED_FIRST = ["a", "b", "c"]
EXPECTED_DIGITS_START = ["zero", "one", "two"]

# vocabulary.md has 26 + 10 + 24 static + 33 dynamic = 93 actual entries
EXPECTED_TOTAL = 93


def test_parse_vocabulary_count():
    labels = _parse_vocabulary(_VOCAB_PATH)
    assert len(labels) == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL} labels, got {len(labels)}: {labels}"
    )


def test_parse_vocabulary_ordering():
    labels = _parse_vocabulary(_VOCAB_PATH)
    # Alphabet comes first
    assert labels[:3] == EXPECTED_FIRST
    # Digits follow alphabet
    digit_start = labels.index("zero")
    assert digit_start == 26
    assert labels[digit_start : digit_start + 3] == EXPECTED_DIGITS_START


def test_no_duplicates():
    labels = _parse_vocabulary(_VOCAB_PATH)
    assert len(labels) == len(set(labels)), "Duplicate labels found"


def test_all_snake_case():
    labels = _parse_vocabulary(_VOCAB_PATH)
    bad = [l for l in labels if l != l.lower() or " " in l]
    assert not bad, f"Non-snake_case labels: {bad}"


def test_build_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.data.label_map._ARTIFACTS", tmp_path)
    monkeypatch.setattr(
        "backend.data.label_map._LABEL_MAP_PATH", tmp_path / "label_map.json"
    )
    result = build_label_map()
    assert (tmp_path / "label_map.json").exists()
    with open(tmp_path / "label_map.json") as f:
        on_disk = json.load(f)
    assert result == on_disk


def test_round_trip():
    m = load_label_map()
    inv = inverse_label_map()
    assert len(m) == EXPECTED_TOTAL
    for label, idx in m.items():
        assert inv[idx] == label


def test_values_are_sequential():
    m = load_label_map()
    indices = list(m.values())
    assert indices == list(range(len(m))), "Indices must be 0..N-1 with no gaps"


def test_load_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.data.label_map._LABEL_MAP_PATH", tmp_path / "missing.json"
    )
    with pytest.raises(FileNotFoundError):
        load_label_map()
