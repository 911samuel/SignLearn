"""Subtask 1: build and load the canonical label map from docs/vocabulary.md."""

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_VOCAB_PATH = _REPO_ROOT / "docs" / "vocabulary.md"
_ARTIFACTS = _REPO_ROOT / "artifacts"
_LABEL_MAP_PATH = _ARTIFACTS / "label_map.json"


def _parse_vocabulary(path: Path) -> list[str]:
    """Return labels in stable category order: alphabet → digits → words."""
    sections = {"Alphabet": [], "Digits": [], "Static Words": [], "Dynamic Words": []}
    current = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("---"):
                break
            for section in sections:
                if re.match(rf"^##\s+{section}", line):
                    current = section
                    break
            else:
                if current and re.match(r"^- (\S+)", line):
                    label = re.match(r"^- (\S+)", line).group(1)
                    sections[current].append(label)
    return (
        sections["Alphabet"]
        + sections["Digits"]
        + sections["Static Words"]
        + sections["Dynamic Words"]
    )


def build_label_map() -> dict[str, int]:
    """Parse vocabulary.md, write artifacts/label_map.json, and return the map."""
    labels = _parse_vocabulary(_VOCAB_PATH)
    label_map = {label: idx for idx, label in enumerate(labels)}
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(_LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=False)
    print(f"Wrote {len(label_map)} classes to {_LABEL_MAP_PATH}")
    return label_map


def load_label_map() -> dict[str, int]:
    """Load the committed label_map.json. Raise if not found (run build first)."""
    if not _LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"{_LABEL_MAP_PATH} not found — run: python -m backend.data.label_map"
        )
    with open(_LABEL_MAP_PATH) as f:
        return json.load(f)


def inverse_label_map() -> dict[int, str]:
    return {v: k for k, v in load_label_map().items()}


# Aliases for raw class-directory names that differ from vocabulary labels.
# Digit directories are stored as "0"-"9" but the vocabulary uses "zero"-"nine".
_RAW_ALIASES: dict[str, str] = {
    "0": "zero", "1": "one",   "2": "two",   "3": "three", "4": "four",
    "5": "five", "6": "six",   "7": "seven", "8": "eight", "9": "nine",
}


def resolve_label(raw: str) -> str:
    """Map a raw class-dir name to the canonical vocabulary label.

    For most classes the raw name equals the vocabulary label (e.g. 'a', 'hello').
    For digit directories ('0'..'9') this returns the word form ('zero'..'nine').
    """
    return _RAW_ALIASES.get(raw, raw)


if __name__ == "__main__":
    m = build_label_map()
    print(f"First 5 : {list(m.items())[:5]}")
    print(f"Last  5 : {list(m.items())[-5:]}")
