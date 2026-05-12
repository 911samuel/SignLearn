"""
Download raw datasets from Kaggle to data/raw/.

Requirements:
  pip install kaggle

Authentication (kaggle CLI ≥ 2.x uses OAuth):
  Option A — API token (recommended):
    1. Go to kaggle.com → Settings → Account → API → "Create New API Token"
    2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
    3. chmod 600 ~/.kaggle/kaggle.json

  Option B — OAuth (browser-based):
    Run:  kaggle auth login
    This opens a browser; no file needed.

Usage:
  python backend/scripts/download_datasets.py               # downloads all
  python backend/scripts/download_datasets.py --dataset alphabet
  python backend/scripts/download_datasets.py --dataset digits

Datasets:
  alphabet  → grassknoted/asl-alphabet   (A-Z, ~87K images, 200x200)
  digits    → ardamavi/sign-language-digits-dataset  (0-9, ~2K images, 100x100)
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).parent.parent.parent / "data" / "raw"

DATASETS = {
    "alphabet": {
        "slug": "grassknoted/asl-alphabet",
        "dest": RAW / "alphabet",
        "zip": "asl_alphabet_train.zip",
    },
    "digits": {
        "slug": "ardamavi/sign-language-digits-dataset",
        "dest": RAW / "digits",
        "zip": "Sign Language Digits Dataset.zip",
    },
}


def _check_kaggle_auth() -> bool:
    """Return True if kaggle CLI can authenticate (json token or OAuth session)."""
    import shutil
    # Try standalone kaggle CLI first (OAuth credentials.json), then python -m kaggle
    cmd_options = []
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        cmd_options.append([kaggle_bin, "datasets", "list", "--max-size", "1"])
    cmd_options.append([sys.executable, "-m", "kaggle", "datasets", "list", "--max-size", "1"])
    for cmd in cmd_options:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
    return False


def download(name: str, info: dict) -> None:
    dest: Path = info["dest"]
    if dest.exists() and any(dest.iterdir()):
        print(f"[{name}] Already present at {dest}, skipping.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Downloading {info['slug']} → {dest} ...")
    import shutil as _shutil
    kaggle_bin = _shutil.which("kaggle") or sys.executable
    cmd = ([kaggle_bin, "datasets", "download"] if _shutil.which("kaggle")
           else [sys.executable, "-m", "kaggle", "datasets", "download"])
    subprocess.run(
        cmd + ["-d", info["slug"], "-p", str(dest), "--unzip"],
        check=True,
    )

    # Remove leftover zip if unzip didn't clean up
    for z in dest.glob("*.zip"):
        z.unlink()

    count = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
    print(f"[{name}] Done — {count} images in {dest}")


def main() -> None:
    if not _check_kaggle_auth():
        print(
            "\nKaggle authentication required. Choose one option:\n\n"
            "  Option A (API token — recommended):\n"
            "    1. Go to kaggle.com → Settings → Account → API → 'Create New API Token'\n"
            "    2. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json\n"
            "    3. chmod 600 ~/.kaggle/kaggle.json\n\n"
            "  Option B (OAuth browser login):\n"
            "    kaggle auth login\n"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download SignLearn datasets from Kaggle.")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Download one dataset only.")
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    for name, info in targets.items():
        download(name, info)


if __name__ == "__main__":
    main()
