"""
Download raw datasets from Kaggle to data/raw/.

Requirements:
  pip install kaggle
  Place ~/.kaggle/kaggle.json (from kaggle.com → Account → API → Create Token)

Usage:
  python scripts/download_datasets.py               # downloads all
  python scripts/download_datasets.py --dataset alphabet
  python scripts/download_datasets.py --dataset digits

Datasets:
  alphabet  → grassknoted/asl-alphabet   (A-Z, ~87K images, 200x200)
  digits    → ardamavi/sign-language-digits-dataset  (0-9, ~2K images, 100x100)
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"

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


def _kaggle_available() -> bool:
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        return False


def download(name: str, info: dict) -> None:
    dest: Path = info["dest"]
    if dest.exists() and any(dest.iterdir()):
        print(f"[{name}] Already present at {dest}, skipping.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Downloading {info['slug']} → {dest} ...")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", info["slug"], "-p", str(dest), "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        print(f"[{name}] Download failed. Check your Kaggle credentials and dataset slug.")
        return

    # Remove leftover zip if unzip didn't clean up
    for z in dest.glob("*.zip"):
        z.unlink()

    count = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
    print(f"[{name}] Done — {count} images in {dest}")


def main() -> None:
    if not _kaggle_available():
        print("kaggle package not found. Run: pip install kaggle")
        sys.exit(1)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"Kaggle credentials missing: {kaggle_json}")
        print("Get your token from kaggle.com → Account → API → Create New Token")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download SignLearn datasets from Kaggle.")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Download one dataset only.")
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    for name, info in targets.items():
        download(name, info)


if __name__ == "__main__":
    main()
