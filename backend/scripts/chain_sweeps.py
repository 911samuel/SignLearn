#!/usr/bin/env python
"""Chain two sweeps: wait for the first to complete, then launch the second.

Usage:
    python backend/scripts/chain_sweeps.py \\
        --wait-for phase3-raw-smoke \\
        --then configs/sweeps/phase3_raw_balanced.yaml \\
        --poll-secs 60
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from backend.model.config import ARTIFACTS_DIR


def _is_sweep_done(sweep_id: str) -> bool:
    summary = ARTIFACTS_DIR / "reports" / "sweeps" / f"{sweep_id}_summary.json"
    return summary.exists()


def _wait_for(sweep_id: str, poll_secs: int) -> None:
    print(f"[chain] Waiting for sweep '{sweep_id}' to complete …")
    while not _is_sweep_done(sweep_id):
        time.sleep(poll_secs)
        print(f"[chain] Still waiting for '{sweep_id}' …", flush=True)
    print(f"[chain] '{sweep_id}' complete ✅")


def _launch(config_path: Path) -> None:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "backend" / "scripts" / "sweep.py"),
        "--config", str(config_path),
    ]
    print(f"[chain] Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=_REPO_ROOT,
                            env={**__import__("os").environ, "PYTHONPATH": str(_REPO_ROOT)})
    if result.returncode != 0:
        print(f"[chain] Sweep exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print("[chain] Sweep complete ✅")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Chain two sweeps sequentially")
    p.add_argument("--wait-for", required=True, help="sweep_id to wait for")
    p.add_argument("--then",     required=True, type=Path, help="YAML config for the next sweep")
    p.add_argument("--poll-secs", type=int, default=60, help="Polling interval in seconds")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    _wait_for(args.wait_for, args.poll_secs)
    _launch(args.then)
