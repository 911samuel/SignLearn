"""Hyperparameter sweep harness for SignLearn.

Loads a YAML sweep config, expands the grid, and invokes ``train_model.py``
once per configuration as a clean subprocess. Each run gets its own
``artifacts/runs/<run_name>/`` directory; the harness then calls
``evaluate_model.py --runs ...`` to emit a single comparison table.

YAML schema
-----------
```yaml
sweep_id: phase3-smoke
epochs: 25                    # default applied to all runs unless overridden
batch_size: 32
data_dir: data/processed       # optional

grid:
  arch:         [bilstm, tcn, cnn_bilstm, conformer_lite]
  feature_mode: [raw+velocity, engineered]
  learning_rate: [1.0e-3, 5.0e-4]
  dropout:       [0.3, 0.4]
```

The cartesian product of `grid` fields drives the experiments; each field
maps directly onto a `TrainConfig` attribute. Run names are auto-generated
as `<sweep_id>__arch=<arch>_fm=<feature_mode>_lr=<lr>_do=<dropout>`.

Usage
-----
python backend/scripts/sweep.py --config configs/sweeps/phase3.yaml
python backend/scripts/sweep.py --config configs/sweeps/phase3.yaml --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml  # type: ignore[import-untyped]

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.model.config import ARTIFACTS_DIR, REPORTS_DIR  # noqa: E402


# Fields that can appear in `grid:` and map directly to TrainConfig attrs.
# Order matters only for stable run-name generation.
_GRID_FIELDS = (
    "arch",            # → arch_name
    "feature_mode",
    "learning_rate",
    "dropout",
    "batch_size",
    "epochs",
    "aug_profile",     # train_word_model.py
    "seq_len",         # train_word_model.py
)


def _expand_grid(grid: dict[str, list]) -> list[dict]:
    """Return the list of (named) parameter dicts in the cartesian product."""
    keys = [k for k in _GRID_FIELDS if k in grid]
    extra = [k for k in grid if k not in _GRID_FIELDS]
    if extra:
        raise ValueError(
            f"Unknown grid fields: {extra}. Allowed: {list(_GRID_FIELDS)}"
        )
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _make_run_name(sweep_id: str, params: dict) -> str:
    """Stable filesystem-safe run name from a parameter dict."""
    bits = []
    for k in _GRID_FIELDS:
        if k in params:
            v = params[k]
            short = {
                "arch": "arch",
                "feature_mode": "fm",
                "learning_rate": "lr",
                "dropout": "do",
                "batch_size": "bs",
                "epochs": "ep",
                "aug_profile": "aug",
                "seq_len": "T",
            }[k]
            # compact value rendering
            if isinstance(v, float):
                v_str = f"{v:g}"
            else:
                v_str = str(v).replace("+", ".")
            bits.append(f"{short}={v_str}")
    return f"{sweep_id}__" + "_".join(bits)


# Short-name mapping needs to know aug_profile/seq_len too.
_SHORT_NAME_PATCH = {"aug_profile": "aug", "seq_len": "T"}


def _build_train_args(
    params: dict,
    defaults: dict,
    run_name: str,
    data_dir: Path | None,
) -> list[str]:
    """Compose the CLI invocation for one training run."""
    trainer_rel = defaults.get("trainer") or "backend/scripts/train_model.py"
    cmd: list[str] = [
        sys.executable,
        str(_REPO_ROOT / trainer_rel),
        "--run-name", run_name,
    ]
    if "arch" in params:
        cmd += ["--arch", str(params["arch"])]
    if "feature_mode" in params:
        cmd += ["--feature-mode", str(params["feature_mode"])]
    if "learning_rate" in params:
        cmd += ["--lr", str(params["learning_rate"])]
    if "dropout" in params:
        cmd += ["--dropout", str(params["dropout"])]
    epochs = params.get("epochs", defaults.get("epochs"))
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    batch = params.get("batch_size", defaults.get("batch_size"))
    if batch is not None:
        cmd += ["--batch-size", str(batch)]
    if "aug_profile" in params:
        cmd += ["--aug-profile", str(params["aug_profile"])]
    if "seq_len" in params:
        cmd += ["--seq-len", str(params["seq_len"])]
    seq_len_default = defaults.get("seq_len")
    if "seq_len" not in params and seq_len_default is not None:
        cmd += ["--seq-len", str(seq_len_default)]
    if data_dir is not None and trainer_rel.endswith("train_model.py"):
        # Only the letter trainer accepts --data-dir; the word trainer hardcodes
        # data/processed/words.
        cmd += ["--data-dir", str(data_dir)]
    words_file = defaults.get("words_file")
    if words_file and trainer_rel.endswith("train_word_model.py"):
        cmd += ["--words-file", str(words_file)]
    return cmd


def _evaluate(run_names: list[str], sweep_id: str) -> Path:
    """Call evaluate_model.py to produce a comparison table for this sweep."""
    out = REPORTS_DIR / "sweeps" / f"{sweep_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "backend" / "scripts" / "evaluate_model.py"),
        "--runs", *run_names,
    ]
    print(f"\n=== Evaluating: {' '.join(cmd[-len(run_names) - 1:])} ===\n")
    subprocess.run(cmd, check=False, cwd=_REPO_ROOT, env={**__import__("os").environ, "PYTHONPATH": str(_REPO_ROOT)})
    # evaluate_model.py writes artifacts/reports/model_comparison.md;
    # copy it into the sweep-scoped location.
    src = REPORTS_DIR / "model_comparison.md"
    if src.exists():
        out.write_text(src.read_text())
        print(f"Sweep comparison → {out}")
    return out


def run_sweep(config_path: Path, dry_run: bool = False) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    sweep_id = cfg.get("sweep_id") or Path(config_path).stem
    grid = cfg.get("grid") or {}
    if not grid:
        raise ValueError(f"{config_path}: 'grid' section is required and must be non-empty.")

    defaults = {
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "trainer": cfg.get("trainer"),
        "seq_len": cfg.get("seq_len"),
        "words_file": cfg.get("words_file"),
    }
    data_dir = Path(cfg["data_dir"]) if cfg.get("data_dir") else None

    experiments = _expand_grid(grid)
    print(f"\n=== Sweep '{sweep_id}': {len(experiments)} runs ===\n")

    run_names: list[str] = []
    summary: list[dict] = []
    t_sweep = time.time()

    for i, params in enumerate(experiments, 1):
        run_name = _make_run_name(sweep_id, params)
        run_names.append(run_name)
        cmd = _build_train_args(params, defaults, run_name, data_dir)
        print(f"--- [{i}/{len(experiments)}] {run_name} ---")
        print("    " + " ".join(cmd))
        if dry_run:
            summary.append({"run_name": run_name, "params": params, "skipped": True})
            continue
        # Skip runs that already completed (history.json present = training finished).
        _history = _REPO_ROOT / "artifacts" / "runs" / run_name / "reports" / "history.json"
        if _history.exists():
            print(f"    ↩ Already complete — skipping.")
            summary.append({"run_name": run_name, "params": params, "skipped": True})
            run_names.append(run_name)  # still include in final evaluation
            continue
        t0 = time.time()
        result = subprocess.run(
            cmd, cwd=_REPO_ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(_REPO_ROOT)},
            check=False,
        )
        summary.append({
            "run_name": run_name,
            "params": params,
            "returncode": result.returncode,
            "elapsed_s": round(time.time() - t0, 1),
        })
        if result.returncode != 0:
            print(f"⚠️  Run {run_name} exited {result.returncode}; continuing sweep.")

    if not dry_run:
        _evaluate(run_names, sweep_id)

    summary_path = REPORTS_DIR / "sweeps" / f"{sweep_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "sweep_id": sweep_id,
        "config": str(config_path),
        "elapsed_s": round(time.time() - t_sweep, 1),
        "experiments": summary,
    }, indent=2))
    print(f"\nSweep summary → {summary_path}")
    return {"sweep_id": sweep_id, "experiments": summary}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="SignLearn hyperparameter sweep")
    p.add_argument("--config",  type=Path, required=True, help="Path to sweep YAML")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_sweep(args.config, dry_run=args.dry_run)
