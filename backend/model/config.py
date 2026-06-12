"""Single source of truth for Phase 2 training hyperparameters.

`num_classes` defaults to the number of vocabulary labels actually present in
``data/processed/train/`` — not the full 93-class label map — because Phase 2
trains only on the classes that have been recorded so far (currently the 10
digits). A compact 0..N-1 remapping is produced by :func:`compact_label_map`
so model outputs align with the present classes.
"""

from dataclasses import dataclass, field
from pathlib import Path

from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN
from backend.data.dataset import list_split
from backend.data.label_map import inverse_label_map

_REPO_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = _REPO_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
LOGS_DIR = ARTIFACTS_DIR / "logs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Re-exported for callers that import these directly from backend.model.config.
# The authoritative definitions live in backend.data.constants.
__all__ = [
    "FEATURE_DIM", "SEQUENCE_LEN",
    "PROCESSED_DIR", "ARTIFACTS_DIR", "CHECKPOINTS_DIR", "LOGS_DIR", "REPORTS_DIR",
    "TrainConfig", "compact_label_map", "compact_class_names", "present_label_indices",
]

for _d in (CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def present_label_indices(processed_dir: Path | None = None) -> list[int]:
    """Sorted list of full-vocab label indices present in the train split."""
    items = list_split("train", processed_dir=processed_dir)
    return sorted({idx for _, idx in items})


def compact_label_map(processed_dir: Path | None = None) -> dict[int, int]:
    """Map full-vocab index → compact 0..N-1 index for present classes only."""
    return {full: compact for compact, full in enumerate(present_label_indices(processed_dir))}


def compact_class_names(processed_dir: Path | None = None) -> list[str]:
    """Vocabulary label names in compact-index order."""
    inv = inverse_label_map()
    return [inv[full] for full in present_label_indices(processed_dir)]


def _default_num_classes() -> int:
    try:
        n = len(present_label_indices())
        return n if n > 0 else 1
    except FileNotFoundError:
        return 1


def _load_yaml(path: Path) -> dict:
    """Lazy yaml import — keeps yaml an optional dep for callers that never use it."""
    import yaml  # type: ignore[import-untyped]
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class TrainConfig:
    """Hyperparameters for the SignLearn sequence classifier.

    ``arch_name`` selects which architecture builder to use from
    :data:`backend.model.architectures.ARCHITECTURE_REGISTRY`. ``feature_mode``
    controls the engineered-feature stack in :mod:`backend.data.features`; when
    not ``"raw"`` the model input width grows accordingly via
    :func:`backend.data.features.output_dim`.
    """

    input_shape: tuple[int, int] = (SEQUENCE_LEN, FEATURE_DIM)
    num_classes: int = field(default_factory=_default_num_classes)

    arch_name: str = "lstm"
    feature_mode: str = "raw"

    lstm_units: tuple[int, int] = (128, 64)
    dense_units: int = 64
    dropout: float = 0.4
    recurrent_dropout: float = 0.2

    # Transformer-specific knobs (ignored by LSTM/BiLSTM builders).
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_d_model: int = 128
    transformer_ff_dim: int = 256

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100

    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainConfig":
        """Load a TrainConfig from a YAML file.

        Unknown keys are silently ignored so YAML files can carry extra
        sweep-only metadata (eg. ``run_name``, ``notes``) without breaking
        the dataclass constructor. Known scalar fields are coerced to the
        dataclass's annotated types where possible.
        """
        data = _load_yaml(Path(path))
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        # tuple coercions for fields the dataclass declares as tuples
        if isinstance(kwargs.get("lstm_units"), list):
            kwargs["lstm_units"] = tuple(kwargs["lstm_units"])
        if isinstance(kwargs.get("input_shape"), list):
            kwargs["input_shape"] = tuple(kwargs["input_shape"])
        return cls(**kwargs)
