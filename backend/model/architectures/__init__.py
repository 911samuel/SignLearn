"""Architecture registry for SignLearn sequence classifiers.

Each builder takes a :class:`backend.model.config.TrainConfig` and returns a
compiled ``tf.keras.Model``. The registry lets training/evaluation scripts
select a model by short name (``"lstm"``, ``"bilstm"``, ``"transformer"``)
without hard-coding imports.
"""

from __future__ import annotations

from typing import Callable

from backend.model.architectures.bilstm import build_bilstm
from backend.model.architectures.cnn_bilstm import build_cnn_bilstm
from backend.model.architectures.conformer_lite import build_conformer_lite
from backend.model.architectures.lstm import build_lstm
from backend.model.architectures.tcn import build_tcn
from backend.model.architectures.transformer import build_transformer

ARCHITECTURE_REGISTRY: dict[str, Callable] = {
    "lstm":            build_lstm,
    "bilstm":          build_bilstm,
    "transformer":     build_transformer,
    "tcn":             build_tcn,
    "cnn_bilstm":      build_cnn_bilstm,
    "conformer_lite":  build_conformer_lite,
}


def build(arch_name: str, config):
    """Build a model from its registered short name."""
    if arch_name not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture {arch_name!r}; "
            f"available: {sorted(ARCHITECTURE_REGISTRY)}"
        )
    return ARCHITECTURE_REGISTRY[arch_name](config)


__all__ = [
    "ARCHITECTURE_REGISTRY", "build",
    "build_lstm", "build_bilstm", "build_transformer",
    "build_tcn", "build_cnn_bilstm", "build_conformer_lite",
]
