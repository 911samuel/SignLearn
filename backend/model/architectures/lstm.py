"""Stacked LSTM architecture (Phase 2 baseline).

Thin re-export of :func:`backend.model.architecture.build_lstm` so the
registry can keep a uniform per-architecture module layout.
"""

from backend.model.architecture import build_lstm

__all__ = ["build_lstm"]
