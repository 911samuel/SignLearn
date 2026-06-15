"""Same-class mixup augmentation.

For tiny per-class sample counts, canonical (cross-class) mixup smears
label boundaries we can't afford. Same-class mixup instead interpolates
between two sequences from the *same* gloss — a smooth augmentation, not
a soft-label regularizer. Labels are unchanged.

Used at batch level inside the tf.data pipeline (see train_word_model.py).
"""

from __future__ import annotations

import numpy as np


def same_class_mixup(
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    alpha: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each sample with another same-class sample.

    Args:
        batch_x: (B, T, D) float32
        batch_y: (B,) int64
        alpha:   beta-distribution param. lambda ~ Beta(alpha, alpha), clipped
                 to [0.5, 1.0] so the original sample dominates and the
                 augmentation reads as "perturbed self" rather than "blend".
        rng:     numpy Generator

    Returns:
        (mixed_x, batch_y) — labels unchanged. If a sample has no other
        same-class partner in the batch, it passes through untouched.
    """
    rng = rng or np.random.default_rng()
    b = batch_x.shape[0]
    out = batch_x.copy()

    # Group by class within this batch.
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(batch_y):
        by_class.setdefault(int(y), []).append(i)

    for cls, idxs in by_class.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            partners = [j for j in idxs if j != i]
            j = int(rng.choice(partners))
            lam = float(rng.beta(alpha, alpha))
            lam = max(lam, 1.0 - lam)  # keep self-weight ≥ 0.5
            out[i] = lam * batch_x[i] + (1.0 - lam) * batch_x[j]

    return out.astype(np.float32), batch_y
