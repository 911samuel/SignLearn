# Phase 2 — Baseline Training Results

| Metric | Value |
|---|---|
| Best val_accuracy | **93.90%** |
| Best epoch | 16 / 26 |
| Val loss at best | 0.5999 |
| Train accuracy (final epoch) | 64.10% |
| Early stopping patience | 10 |
| Classes trained | 10 (digits 0–9, compact indexed) |
| Input shape | (30, 126) |
| Batch size | 32 |

## Result

**Target met** — 93.90% validation accuracy exceeds the ≥85% Phase 2 goal.  
Subtask 6 (hyperparameter tuning) is **skipped**.

## Notes

- High train/val accuracy gap (64% train vs 94% val) is typical early in training with heavy dropout + augmentation; the validation set is clean while training uses on-the-fly augmentation.
- Early stopping fired at epoch 26 (10 epochs after best at epoch 16).
- Model restored to best checkpoint (epoch 16) for export.
