# Phase 2 — Hyperparameter Search Results

**Configurations tested:** 2  
**Best val_accuracy:** 0.6667  
**Best config:** `{'lstm_units': [128, 64], 'dropout': 0.4, 'learning_rate': 0.001}`

## Results (sorted by val_accuracy)

| Run | lstm_units | dropout | lr | val_acc | epoch | params | time (s) |
|-----|-----------|---------|-----|---------|-------|--------|---------|
| 2 | [128, 64] | 0.4 | 1e-03 | 0.6667 | 1 | 184,323 | 3.0 |
| 1 | [64, 32] | 0.3 | 1e-04 | 0.3333 | 1 | 63,619 | 2.9 |
