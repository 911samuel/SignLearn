# SignLearn — Data Collection Recommendation

> Updated after the 36-class BiLSTM+raw analysis (Phase 3, `feature/model-training`).
> Current best: BiLSTM+raw 83.5% test accuracy on 36 classes (26 letters + 10 digits).

## Executive summary

The 36-class Kaggle-sourced dataset has **4 fundamental confusion clusters** that cannot be fixed by adding more synthetic data — they require real webcam recordings with depth variation. Record these classes first before expanding to the full 93-class vocabulary.

| Priority | Classes | Root cause | Fix |
|---|---|---|---|
| P0 (critical) | `k`, `v` | Kaggle synthetic: `|index_z - middle_z| = 0.0000` for both | Record with depth variation at multiple angles |
| P0 (critical) | `r`, `u` | Same 2D projection issue | Record with depth variation |
| P1 (wontfix) | `two` ↔ `v`, `six` ↔ `w` | ASL semantic equivalence (same handshape) | Accept this ambiguity; context/phrase models needed for disambiguation |
| P2 (dataset) | `e` ↔ `s` | Kaggle synthetic lacks variation | Record with multiple signers |

**Current confusion matrix highlights** (BiLSTM+raw, bilstm-v2-36cls):

| True label | Predicted | Error rate |
|---|---|---:|
| `k` | `v` | 97% (435/450 samples misclassified) |
| `r` | `u` | 96% (459/480 samples misclassified) |
| `two` | `v` | 100% (42/42 — same handshape) |
| `six` | `w` | 100% (38/38 — same handshape) |
| `e` | `s` | 53% (218/413 misclassified) |

## Immediate recording priorities (Phase 3)

### Priority 0 — Fix Kaggle 2D projection errors

These cannot be fixed with more synthetic data. Only real webcam video at different angles will expose the z-coordinate difference that distinguishes these signs.

**`k` vs `v`** — The index and middle fingers extend together in `v` (apart) and cross in `k` (index over middle). In the Kaggle dataset both have `|index_z - middle_z| = 0`, making them indistinguishable. In real 3D recording the crossed finger in `k` is visibly closer to the camera.

**`r` vs `u`** — `u` has two extended fingers held together; `r` crosses them. Same projection issue.

Recording target:
- ≥ 5 signers × ≥ 100 sequences each = 500+ samples per class
- Camera angles: front (0°), 30° left, 30° right — variation exposes z-depth
- Both dominant and non-dominant hand (handedness matters)

### Priority 1 — Augment `e` and `s`

`e` has a curled position that shares silhouette with `s` (closed fist). More signer diversity helps:
- ≥ 3 signers, ≥ 80 samples each, diverse hand sizes

### Priority 2 — Accept two/six ↔ v/w ambiguity

ASL digit `2` and letter `v` use the same handshape. Digit `6` and letter `w` use the same handshape. This is a vocabulary design issue — no model can disambiguate without linguistic context. Options:
1. **Remove one of each pair from the 36-class classifier** (e.g., train on `a-z` + `zero,one,three,four,five,seven,eight,nine` — omit `two` and `six`).
2. **Add a context model** (n-gram LM or CTC decoder over the output stream) to infer digits vs letters from surrounding context.
3. **Accept as known limitation** for the current demo.

## Full 93-class vocabulary recording plan

### Target budget

| Class group | Count | Min samples / class | Total sequences |
|---|---:|---:|---:|
| Fix k, v, r, u (re-record) | 4 | 500 | 2,000 |
| Augment e, s | 2 | +200 each | 400 |
| Dynamic words (33) | 33 | 500 | 16,500 |
| Static words (24) | 24 | 300 | 7,200 |
| **Grand total (new)** | **63** | — | **~26,100** |

Dynamic words need 500 samples because temporal motion benefits from speed diversity.

### Recording protocol

```bash
# Record with full diversity matrix (prompts for angle + lighting)
python backend/scripts/record_vocabulary.py --words k v r u --diversity-matrix

# Audit after each session
python backend/scripts/audit_dataset.py
```

1. **Diversity matrix per class** (minimum):
   - Signers: ≥ 3 individuals with different hand sizes
   - Camera angles: front (0°), 30L°, 30R° — **critical for k/r disambiguation**
   - Lighting: bright indoor, dim indoor
   - Wardrobe: short sleeve, long sleeve

2. **Naming convention**: `<label>_s<NN>_<idx>.npy` (e.g., `k_s02_0042.npy`). Use `--signer-id` flag in `record_vocabulary.py`. Keep one signer per split to prevent leakage.

3. **Sanity gate** after recording:
   ```bash
   python backend/scripts/audit_dataset.py
   # Must exit 0; check audit_signers.md shows zero cross-split signer overlap
   ```

## Feature mode guidance

For the current Kaggle-sourced static image dataset:
- **Use `raw` features** — velocity/acceleration are near-zero for static images (all 30 frames identical), so `engineered` features add noise not signal
- After recording real dynamic words: switch to `raw+velocity` to encode motion

## Decision tree for model improvement

```
Current 83.5% test acc
  ├─ Fix k/v/r/u by recording (expected gain: +3-5%)
  ├─ Fix e/s by adding signer diversity (expected gain: +1%)
  ├─ Phase 3 sweep selects best arch (expected gain: +2-5% over BiLSTM)
  └─ Full 93-class training (requires 26K new sequences)
```

Target: 92%+ F1 after fixing the P0 confusions + architecture sweep.
