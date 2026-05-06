# SignLearn — Research Notes
> **Purpose**: Engineering reference only. Guides implementation decisions for the real-time ASL recognition pipeline.

---

## 1. MediaPipe Hands — Hand Landmark Detection

**Paper / Source**: Google MediaPipe Hands (Zhang et al., 2020)

### Summary
- Uses a two-stage pipeline: a palm detector (BlazePalm) followed by a hand landmark model.
- Outputs **21 3D landmarks** per hand, each with `(x, y, z)` coordinates normalized to the image frame.
- `x`, `y` are normalized to `[0.0, 1.0]` relative to image width/height; `z` is depth relative to the wrist.
- Designed for **real-time inference** — runs at 21–30+ FPS on CPU; GPU mode exceeds 30 FPS comfortably.
- Supports up to **2 hands** simultaneously; single-hand mode is lower latency.

### What We Will Use
- **Input**: Raw webcam frames (BGR, 640×480 default) captured via OpenCV.
- **Output per frame**: `21 landmarks × 3 coordinates = 63 values` → flattened to a **1D vector of shape `(63,)`** per hand.
- **Multi-hand**: We target single-hand detection initially to keep feature dimensionality fixed and reduce ambiguity.
- **Model input construction**: Each frame produces one `(63,)` vector. A sliding window of `N` consecutive frames forms a sequence fed into the LSTM.
- **Normalization**: Landmarks will be re-normalized relative to the wrist (`landmark[0]`) to achieve translation invariance across signers.

---

## 2. LSTM — Long Short-Term Memory Networks

**Paper / Source**: Hochreiter & Schmidhuber (1997); Graves et al. (applied to sequence classification)

### Summary
- LSTMs maintain a **cell state** and **hidden state** across timesteps, solving the vanishing gradient problem of vanilla RNNs.
- Gating mechanism (`input`, `forget`, `output` gates) selectively retains or discards temporal information.
- Well-suited for **variable-length sequences** with long-range dependencies — exactly the nature of ASL signs.
- Performs **sequence-to-label** classification when only the final hidden state is passed to a dense output layer.
- Computationally lighter than attention-based transformers for short-to-medium sequences (< 100 frames).

### What We Will Use
- **Input shape**: `(sequence_length, feature_size)` → concretely `(30, 126)` — 30 frames, 126 MediaPipe features per frame (two hands × 21 landmarks × 3 coords).
- **Why LSTM over CNN**: CNNs process spatial features from raw pixels; our pipeline extracts landmarks via MediaPipe, making the spatial step already done. LSTM operates directly on the structured landmark sequence.
- **Architecture**: Stacked LSTM (2 layers) → Dropout → Dense → Softmax over vocabulary size (93 classes).
- **Output**: Probability distribution over all 93 vocabulary labels; `argmax` gives the predicted sign.
- **Sequence window**: 30 frames at ~15 FPS ≈ 2 seconds of signing — sufficient for most single-sign gestures.

---

## 3. Sign Language Datasets — WLASL & Alternatives

**Paper / Source**: Li et al., "Word-Level Deep Sign Language Recognition from Video" (WLASL, 2020)

### Summary
- WLASL contains **21,083 videos** across **2,000 ASL word classes**, sourced from online ASL dictionaries.
- Videos vary in: signer identity, lighting, background, camera angle, and signing speed — high real-world variance.
- Subset **WLASL100** (100 classes) is commonly used for benchmarking due to data imbalance in larger splits.
- Key challenge: **inter-signer variation** — the same word signed differently by different people is the largest source of error.
- Raw videos require pose extraction (e.g., via MediaPipe or OpenPose) before any landmark-based model can use them.

### What We Will Use
- **Initial scope**: We will NOT use WLASL in the first training iteration — our vocabulary (93 labels) is custom and does not align 1:1 with WLASL classes.
- **Primary strategy**: **Custom data collection** — record 30–60 video samples per class using our own MediaPipe extraction pipeline, ensuring consistency in output format.
- **WLASL as fallback**: For classes with insufficient custom samples, pre-extracted landmark sequences from WLASL100 may be adapted where vocabulary overlaps (e.g., `hello`, `stop`, `help`).
- **Acknowledged limitation**: Small custom dataset risks overfitting. Mitigation: data augmentation on landmark sequences (temporal jitter, horizontal flip for handedness, scaling noise).

---

## 4. Hybrid Approaches — CNN + Sequence Models

**Paper / Source**: Various (e.g., Pigou et al. 2016; Joze & Koller 2019 — CNN-LSTM pipelines for gesture recognition)

### Summary
- Hybrid pipelines use a **CNN as a spatial feature extractor** (per frame) feeding into a **sequence model** (LSTM/GRU/Transformer) for temporal classification.
- CNNs learn spatial patterns directly from raw pixels — useful when hand pose estimation is unavailable or unreliable.
- Requires significantly more compute: CNN inference per frame + sequence model = higher latency and memory cost.
- State-of-the-art results on raw video benchmarks, but overkill when structured landmarks are already available.
- Requires large labeled video datasets to train the CNN component effectively; prone to overfitting on small datasets.

### What We Will Use
- **We are NOT implementing a CNN stage.** MediaPipe replaces it entirely by outputting structured `(21, 3)` landmark data — no pixel-level spatial learning needed.
- **Design justification**:
  - MediaPipe runs at real-time speed on CPU, removing the CNN bottleneck.
  - Landmark-based input is **signer-agnostic** (appearance-invariant), reducing the data needed for generalization.
  - LSTM on 63-dimensional input trains orders of magnitude faster than a CNN-LSTM on raw frames.
  - Target deployment is a laptop/mobile webcam — real-time constraint rules out heavy CNN inference.
- **Trade-off accepted**: We lose the ability to capture subtle non-hand cues (facial expression, body posture). This is acceptable for our vocabulary scope but noted as a future extension.

---

## Architecture Decision Summary

| Component | Decision | Rationale |
|---|---|---|
| Landmark extraction | MediaPipe Hands (CPU) | Real-time, structured output, no GPU required |
| Feature vector | 21 × 3 = 63 values/frame | Fixed-size, translation-invariant after wrist normalization |
| Sequence length | 30 frames | ~2s window, captures most single-sign gestures |
| Sequence model | 2-layer LSTM | Temporal modeling on structured data, lightweight |
| Output | Softmax over 93 classes | Matches `vocabulary.md` label set directly |
| Training data | Custom collection + WLASL fallback | Ensures format consistency; avoids dataset mismatch |
| CNN stage | ❌ Omitted | Replaced by MediaPipe; not needed for landmark pipelines |
