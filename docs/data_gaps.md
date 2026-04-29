# Data Gaps Analysis

## 1. Static vs. Dynamic Mismatch
The current raw datasets (Alphabet and Digits) consist entirely of **static images**.
- **The Gap**: Our `vocabulary.md` includes many "Dynamic Words" (e.g., `eat`, `drink`, `go`, `stop`) that require motion and temporal context.
- **Risk**: A model trained only on these images will fail to recognize any movement-based signs and will likely misclassify a moving hand as a static letter.

## 2. Lack of Sequences
- **The Gap**: We lack video/sequence data for the signs. MediaPipe landmarks need to be extracted across a time-series (e.g., 30 frames) to train the LSTM.
- **Problem**: There are no readily available, high-quality open datasets that match our exact 75-label vocabulary in a structured sequence format ready for landmark extraction.

## 3. Background and Lighting Bias
- **The Gap**: The Kaggle ASL Alphabet dataset is notorious for having very consistent backgrounds per class.
- **Risk**: The model might learn to recognize the "wall" or "lighting" behind a specific hand shape rather than the hand itself.

## 4. Handedness Variation
- **The Gap**: Most datasets favor right-handed signers.
- **Risk**: Poor performance for left-handed users unless we implement horizontal flip augmentation during preprocessing.

---

## Fallback & Strategy

### Primary Fallback: Self-Recorded Dataset
Given the gaps in existing public data for dynamic ASL signs, we will implement a **Custom Data Collection Pipeline**:
1. **Tool**: Create a Python script (`collect_data.py`) using OpenCV and MediaPipe.
2. **Method**: Record 30-frame sequences of landmarks directly to `.npy` or `.csv` files.
3. **Volume**: Aim for 50 samples per class for all 75 labels.
4. **Benefit**: This ensures the data perfectly matches our model input shape `(30, 63)` and covers both static and dynamic words consistently.

### Secondary Strategy: Augmentation
- Artificially create "dynamic" data by applying small translations and rotations to static landmark sets.
- Use horizontal flipping to double the dataset for left/right-hand compatibility.
