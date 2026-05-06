
# 🚀 Phase 1 — Data Pipeline & Preprocessing (Refined Execution Plan)

## 🎯 Core Objective (clarified)

Build a **deterministic, reproducible, and model-ready pipeline** that transforms raw videos into **fixed-length, normalized landmark tensors** with zero leakage and high signal quality.

---

# 1) Data Cleaning & Dataset Design (Do this properly or everything fails later)

### What most people get wrong:

* Mixing subjects across splits → model memorizes people, not signs
* Inconsistent labels → silent training corruption
* Ignoring class imbalance → biased model

### Your improved approach:

#### a) Standardization Rules

* Naming:

```
<sign_label>_<subject_id>_<sample_id>.mp4
```

Example:

```
hello_s01_001.mp4
```

#### b) Label Encoding

* Create a **single source of truth**

```python
label_map = {
    "hello": 0,
    "thanks": 1,
    ...
}
```

Save as:

```
label_map.json
```

---

#### c) Split Strategy (IMPORTANT)

Do **NOT** randomly split per file.

👉 Instead:

* Split by **subject_id** (person-independent model)

```
Train: subjects 1–7
Val: subjects 8–9
Test: subjects 10–11
```

This avoids overfitting to specific hands/faces.

---

#### d) Dataset Validation Script

Build:

```bash
validate_dataset.py
```

Checks:

* Missing labels
* Duplicate samples
* Corrupt videos
* Class imbalance report

---

# 2) Landmark Extraction Pipeline (High-impact component)

## 🔧 Architecture

Build this as a **batch processor**, not a simple script.

```
raw_videos/
   └── hello/
         └── *.mp4

processed/
   ├── train/
   ├── val/
   └── test/
```

---

## 🎯 Extraction Strategy

### Use:

* MediaPipe Hands (fast) OR Holistic (richer but heavier)

### Output format:

Each sample:

```
(T, 126) → both hands
(T, 63)  → single hand
```

---

## 🔥 Critical Enhancements (this is where quality comes from)

### a) Frame Sampling

Do NOT process every frame blindly.

Instead:

```python
frames = uniform_sample(video, target_frames=30)
```

---

### b) Missing Landmark Handling

MediaPipe WILL fail sometimes.

Handle explicitly:

```python
if no_hand_detected:
    use_previous_frame OR zeros
```

---

### c) Normalization (very important)

Raw coordinates are useless across different camera positions.

Normalize per frame:

```python
# Center around wrist
landmarks -= wrist_position

# Scale
landmarks /= max(distance_between_points)
```

---

### d) Temporal Normalization

Standardize to fixed length:

```python
def normalize_sequence(seq, target_len=30):
    return interpolate(seq, target_len)
```

Avoid simple padding unless necessary—interpolation is better.

---

## 🧠 Output Format

Save as:

```
.npy → float32
shape: (30, 126)
```

Also store metadata:

```
sample.json
```

---

# 3) Data Augmentation (Make or break for generalization)

## ❌ What to avoid:

* Augmenting raw images (too heavy)
* Random noise without structure

## ✅ What to do instead: Landmark-level augmentation

### a) Spatial Augmentations

```python
rotate(points, angle_range=±10°)
scale(points, 0.9–1.1)
translate(points, small shift)
```

---

### b) Temporal Augmentations

```python
drop_frames(sequence, p=0.1)
speed_up / slow_down
```

---

### c) Noise Injection

```python
points += gaussian_noise(σ=0.01)
```

---

### Implementation Strategy

Do NOT save augmented data.

👉 Apply **on-the-fly** inside generator.

---

# 4) Pipeline Validation (Don’t skip this)

## Visual Debugging Tool

Create:

```bash
visualize_landmarks.py
```

* Draw skeleton using OpenCV
* Overlay on frame OR blank canvas

Check:

* Alignment consistency
* No flipped coordinates
* Smooth motion

---

## Statistical Validation

Run checks:

* Mean/std of features
* Sequence length consistency
* Class distribution

---

# 5) Data Loader (Training-ready pipeline)

## Best option: `tf.data` (not Keras Sequence)

Why:

* Faster
* Scales better
* GPU pipeline support

---

### Example Structure:

```python
def load_npy(path):
    data = np.load(path)
    return data.astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.map(load_fn)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

---

### Add augmentation inside pipeline:

```python
dataset = dataset.map(augment_fn)
```

---

# 6) Performance Optimization (You’ll need this soon)

* Use multiprocessing for extraction
* Cache `.npy` files
* Avoid reprocessing videos
* Use progress tracking (tqdm)

---

# 📦 Final Deliverables (Production-level)

You should end Phase 1 with:

### ✅ Data

* `train/`, `val/`, `test/` directories
* `.npy` sequences (normalized, fixed length)
* `label_map.json`

---

### ✅ Code Modules

* `extract_landmarks.py`
* `normalize.py`
* `augment.py`
* `dataset.py` (tf.data pipeline)
* `visualize_landmarks.py`
* `validate_dataset.py`

---

### ✅ Quality Guarantees

* No subject leakage
* Fixed sequence length
* Normalized coordinates
* Balanced dataset (or weighted)

---

# ⚠️ Straight truth (important)

If you:

* Skip normalization → model won’t generalize
* Mix subjects across splits → fake high accuracy
* Ignore temporal consistency → LSTM underperforms

This phase determines **~70% of your final model performance**.