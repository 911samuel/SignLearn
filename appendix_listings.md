# Appendix A — Selected Code Listings

The three listings in this appendix are the load-bearing implementations referenced throughout Chapters 4 and 5. Each is reproduced verbatim from the public repository at tag `defense-2026-06`; full files (including imports, docstrings, and test fixtures) are at the file paths shown.

The listings are presented in execution order: data conditioning → model → deployment-time gating.

---

## Listing A.1 — Landmark normalisation

**File:** `backend/data/normalize.py`
**Referenced in:** §4.2.1, §5.2, §5.5

Each raw frame from MediaPipe Hands is a 126-dimensional vector encoding two hands × 21 landmarks × 3 coordinates. Before any sequence is passed to the classifier, every frame is wrist-centred (so absolute screen position becomes irrelevant) and unit-scaled per hand (so camera-to-hand distance becomes irrelevant). Empty hand slots are preserved as zeros. This per-hand operation does substantially more recognition work than the choice of sequence backbone (see §5.5).

```python
import numpy as np
from backend.data.constants import COORDS, FEATURE_DIM, HAND_DIM, N_LANDMARKS

TWO_HAND_DIM = FEATURE_DIM   # 126 = 2 hands × 21 landmarks × 3 coords
WRIST_IDX    = 0             # landmark index of the wrist


def _split_hands(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a (126,) frame into two (21, 3) arrays (left, right)."""
    if frame.shape != (TWO_HAND_DIM,):
        raise ValueError(f"Expected ({TWO_HAND_DIM},), got {frame.shape}")
    left  = frame[:HAND_DIM].reshape(N_LANDMARKS, COORDS)
    right = frame[HAND_DIM:].reshape(N_LANDMARKS, COORDS)
    return left, right


def _merge_hands(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Merge two (21, 3) arrays back into a (126,) frame."""
    return np.concatenate([left.flatten(), right.flatten()]).astype(np.float32)


def _hand_is_empty(hand: np.ndarray) -> bool:
    """True when a hand slot is all zeros (absent / padded)."""
    return not np.any(hand)


def _wrist_center_hand(hand: np.ndarray) -> np.ndarray:
    """Translate so wrist (landmark 0) is at the origin."""
    if _hand_is_empty(hand):
        return hand.copy()
    return (hand - hand[WRIST_IDX]).astype(np.float32)


def _scale_unit_hand(hand: np.ndarray) -> np.ndarray:
    """Scale so max pairwise Euclidean distance among landmarks equals 1."""
    if _hand_is_empty(hand):
        return hand.copy()

    diff = hand[:, None, :] - hand[None, :, :]   # (21, 21, 3)
    dists = np.linalg.norm(diff, axis=-1)         # (21, 21)
    max_dist = dists.max()
    if max_dist < 1e-9:
        return hand.copy()
    return (hand / max_dist).astype(np.float32)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Wrist-centre and unit-scale each hand in a (126,) frame independently.

    Zero-padded hand slots are left as zeros so the (30, 126) tensor shape
    is preserved when only one hand is in frame.
    """
    left, right = _split_hands(frame)
    left  = _scale_unit_hand(_wrist_center_hand(left))
    right = _scale_unit_hand(_wrist_center_hand(right))
    return _merge_hands(left, right)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply normalize_frame to every frame in a (T, 126) sequence."""
    if seq.ndim != 2 or seq.shape[1] != TWO_HAND_DIM:
        raise ValueError(f"Expected (T, {TWO_HAND_DIM}), got {seq.shape}")
    return np.stack([normalize_frame(f) for f in seq], axis=0).astype(np.float32)
```

*The dataset audit at `artifacts/reports/dataset_audit.md` confirms that post-normalisation feature ranges fall within approximately [−1, 1] across all 36 classes, validating the normalisation choice empirically.*

---

## Listing A.2 — Temporal Convolutional Network (production architecture)

**File:** `backend/model/architectures/tcn.py`
**Referenced in:** §4.3, §5.3, §5.5

The production classifier is a stack of four dilated 1D-convolutional residual blocks with dilation rates 1, 2, 4, 8, followed by global average pooling and a dense head. Compared to LSTMs, dilated convolutions are fully parallel along the time axis (faster training and inference) and grow their receptive field exponentially with dilation rate. The configuration shown — 64 filters, kernel size 3, dropout 0.4 — is the winning point of the Phase-3 sweep.

```python
import tensorflow as tf
from backend.model.config import TrainConfig


def _tcn_block(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    """Single TCN residual block: two dilated convs + residual connection."""
    residual = x
    in_channels = x.shape[-1]

    y = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, padding="same",
        dilation_rate=dilation_rate, activation="relu",
        name=f"{name}_conv1",
    )(x)
    y = tf.keras.layers.LayerNormalization(name=f"{name}_ln1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(y)

    y = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, padding="same",
        dilation_rate=dilation_rate, activation="relu",
        name=f"{name}_conv2",
    )(y)
    y = tf.keras.layers.LayerNormalization(name=f"{name}_ln2")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop2")(y)

    if in_channels != filters:
        residual = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=1, padding="same",
            name=f"{name}_proj",
        )(residual)
    return tf.keras.layers.Add(name=f"{name}_add")([residual, y])


def build_tcn(config: TrainConfig) -> tf.keras.Model:
    """Build the production TCN classifier.

    Architecture: Masking -> 4 dilated TCN blocks (dilations 1, 2, 4, 8)
                  -> GlobalAveragePooling1D -> Dense -> Softmax.
    """
    filters   = getattr(config, "tcn_filters", 64)
    kernel    = getattr(config, "tcn_kernel_size", 3)
    dilations = getattr(config, "tcn_dilations", (1, 2, 4, 8))

    inputs = tf.keras.Input(shape=config.input_shape, name="landmark_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0, name="masking")(inputs)

    for d in dilations:
        x = _tcn_block(
            x, filters=filters, kernel_size=kernel, dilation_rate=d,
            dropout=config.dropout, name=f"tcn_d{d}",
        )

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu",
                              name="dense_1")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout_dense")(x)
    outputs = tf.keras.layers.Dense(
        config.num_classes, activation="softmax", name="output",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name="signlearn_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

*The trained checkpoint is `artifacts/runs/phase3-raw-balanced__arch=tcn_fm=raw_lr=0.0005_do=0.4/checkpoints/tcn_best.keras`; the ONNX export is `tcn_best.onnx` in the same directory (538 KB).*

---

## Listing A.3 — Real-time prediction smoothing

**File:** `backend/api/smoothing.py`
**Referenced in:** §4.2.2, §4.2.5, §5.6

Raw softmax outputs at 30 fps are too noisy to surface directly in the UI. The `PredictionSmoother` applies three deployment-time mechanisms: an exponential moving average over probability vectors, a fixed confidence threshold below which no label is emitted, and a repeat-cooldown that prevents the same label from being re-emitted on every prediction tick. These three constants — `ema_alpha=0.6`, `conf_threshold=0.75`, `repeat_cooldown_frames=15` — were tuned on a held-out webcam recording session and never adjusted against the test set.

```python
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import time


@dataclass
class SmoothingConfig:
    ema_alpha: float = 0.6            # weight on the new observation
    conf_threshold: float = 0.75      # min smoothed top-1 prob to emit
    repeat_cooldown_frames: int = 15  # suppress same label within N frames
    stride: int = 1                   # emit every K frames
    hysteresis_frames: int = 2        # consecutive frames needed to switch


class PredictionSmoother:
    """Rolling smoother over softmax probability vectors.

    `update(probs)` returns either `None` (suppressed) or a wire-format
    dict `{"label", "confidence", "ready"}`.
    """

    def __init__(self, class_names: Sequence[str],
                 cfg: SmoothingConfig | None = None) -> None:
        self.class_names = list(class_names)
        self.cfg = cfg or SmoothingConfig()
        self._ema: np.ndarray | None = None
        self._frames_since_emit: int = self.cfg.repeat_cooldown_frames + 1
        self._frames_seen: int = 0
        self._last_label: str | None = None
        self._candidate_label: str | None = None
        self._candidate_streak: int = 0
        self.last_used_ts: float = time.time()

    def reset(self) -> None:
        self._ema = None
        self._frames_since_emit = self.cfg.repeat_cooldown_frames + 1
        self._frames_seen = 0
        self._last_label = None
        self._candidate_label = None
        self._candidate_streak = 0

    def update(self, probs: np.ndarray) -> dict | None:
        """Ingest a softmax vector and emit a wire-format dict or None."""
        probs = np.asarray(probs, dtype=np.float32).ravel()
        if probs.shape[0] != len(self.class_names):
            raise ValueError(
                f"probs has {probs.shape[0]} dims but smoother was built "
                f"for {len(self.class_names)} classes"
            )
        self.last_used_ts = time.time()

        # 1. EMA over probability vectors.
        if self._ema is None:
            self._ema = probs.copy()
        else:
            a = self.cfg.ema_alpha
            self._ema = a * probs + (1.0 - a) * self._ema

        self._frames_seen += 1
        self._frames_since_emit += 1

        if self._frames_seen % max(1, self.cfg.stride) != 0:
            return None

        top_idx = int(np.argmax(self._ema))
        top_prob = float(self._ema[top_idx])
        label = self.class_names[top_idx]

        # 2. Confidence gate.
        if top_prob < self.cfg.conf_threshold:
            self._candidate_label = None
            self._candidate_streak = 0
            return {"label": None, "confidence": None, "ready": True}

        # 3. Hysteresis: require N consecutive over-threshold frames before
        #    switching to a *new* label. First emission is exempt.
        if label != self._last_label and self._last_label is not None:
            if label == self._candidate_label:
                self._candidate_streak += 1
            else:
                self._candidate_label = label
                self._candidate_streak = 1
            if self._candidate_streak < max(1, self.cfg.hysteresis_frames):
                return None

        self._candidate_label = None
        self._candidate_streak = 0

        # 4. Repeat-suppression: don't re-emit same label within cooldown.
        if (label == self._last_label
                and self._frames_since_emit <= self.cfg.repeat_cooldown_frames):
            return None

        self._last_label = label
        self._frames_since_emit = 0
        return {"label": label, "confidence": round(top_prob, 4),
                "ready": True}
```

*Per-connection state is held in a `SmootherRegistry` (not shown) that maps WebSocket session IDs to `PredictionSmoother` instances and evicts smoothers idle for more than 300 seconds.*

---

## Reproducing the headline result

The 97.84 % test accuracy reported in §5.3.1 can be reproduced from a clean clone of the repository in six commands. See §4.5.1 for full discussion.

```bash
make setup
python backend/scripts/download_datasets.py --dataset alphabet
python backend/scripts/download_datasets.py --dataset digits
make augment TARGET=600
make train ARCH=tcn FEATURE_MODE=raw \
           RUN_NAME=tcn-raw-replication EPOCHS=100
python backend/scripts/evaluate_model.py --run tcn-raw-replication
```

Total runtime on a single CPU core is approximately 60 minutes, dominated by the training step. The resulting `metrics.json` contains the headline accuracy, macro precision / recall / F1, per-class scores, and the top-20 confusion pairs.
