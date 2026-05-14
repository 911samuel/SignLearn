"""Temporal Convolution Network (TCN) for SignLearn.

Stacked dilated 1D convolutions with residual connections. Compared to LSTMs:
- Fully parallel along the time axis → faster training and inference.
- Receptive field grows exponentially with dilation rate, capturing long-range
  motion without the vanishing-gradient pain of deep RNNs.

Reference: Bai et al., "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling" (2018).
"""

from __future__ import annotations

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
    """Single TCN residual block: two dilated causal-padded convs + residual."""
    residual = x
    in_channels = x.shape[-1]

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        activation="relu",
        name=f"{name}_conv1",
    )(x)
    y = tf.keras.layers.LayerNormalization(name=f"{name}_ln1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(y)

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        activation="relu",
        name=f"{name}_conv2",
    )(y)
    y = tf.keras.layers.LayerNormalization(name=f"{name}_ln2")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop2")(y)

    # Project residual if channel count doesn't match.
    if in_channels != filters:
        residual = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=1, padding="same", name=f"{name}_proj",
        )(residual)

    return tf.keras.layers.Add(name=f"{name}_add")([residual, y])


def build_tcn(config: TrainConfig) -> tf.keras.Model:
    """Build a TCN classifier.

    Architecture: Masking → 4 dilated TCN blocks (dilations 1,2,4,8) → GAP → Dense.
    Reads optional ``config.tcn_filters`` (default 64) and ``config.tcn_kernel_size``
    (default 3) so the sweep harness can tune them via YAML.
    """
    filters = getattr(config, "tcn_filters", 64)
    kernel = getattr(config, "tcn_kernel_size", 3)
    dilations = getattr(config, "tcn_dilations", (1, 2, 4, 8))

    inputs = tf.keras.Input(shape=config.input_shape, name="landmark_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0, name="masking")(inputs)

    for i, d in enumerate(dilations):
        x = _tcn_block(
            x,
            filters=filters,
            kernel_size=kernel,
            dilation_rate=d,
            dropout=config.dropout,
            name=f"tcn_d{d}",
        )

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout_dense")(x)
    outputs = tf.keras.layers.Dense(
        config.num_classes, activation="softmax", name="output",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="signlearn_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
