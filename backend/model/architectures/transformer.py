"""Transformer encoder architecture for SignLearn (Phase 5).

A small Transformer encoder over the time axis. Each timestep is projected
into ``d_model`` dimensions, summed with learned positional embeddings, then
passed through ``transformer_layers`` blocks of (multi-head self-attention +
feed-forward). Global average pooling collapses the time axis before the
classification head.
"""

from __future__ import annotations

import tensorflow as tf

from backend.model.config import TrainConfig


def _encoder_block(x: tf.Tensor, *, d_model: int, n_heads: int, ff_dim: int,
                   dropout: float, name: str) -> tf.Tensor:
    """One pre-norm Transformer encoder block."""
    # Self-attention sub-layer.
    norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
        dropout=dropout,
        name=f"{name}_mha",
    )(norm1, norm1)
    attn = tf.keras.layers.Dropout(dropout, name=f"{name}_attn_drop")(attn)
    x = tf.keras.layers.Add(name=f"{name}_resid1")([x, attn])

    # Feed-forward sub-layer.
    norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    ff = tf.keras.layers.Dense(ff_dim, activation="gelu", name=f"{name}_ff1")(norm2)
    ff = tf.keras.layers.Dropout(dropout, name=f"{name}_ff_drop")(ff)
    ff = tf.keras.layers.Dense(d_model, name=f"{name}_ff2")(ff)
    x = tf.keras.layers.Add(name=f"{name}_resid2")([x, ff])
    return x


def build_transformer(config: TrainConfig) -> tf.keras.Model:
    """Build a small Transformer encoder for sign-sequence classification."""
    seq_len, feat_dim = config.input_shape
    d_model = config.transformer_d_model
    n_heads = config.transformer_heads
    ff_dim  = config.transformer_ff_dim
    n_layers = config.transformer_layers

    inputs = tf.keras.Input(shape=config.input_shape, name="landmark_sequence")

    # Project features to model dimension. (No Masking layer — attention handles
    # zero frames implicitly via low contribution; padding-mask support can be
    # added later if mixed-length sequences become common.)
    x = tf.keras.layers.Dense(d_model, name="input_proj")(inputs)

    # Learned positional embeddings (small, only seq_len of them).
    pos = tf.keras.layers.Embedding(
        input_dim=seq_len, output_dim=d_model, name="pos_embed",
    )(tf.range(start=0, limit=seq_len, delta=1))
    x = x + pos

    x = tf.keras.layers.Dropout(config.dropout, name="input_drop")(x)

    for i in range(n_layers):
        x = _encoder_block(
            x, d_model=d_model, n_heads=n_heads, ff_dim=ff_dim,
            dropout=config.dropout, name=f"enc_{i}",
        )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="time_pool")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout_dense")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="signlearn_transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
