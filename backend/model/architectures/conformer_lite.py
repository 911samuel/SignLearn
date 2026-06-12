"""Conformer-Lite for SignLearn.

A scaled-down Conformer (Gulati et al., 2020) — alternating depthwise-separable
1D conv blocks with multi-head self-attention. Conv captures fine-grained local
motion patterns; self-attention captures global context. Uses pre-LayerNorm
for training stability with a small dataset.

This is the "lite" variant: 2 blocks, 4 heads, d_model=128 — about the same
parameter count as the existing Transformer but with stronger inductive bias
for sequence modeling.
"""

from __future__ import annotations

import tensorflow as tf

from backend.model.config import TrainConfig


def _conv_module(x: tf.Tensor, *, dim: int, kernel: int, dropout: float, name: str) -> tf.Tensor:
    """Conv module: LN → pointwise conv → GLU → depthwise conv → BN → SiLU → pointwise → dropout."""
    y = tf.keras.layers.LayerNormalization(name=f"{name}_ln")(x)
    y = tf.keras.layers.Conv1D(dim * 2, 1, padding="same", name=f"{name}_pw1")(y)
    # GLU: split last dim in half, gate by sigmoid of second half.
    y = tf.keras.layers.Lambda(
        lambda t: t[..., :dim] * tf.sigmoid(t[..., dim:]),
        output_shape=lambda s: (*s[:-1], dim),
        name=f"{name}_glu",
    )(y)
    y = tf.keras.layers.DepthwiseConv1D(
        kernel_size=kernel, padding="same", name=f"{name}_dw",
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(y)
    y = tf.keras.layers.Activation("swish", name=f"{name}_swish")(y)
    y = tf.keras.layers.Conv1D(dim, 1, padding="same", name=f"{name}_pw2")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop")(y)
    return y


def _ff_module(x: tf.Tensor, *, dim: int, expansion: int, dropout: float, name: str) -> tf.Tensor:
    """Macaron-style feed-forward module (residual scaling 1/2 is applied by caller)."""
    y = tf.keras.layers.LayerNormalization(name=f"{name}_ln")(x)
    y = tf.keras.layers.Dense(dim * expansion, activation="swish", name=f"{name}_fc1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(y)
    y = tf.keras.layers.Dense(dim, name=f"{name}_fc2")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop2")(y)
    return y


def _attn_module(
    x: tf.Tensor, *, dim: int, heads: int, dropout: float, name: str,
) -> tf.Tensor:
    """Pre-LN multi-head self-attention."""
    y = tf.keras.layers.LayerNormalization(name=f"{name}_ln")(x)
    y = tf.keras.layers.MultiHeadAttention(
        num_heads=heads, key_dim=dim // heads, dropout=dropout, name=f"{name}_mha",
    )(y, y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop")(y)
    return y


def _conformer_block(
    x: tf.Tensor,
    *,
    dim: int,
    heads: int,
    conv_kernel: int,
    ff_expansion: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    """One Conformer block: 1/2 FF → MHA → Conv → 1/2 FF → LayerNorm."""
    ff1 = _ff_module(x, dim=dim, expansion=ff_expansion, dropout=dropout, name=f"{name}_ff1")
    ff1_scaled = tf.keras.layers.Lambda(lambda t: 0.5 * t, name=f"{name}_ff1_scale")(ff1)
    x = tf.keras.layers.Add(name=f"{name}_add_ff1")([x, ff1_scaled])

    attn = _attn_module(x, dim=dim, heads=heads, dropout=dropout, name=f"{name}_attn")
    x = tf.keras.layers.Add(name=f"{name}_add_attn")([x, attn])

    conv = _conv_module(x, dim=dim, kernel=conv_kernel, dropout=dropout, name=f"{name}_conv")
    x = tf.keras.layers.Add(name=f"{name}_add_conv")([x, conv])

    ff2 = _ff_module(x, dim=dim, expansion=ff_expansion, dropout=dropout, name=f"{name}_ff2")
    ff2_scaled = tf.keras.layers.Lambda(lambda t: 0.5 * t, name=f"{name}_ff2_scale")(ff2)
    x = tf.keras.layers.Add(name=f"{name}_add_ff2")([x, ff2_scaled])

    return tf.keras.layers.LayerNormalization(name=f"{name}_post_ln")(x)


def build_conformer_lite(config: TrainConfig) -> tf.keras.Model:
    """Build a Conformer-Lite classifier.

    Pulls knobs from ``config``: ``transformer_layers`` (block count),
    ``transformer_heads``, ``transformer_d_model``, ``conformer_conv_kernel``
    (default 9), ``conformer_ff_expansion`` (default 4).
    """
    n_blocks = getattr(config, "transformer_layers", 2)
    heads = getattr(config, "transformer_heads", 4)
    dim = getattr(config, "transformer_d_model", 128)
    conv_kernel = getattr(config, "conformer_conv_kernel", 9)
    ff_expansion = getattr(config, "conformer_ff_expansion", 4)

    seq_len = config.input_shape[0]
    inputs = tf.keras.Input(shape=config.input_shape, name="landmark_sequence")

    # Project input → d_model and add learned positional embedding.
    x = tf.keras.layers.Dense(dim, name="input_proj")(inputs)

    class _AddPositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, seq_len: int, dim: int, **kwargs):
            super().__init__(**kwargs)
            self.pos_embed = tf.keras.layers.Embedding(
                input_dim=seq_len, output_dim=dim, name="pos_embed",
            )
            self.seq_len = seq_len

        def call(self, inputs):
            positions = tf.range(start=0, limit=self.seq_len, delta=1)
            return inputs + self.pos_embed(positions)

    x = _AddPositionalEmbedding(seq_len, dim, name="add_pos")(x)

    for i in range(n_blocks):
        x = _conformer_block(
            x,
            dim=dim,
            heads=heads,
            conv_kernel=conv_kernel,
            ff_expansion=ff_expansion,
            dropout=config.dropout,
            name=f"block_{i}",
        )

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout_dense")(x)
    outputs = tf.keras.layers.Dense(
        config.num_classes, activation="softmax", name="output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="signlearn_conformer_lite",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
