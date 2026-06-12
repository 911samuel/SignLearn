"""Bidirectional LSTM architecture for SignLearn (Phase 5)."""

from __future__ import annotations

import tensorflow as tf

from backend.model.config import TrainConfig


def build_bilstm(config: TrainConfig) -> tf.keras.Model:
    """BiLSTM(128) → BiLSTM(64) → Dense → Softmax.

    Bidirectional layers let each timestep attend to past *and* future context,
    which is well-suited to fixed-window sign clips where the discriminative
    motion may occur anywhere in the 30-frame buffer.
    """
    inputs = tf.keras.Input(shape=config.input_shape, name="landmark_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0, name="masking")(inputs)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config.lstm_units[0],
            return_sequences=True,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout,
        ),
        name="bilstm_1",
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config.lstm_units[1],
            return_sequences=False,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout,
        ),
        name="bilstm_2",
    )(x)

    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout_dense")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="signlearn_bilstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
