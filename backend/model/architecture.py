"""Stacked LSTM architecture for ASL sign recognition.

Input:  (batch, 30, 126)  — both-hands landmark sequences
Output: (batch, num_classes)  — softmax probabilities over compact class set
"""

import tensorflow as tf

from backend.model.config import TrainConfig


def build_lstm(config: TrainConfig) -> tf.keras.Model:
    """Build, compile, and return the stacked LSTM model.

    Architecture:
        Masking → LSTM(128, return_sequences=True) → LSTM(64)
        → Dense(64, relu) → Dropout → Dense(num_classes, softmax)

    The Masking layer treats all-zero frames (no hand detected) as padding so
    they don't contribute to LSTM state updates.
    """
    inputs = tf.keras.Input(shape=config.input_shape, name="landmark_sequence")

    x = tf.keras.layers.Masking(mask_value=0.0, name="masking")(inputs)

    x = tf.keras.layers.LSTM(
        config.lstm_units[0],
        return_sequences=True,
        dropout=config.dropout,
        recurrent_dropout=config.recurrent_dropout,
        name="lstm_1",
    )(x)

    x = tf.keras.layers.LSTM(
        config.lstm_units[1],
        return_sequences=False,
        dropout=config.dropout,
        recurrent_dropout=config.recurrent_dropout,
        name="lstm_2",
    )(x)

    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout_dense")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="signlearn_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
