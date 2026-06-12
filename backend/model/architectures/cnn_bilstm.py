"""CNN-BiLSTM hybrid for SignLearn.

A 1D convolutional front-end extracts local temporal motifs (3-5 frame patterns
like quick wrist twists, finger taps) before a recurrent stack models long-range
dependencies. This is a well-validated pattern for skeleton-based action
recognition — convs catch what BiLSTM alone smears over its hidden state.
"""

from __future__ import annotations

import tensorflow as tf

from backend.model.config import TrainConfig


def build_cnn_bilstm(config: TrainConfig) -> tf.keras.Model:
    """Conv1D(64,5) → Conv1D(64,5) → BiLSTM(128) → BiLSTM(64) → Dense.

    Reads optional ``config.conv_filters`` (default 64) and
    ``config.conv_kernel_size`` (default 5).
    """
    conv_filters = getattr(config, "conv_filters", 64)
    conv_kernel = getattr(config, "conv_kernel_size", 5)

    inputs = tf.keras.Input(shape=config.input_shape, name="landmark_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0, name="masking")(inputs)

    # Local temporal feature extraction.
    x = tf.keras.layers.Conv1D(
        conv_filters, conv_kernel, padding="same", activation="relu", name="conv1",
    )(x)
    x = tf.keras.layers.LayerNormalization(name="ln1")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="drop1")(x)

    x = tf.keras.layers.Conv1D(
        conv_filters, conv_kernel, padding="same", activation="relu", name="conv2",
    )(x)
    x = tf.keras.layers.LayerNormalization(name="ln2")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="drop2")(x)

    # Bidirectional temporal modeling.
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
    outputs = tf.keras.layers.Dense(
        config.num_classes, activation="softmax", name="output",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="signlearn_cnn_bilstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
