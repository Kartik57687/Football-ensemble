"""
Layer 1 base learner: 1D-CNN -- Sections III and IV-A.

    "Although separate home and away sequences are splitted, they are processed
     concurrently in the same pipeline using parallel 1D convolutional core
     components before joining into a common decision layer."

That gives a two-branch functional model: one Conv1D stack per team-history
tensor, concatenated into a shared classification head.

Convolution and pooling follow Eqs. 10-11:
    z_t = sigma(w . x_{t:t+k-1} + b)
    p   = Pool(z_1, ..., z_{T-k+1})
"joined... This output is flattened and passed to fully connected layers."

UNSPECIFIED (docs/unspecified_details.md #9): Table II gives one value each for
filters (128), kernel size (3), pool size (2) and dropout (0.3), which implies a
single convolution + pooling block per branch. The width of the shared dense
layer after concatenation is not stated; it is set to `filters` (128) so that no
new magnitude is invented, and is exposed as `dense_units` for transparency.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .. import config


def build_cnn(sequence_shape, params: dict | None = None, seed: int | None = None):
    """
    Parallel Conv1D branches (home history, away history) -> concatenate ->
    dense decision layer -> softmax.

    Parameters
    ----------
    sequence_shape : tuple
        (window_size, n_features) -- (15, d) per Section III.
    """
    params = params or config.CNN_PARAMS
    seed = config.RANDOM_SEED if seed is None else seed
    tf.keras.utils.set_random_seed(seed)

    dense_units = params.get("dense_units", params["filters"])

    home_in = layers.Input(shape=sequence_shape, name="home_sequence")
    away_in = layers.Input(shape=sequence_shape, name="away_sequence")

    def conv_branch(x, tag: str):
        """One 1D convolutional core component (Eqs. 10-11)."""
        x = layers.Conv1D(
            filters=params["filters"],
            kernel_size=params["kernel_size"],
            activation=params["activation"],
            padding="valid",
            name=f"{tag}_conv1d",
        )(x)
        x = layers.MaxPooling1D(pool_size=params["pool_size"], name=f"{tag}_pool")(x)
        x = layers.Flatten(name=f"{tag}_flatten")(x)
        return x

    merged = layers.Concatenate(name="common_decision_layer")(
        [conv_branch(home_in, "home"), conv_branch(away_in, "away")]
    )
    merged = layers.Dropout(params["dropout"], name="dropout")(merged)
    merged = layers.Dense(dense_units, activation=params["activation"], name="dense")(merged)
    out = layers.Dense(config.N_CLASSES, activation="softmax", name="softmax")(merged)

    model = models.Model(inputs=[home_in, away_in], outputs=out, name="dual_branch_1dcnn")
    model.compile(
        optimizer=params["optimizer"],
        loss=params["loss"],  # categorical_crossentropy (Table II)
        metrics=["accuracy"],
    )
    return model


def train_cnn(
    home_train,
    away_train,
    y_train,
    class_weights=None,
    params=None,
    seed=None,
    verbose=True,
):
    """
    Fit the dual-branch CNN.

    Table II specifies categorical cross-entropy, so integer labels are one-hot
    encoded here. Section III specifies class weights for this model only.
    """
    params = params or config.CNN_PARAMS
    model = build_cnn(home_train.shape[1:], params, seed)

    y_onehot = tf.keras.utils.to_categorical(y_train, num_classes=config.N_CLASSES)

    model.fit(
        [home_train, away_train],
        y_onehot,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        class_weight=class_weights if params.get("use_class_weights", True) else None,
        verbose=2 if verbose else 0,
    )
    if verbose:
        print("  1D-CNN trained.")
    return model


def predict_proba_cnn(model, home_seq, away_seq) -> np.ndarray:
    """Softmax probability vector p^(3) per match (Eq. 12)."""
    return model.predict([home_seq, away_seq], verbose=0).astype(np.float32)
