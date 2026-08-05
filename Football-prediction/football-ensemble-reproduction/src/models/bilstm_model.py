"""
Layer 1 base learner: BiLSTM -- Section IV-A.

    "Bidirectional Long Short-Term Memory (BiLSTM) networks extend traditional
     LSTMs by processing sequences in both forward and backward directions,
     effectively learning past and future temporal dependencies."

Input shape follows Section III: (n, 1, d).

UNSPECIFIED (docs/unspecified_details.md #8): the paper gives only
"Hidden Units: 64", "Dropout: 0.3" and "Activation: tanh". It does not state the
number of recurrent layers or the classification head. A single Bidirectional
LSTM layer followed by dropout and a softmax Dense layer is the minimal
architecture consistent with those three values; nothing further is added.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .. import config


def build_bilstm(input_shape, params: dict | None = None, seed: int | None = None):
    """
    Single Bidirectional LSTM -> Dropout -> softmax Dense.

    Parameters
    ----------
    input_shape : tuple
        (timesteps, n_features) -- (1, d) per Section III.
    """
    params = params or config.BILSTM_PARAMS
    seed = config.RANDOM_SEED if seed is None else seed
    tf.keras.utils.set_random_seed(seed)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Bidirectional(
                layers.LSTM(
                    params["hidden_units"],
                    activation=params["activation"],
                    return_sequences=False,
                )
            ),
            layers.Dropout(params["dropout"]),
            layers.Dense(config.N_CLASSES, activation="softmax"),
        ],
        name="bilstm_base_learner",
    )

    model.compile(
        optimizer=params["optimizer"],
        loss=params["loss"],  # sparse_categorical_crossentropy (Table II)
        metrics=["accuracy"],
    )
    return model


def train_bilstm(X_train_3d, y_train, params=None, seed=None, verbose=True):
    """
    Fit the BiLSTM. Labels stay integer-encoded because Table II specifies
    sparse categorical cross-entropy for this model.

    The paper applies class weights to the CNN only, so none are used here.
    """
    params = params or config.BILSTM_PARAMS
    model = build_bilstm(X_train_3d.shape[1:], params, seed)

    model.fit(
        X_train_3d,
        y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2 if verbose else 0,
    )
    if verbose:
        print("  BiLSTM trained.")
    return model


def predict_proba_bilstm(model, X_3d) -> np.ndarray:
    """Softmax probability vector p^(2) per match (Eq. 12)."""
    return model.predict(X_3d, verbose=0).astype(np.float32)
