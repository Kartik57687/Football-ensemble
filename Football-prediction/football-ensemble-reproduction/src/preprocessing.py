"""
Preprocessing -- Section III of the paper.

Two scalers are produced, exactly as the paper specifies:

    "StandardScaler: for XGBoost and BiLSTM input
     Min-Max Normalization: used earlier during sequence construction for CNN"

Both are fitted on the training split only and applied to the test split, so
that no test-set statistics leak into training. The paper does not state where
the scalers are fitted; fitting on train only is the conservative reading.
See docs/unspecified_details.md #3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from . import config


def impute_missing(df: pd.DataFrame, feature_columns: list[str], verbose: bool = True):
    """
    Section III: "Missing or anomalous values are either imputed with median
    values or dropped based on the context."

    Median imputation is applied to the numerical predictors. Rows are dropped
    only if the target itself cannot be derived.
    """
    df = df.copy()
    n_missing = int(df[feature_columns].isna().sum().sum())

    if n_missing:
        medians = df[feature_columns].median()
        df[feature_columns] = df[feature_columns].fillna(medians)

    before = len(df)
    df = df.dropna(subset=[config.TARGET_COLUMN])
    dropped = before - len(df)

    if verbose:
        print(f"Missing feature values imputed (median): {n_missing:,}")
        print(f"Rows dropped (undefined target)        : {dropped:,}")

    return df


def stratified_split(df: pd.DataFrame, y: np.ndarray, test_size=None, seed=None):
    """
    80:20 stratified split (Section III).

    Returns positional index arrays rather than frames, so that the tabular
    features and the CNN sequence tensors can be split with the *same* indices.
    Alignment matters: Layer 2 concatenates the three base models' softmax
    outputs row-by-row (Eq. 13).
    """
    test_size = config.TEST_SIZE if test_size is None else test_size
    seed = config.RANDOM_SEED if seed is None else seed

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y if config.STRATIFY else None,
    )
    return train_idx, test_idx


def fit_standard_scaler(X_train: np.ndarray):
    """StandardScaler for the XGBoost and BiLSTM inputs."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def fit_minmax_scaler(X_train: np.ndarray):
    """Min-Max scaler used before CNN sequence construction."""
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def reshape_for_bilstm(X: np.ndarray) -> np.ndarray:
    """
    Section III: "The scaled tabular data is reshaped into a three-dimensional
    format: (n, 1, d) ... despite the fact that the input window size is one
    (non-sequential/sample)."
    """
    return X.reshape((X.shape[0], config.BILSTM_PARAMS["timesteps"], X.shape[1]))


def compute_cnn_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """
    Section III: "while training CNN, we computed class weights and used them in
    the loss function to adjust for representations of the dominant classes."

    Applied to the CNN only -- the paper does not apply class weights to XGBoost
    or the BiLSTM.
    """
    classes = np.arange(config.N_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def prepare_tabular(df, feature_columns, train_idx, test_idx):
    """
    Produce the standardised tabular matrices for XGBoost and the BiLSTM.

    Returns (X_train_std, X_test_std, scaler).
    """
    X = df[feature_columns].to_numpy(dtype=np.float32)
    scaler = fit_standard_scaler(X[train_idx])
    return (
        scaler.transform(X[train_idx]).astype(np.float32),
        scaler.transform(X[test_idx]).astype(np.float32),
        scaler,
    )
