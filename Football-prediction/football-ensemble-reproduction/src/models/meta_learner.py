"""
Layer 2 (meta-feature construction) and Layer 3 (meta-learner).

Layer 2 -- Section IV-B:
    "Instead of relying upon the actual predictions of the base learners, we
     take them out of the black box and look at the softmax probability
     distributions over the three output classes (Win, Draw, Loss)."

    x_meta = [p^(1), p^(2), p^(3)] in R^9                            (Eq. 13)
    X_meta in R^{n x 9}

Layer 3 -- Section IV-C:
    "we use a multinomial LR classifier on the meta-feature matrix."

    P(y = k | x_meta) = exp(w_k^T x_meta) / sum_j exp(w_j^T x_meta)   (Eq. 14)

Chosen for interpretability, calibration, and low overfitting risk (Section IV-D).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import config


def build_meta_features(*probability_matrices: np.ndarray) -> np.ndarray:
    """
    Concatenate base-learner softmax outputs into X_meta (Eq. 13).

    Argument order defines column order and must be identical between the
    fitting call and every prediction call.
    """
    for i, p in enumerate(probability_matrices):
        if p.shape[1] != config.N_CLASSES:
            raise ValueError(
                f"Base model {i} produced {p.shape[1]} columns, expected {config.N_CLASSES}"
            )
    n_rows = {p.shape[0] for p in probability_matrices}
    if len(n_rows) != 1:
        raise ValueError(f"Base models disagree on row count: {n_rows}")

    # The full ensemble yields 9 columns (Eq. 13). Ablation variants drop a base
    # model and therefore legitimately yield 6, so the width is checked as a
    # multiple of N_CLASSES rather than pinned to 9.
    X_meta = np.hstack(probability_matrices).astype(np.float32)
    expected = len(probability_matrices) * config.N_CLASSES
    if X_meta.shape[1] != expected:
        raise ValueError(f"Expected {expected} meta-features, got {X_meta.shape[1]}")
    return X_meta


def meta_feature_names(model_names=("XGBoost", "BiLSTM", "CNN")) -> list[str]:
    """Column labels for X_meta, used when inspecting learned LR weights."""
    return [f"{m}_P({c})" for m in model_names for c in config.CLASS_NAMES]


def build_meta_learner(params: dict | None = None, seed: int | None = None):
    """
    Multinomial logistic regression with the Table II settings
    (L2 penalty, lbfgs solver, max_iter 1000, tol 1e-4).

    Compatibility note: scikit-learn 1.8 deprecated the explicit `penalty`
    argument in favour of `l1_ratio` / `C`. Plain L2 is the default there, so on
    1.8+ the argument is dropped rather than passed -- the fitted model is
    identical, only the API changed.
    """
    params = dict(params or config.META_LEARNER_PARAMS)
    seed = config.RANDOM_SEED if seed is None else seed

    if params.get("penalty") == "l2":
        import sklearn
        major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
        if (major, minor) >= (1, 8):
            params.pop("penalty")

    return LogisticRegression(random_state=seed, **params)


def train_meta_learner(X_meta_train, y_train, params=None, seed=None, verbose=True):
    """Fit Layer 3 on the 9-dimensional meta-feature matrix."""
    model = build_meta_learner(params, seed)
    model.fit(X_meta_train, y_train)
    if verbose:
        print("  Logistic Regression meta-learner trained.")
    return model


def predict_proba_meta(model, X_meta) -> np.ndarray:
    """Final ensemble class probabilities (Eq. 14)."""
    return model.predict_proba(X_meta).astype(np.float32)


def meta_learner_weights(model, model_names=("XGBoost", "BiLSTM", "CNN")) -> pd.DataFrame:
    """
    Expose the learned weights w_k.

    Section IV-D: "Interpretability: The learned weights reflect the
    contribution of each base model to each class."
    """
    return pd.DataFrame(
        model.coef_,
        index=[f"class={c}" for c in config.CLASS_NAMES],
        columns=meta_feature_names(model_names),
    )
