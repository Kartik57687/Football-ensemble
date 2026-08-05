"""
Layer 1 base learner: XGBoost -- Section IV-A.

    "Extreme Gradient Boosting (XGBoost) is an ensemble of decision trees that
     is created on the principle of additive gradient boosting... XGBoost is
     ideal for structured tabular match statistics (e.g., possession, pass
     accuracy) and offers robustness to outliers and multicollinearity."

Objective (Eqs. 1-3) is the standard regularised softmax objective; it is
supplied by the `multi:softprob` objective together with `reg_lambda` (lambda in
Eq. 2). Hyperparameters come from Table II.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from .. import config


def build_xgboost(params: dict | None = None, seed: int | None = None) -> XGBClassifier:
    """Instantiate the XGBoost base learner with the paper's Table II settings."""
    params = dict(params or config.XGBOOST_PARAMS)
    seed = config.RANDOM_SEED if seed is None else seed

    # XGBClassifier infers num_class from the labels; passing it explicitly
    # raises in recent versions.
    params.pop("num_class", None)
    params["random_state"] = seed

    return XGBClassifier(**params)


def train_xgboost(X_train, y_train, params=None, seed=None, verbose=True):
    """Fit XGBoost on the standardised tabular features."""
    model = build_xgboost(params, seed)
    model.fit(X_train, y_train, verbose=False)
    if verbose:
        print("  XGBoost trained.")
    return model


def predict_proba_xgboost(model, X) -> np.ndarray:
    """
    Softmax probability vector p^(1) = [p_W, p_D, p_L] per match (Eq. 12).

    Column order follows config.CLASS_NAMES.
    """
    return model.predict_proba(X).astype(np.float32)
