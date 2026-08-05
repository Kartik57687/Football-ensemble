"""
Three-layer ensemble training -- Algorithm 1 of the paper.

    Layer 1: train XGBoost, BiLSTM and 1D-CNN independently on D
    Layer 2: concatenate their softmax outputs into x_meta in R^9
    Layer 3: fit multinomial logistic regression on {x_meta, y}

Meta-feature protocol
---------------------
Algorithm 1 (lines 1-12) trains the base models on D and then computes
meta-features over X before fitting the meta-learner -- i.e. the meta-learner
sees the base models' *in-sample* probabilities. That is what
`config.USE_OUT_OF_FOLD_META_FEATURES = False` reproduces, and it is the
default because the brief is to implement the paper as written.

This is worth being explicit about: in-sample stacking lets the meta-learner fit
on probabilities the base models have already memorised, which inflates the
apparent gain of Layer 3. The standard remedy (Wolpert, 1992) is out-of-fold
meta-features, available via `USE_OUT_OF_FOLD_META_FEATURES = True`. Both paths
are implemented; only the flag changes. See docs/unspecified_details.md #5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, data_loader, preprocessing, sequence_builder
from .models import bilstm_model, cnn_model, meta_learner, xgboost_model
from .utils.seed import set_global_seed


@dataclass
class EnsembleArtifacts:
    """Everything produced by a full training run."""

    xgb: Any = None
    bilstm: Any = None
    cnn: Any = None
    meta: Any = None

    standard_scaler: Any = None
    minmax_scaler: Any = None

    feature_columns: list = field(default_factory=list)
    train_idx: np.ndarray = None
    test_idx: np.ndarray = None

    y_train: np.ndarray = None
    y_test: np.ndarray = None

    # Base-learner softmax outputs, columns ordered as config.CLASS_NAMES.
    proba_train: dict = field(default_factory=dict)
    proba_test: dict = field(default_factory=dict)

    X_meta_train: np.ndarray = None
    X_meta_test: np.ndarray = None

    class_weights: dict = field(default_factory=dict)


def prepare_data(df=None, feature_columns=None, verbose=True):
    """
    Load, impute, split, scale and window. Produces every input tensor the three
    base learners need, all aligned on the same 80:20 stratified split.
    """
    if df is None:
        df, feature_columns = data_loader.load_dataset(verbose=verbose)

    df = preprocessing.impute_missing(df, feature_columns, verbose=verbose)
    y_all = data_loader.encode_labels(df[config.TARGET_COLUMN])

    train_idx, test_idx = preprocessing.stratified_split(df, y_all)

    # Tabular inputs (StandardScaler) -- XGBoost and BiLSTM.
    X_train_std, X_test_std, std_scaler = preprocessing.prepare_tabular(
        df, feature_columns, train_idx, test_idx
    )

    # Sequence inputs (Min-Max) -- 1D-CNN.
    (home_tr, away_tr), (home_te, away_te), mm_scaler = sequence_builder.build_cnn_inputs(
        df, feature_columns, train_idx, test_idx, verbose=verbose
    )

    y_train, y_test = y_all[train_idx], y_all[test_idx]

    if verbose:
        print(f"Train / test split        : {len(train_idx):,} / {len(test_idx):,} "
              f"({100 * (1 - config.TEST_SIZE):.0f}:{100 * config.TEST_SIZE:.0f}, stratified)")

    return {
        "df": df,
        "feature_columns": feature_columns,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_all": y_all,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_std": X_train_std,
        "X_test_std": X_test_std,
        "X_train_lstm": preprocessing.reshape_for_bilstm(X_train_std),
        "X_test_lstm": preprocessing.reshape_for_bilstm(X_test_std),
        "home_train": home_tr,
        "away_train": away_tr,
        "home_test": home_te,
        "away_test": away_te,
        "standard_scaler": std_scaler,
        "minmax_scaler": mm_scaler,
    }


def train_base_learners(data: dict, seed=None, verbose=True):
    """Layer 1 -- three heterogeneous models trained independently on the same split."""
    seed = config.RANDOM_SEED if seed is None else seed

    if verbose:
        print("\n[Layer 1] Training base learners")

    xgb = xgboost_model.train_xgboost(
        data["X_train_std"], data["y_train"], seed=seed, verbose=verbose
    )

    bilstm = bilstm_model.train_bilstm(
        data["X_train_lstm"], data["y_train"], seed=seed, verbose=verbose
    )

    class_weights = preprocessing.compute_cnn_class_weights(data["y_train"])
    if verbose:
        pretty = {config.CLASS_NAMES[k]: round(v, 4) for k, v in class_weights.items()}
        print(f"  CNN class weights: {pretty}")

    cnn = cnn_model.train_cnn(
        data["home_train"],
        data["away_train"],
        data["y_train"],
        class_weights=class_weights,
        seed=seed,
        verbose=verbose,
    )

    return xgb, bilstm, cnn, class_weights


def base_probabilities(xgb, bilstm, cnn, data: dict, split: str):
    """
    Softmax outputs of the three base learners for one split (Eq. 12).

    Returns an ordered dict; its ordering fixes the meta-feature column order.
    """
    suffix = "train" if split == "train" else "test"
    return {
        "XGBoost": xgboost_model.predict_proba_xgboost(xgb, data[f"X_{suffix}_std"]),
        "BiLSTM": bilstm_model.predict_proba_bilstm(bilstm, data[f"X_{suffix}_lstm"]),
        "CNN": cnn_model.predict_proba_cnn(
            cnn, data[f"home_{suffix}"], data[f"away_{suffix}"]
        ),
    }


def _out_of_fold_meta_features(data: dict, seed, verbose=True):
    """
    Leakage-free alternative to Algorithm 1's in-sample meta-features.

    Only used when config.USE_OUT_OF_FOLD_META_FEATURES is True. This is NOT the
    paper's protocol; it is offered for comparison.
    """
    n = len(data["y_train"])
    oof = {name: np.zeros((n, config.N_CLASSES), dtype=np.float32)
           for name in ("XGBoost", "BiLSTM", "CNN")}

    skf = StratifiedKFold(n_splits=config.OOF_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr, va) in enumerate(skf.split(np.zeros(n), data["y_train"]), start=1):
        if verbose:
            print(f"  OOF fold {fold}/{config.OOF_FOLDS}")

        fold_xgb = xgboost_model.train_xgboost(
            data["X_train_std"][tr], data["y_train"][tr], seed=seed, verbose=False
        )
        fold_lstm = bilstm_model.train_bilstm(
            data["X_train_lstm"][tr], data["y_train"][tr], seed=seed, verbose=False
        )
        fold_cnn = cnn_model.train_cnn(
            data["home_train"][tr],
            data["away_train"][tr],
            data["y_train"][tr],
            class_weights=preprocessing.compute_cnn_class_weights(data["y_train"][tr]),
            seed=seed,
            verbose=False,
        )

        oof["XGBoost"][va] = xgboost_model.predict_proba_xgboost(
            fold_xgb, data["X_train_std"][va])
        oof["BiLSTM"][va] = bilstm_model.predict_proba_bilstm(
            fold_lstm, data["X_train_lstm"][va])
        oof["CNN"][va] = cnn_model.predict_proba_cnn(
            fold_cnn, data["home_train"][va], data["away_train"][va])

    return oof


def train_ensemble(data: dict | None = None, seed=None, verbose=True) -> EnsembleArtifacts:
    """
    End-to-end Algorithm 1: Layer 1 -> Layer 2 -> Layer 3.
    """
    seed = set_global_seed(seed)
    if data is None:
        data = prepare_data(verbose=verbose)

    xgb, bilstm, cnn, class_weights = train_base_learners(data, seed=seed, verbose=verbose)

    if verbose:
        print("\n[Layer 2] Constructing meta-feature matrix")

    proba_test = base_probabilities(xgb, bilstm, cnn, data, "test")

    if config.USE_OUT_OF_FOLD_META_FEATURES:
        if verbose:
            print("  Mode: out-of-fold (NON-paper variant)")
        proba_train = _out_of_fold_meta_features(data, seed, verbose=verbose)
    else:
        if verbose:
            print("  Mode: in-sample, per Algorithm 1")
        proba_train = base_probabilities(xgb, bilstm, cnn, data, "train")

    order = ("XGBoost", "BiLSTM", "CNN")
    X_meta_train = meta_learner.build_meta_features(*[proba_train[m] for m in order])
    X_meta_test = meta_learner.build_meta_features(*[proba_test[m] for m in order])

    if verbose:
        print(f"  X_meta shape: {X_meta_train.shape} (n x {config.N_META_FEATURES})")
        print("\n[Layer 3] Training meta-learner")

    meta = meta_learner.train_meta_learner(X_meta_train, data["y_train"], verbose=verbose)

    return EnsembleArtifacts(
        xgb=xgb,
        bilstm=bilstm,
        cnn=cnn,
        meta=meta,
        standard_scaler=data["standard_scaler"],
        minmax_scaler=data["minmax_scaler"],
        feature_columns=data["feature_columns"],
        train_idx=data["train_idx"],
        test_idx=data["test_idx"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        proba_train=proba_train,
        proba_test=proba_test,
        X_meta_train=X_meta_train,
        X_meta_test=X_meta_test,
        class_weights=class_weights,
    )


def save_models(artifacts: EnsembleArtifacts, out_dir=None, verbose=True):
    """Persist the fitted models and scalers to outputs/models/."""
    import joblib

    out_dir = out_dir or config.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts.xgb.save_model(out_dir / "xgboost_model.json")
    artifacts.bilstm.save(out_dir / "bilstm_model.keras")
    artifacts.cnn.save(out_dir / "cnn_model.keras")
    joblib.dump(artifacts.meta, out_dir / "meta_learner_logreg.joblib")
    joblib.dump(artifacts.standard_scaler, out_dir / "standard_scaler.joblib")
    joblib.dump(artifacts.minmax_scaler, out_dir / "minmax_scaler.joblib")

    if verbose:
        print(f"Models saved to {out_dir}")
