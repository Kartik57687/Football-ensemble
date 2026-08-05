"""
5-fold stratified cross-validation -- Section V-B, Table III.

    "we implemented 5-fold stratified cross-validation across the dataset.
     Stratified cross-validation helps ensure that each fold reflects the
     original class distribution (Win, Draw, Loss)... We report the mean and
     standard deviation of useful metrics (Accuracy, Precision, Recall, and
     F1-score) for all base models and the final ensemble learner."

Per-fold accuracy vectors are retained so that statistical_tests.py can run the
paired t-test and Wilcoxon signed-rank test of Section V-C on exactly the same
numbers.

Scalers are refitted inside each fold on that fold's training portion, so no
fold's validation statistics leak into its own preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, preprocessing, sequence_builder
from .models import bilstm_model, cnn_model, meta_learner, xgboost_model
from .utils import metrics as M
from .utils.seed import set_global_seed

MODEL_ORDER = ("XGBoost", "BiLSTM", "CNN", "Ensemble (LR)")


def run_cross_validation(df, feature_columns, n_folds=None, seed=None, verbose=True):
    """
    Run stratified K-fold CV over the whole dataset.

    Returns
    -------
    summary : DataFrame
        Mean +/- std of each metric per model (Table III).
    per_fold : DataFrame
        Long-form per-fold scores, consumed by statistical_tests.py.
    """
    from . import data_loader

    n_folds = n_folds or config.CV_FOLDS
    seed = set_global_seed(seed)

    df = preprocessing.impute_missing(df, feature_columns, verbose=False)
    y_all = data_loader.encode_labels(df[config.TARGET_COLUMN])
    X_all = df[feature_columns].to_numpy(dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    records = []

    for fold, (tr, va) in enumerate(skf.split(X_all, y_all), start=1):
        if verbose:
            print(f"\n--- Fold {fold}/{n_folds} "
                  f"(train {len(tr):,} / val {len(va):,}) ---")

        y_tr, y_va = y_all[tr], y_all[va]

        # Tabular branch: StandardScaler fitted on this fold's training portion.
        std_scaler = preprocessing.fit_standard_scaler(X_all[tr])
        X_tr = std_scaler.transform(X_all[tr]).astype(np.float32)
        X_va = std_scaler.transform(X_all[va]).astype(np.float32)
        X_tr_lstm = preprocessing.reshape_for_bilstm(X_tr)
        X_va_lstm = preprocessing.reshape_for_bilstm(X_va)

        # Sequence branch: Min-Max fitted on this fold's training portion.
        mm_scaler = preprocessing.fit_minmax_scaler(X_all[tr])
        X_mm = mm_scaler.transform(X_all).astype(np.float32)
        home_seq, away_seq = sequence_builder.build_team_sequences(
            df, X_mm, verbose=False
        )

        # Layer 1
        xgb = xgboost_model.train_xgboost(X_tr, y_tr, seed=seed, verbose=False)
        lstm = bilstm_model.train_bilstm(X_tr_lstm, y_tr, seed=seed, verbose=False)
        cnn = cnn_model.train_cnn(
            home_seq[tr], away_seq[tr], y_tr,
            class_weights=preprocessing.compute_cnn_class_weights(y_tr),
            seed=seed, verbose=False,
        )

        # Layer 2
        p_tr = {
            "XGBoost": xgboost_model.predict_proba_xgboost(xgb, X_tr),
            "BiLSTM": bilstm_model.predict_proba_bilstm(lstm, X_tr_lstm),
            "CNN": cnn_model.predict_proba_cnn(cnn, home_seq[tr], away_seq[tr]),
        }
        p_va = {
            "XGBoost": xgboost_model.predict_proba_xgboost(xgb, X_va),
            "BiLSTM": bilstm_model.predict_proba_bilstm(lstm, X_va_lstm),
            "CNN": cnn_model.predict_proba_cnn(cnn, home_seq[va], away_seq[va]),
        }
        order = ("XGBoost", "BiLSTM", "CNN")
        Xm_tr = meta_learner.build_meta_features(*[p_tr[m] for m in order])
        Xm_va = meta_learner.build_meta_features(*[p_va[m] for m in order])

        # Layer 3
        meta = meta_learner.train_meta_learner(Xm_tr, y_tr, verbose=False)

        preds = {name: p_va[name].argmax(axis=1) for name in order}
        preds["Ensemble (LR)"] = meta.predict(Xm_va)

        for name in MODEL_ORDER:
            m = M.compute_all_metrics(y_va, preds[name])
            records.append({"fold": fold, "model": name, **m})
            if verbose:
                print(f"  {name:<15} acc={m['accuracy']:.4f}  "
                      f"macroF1={m['f1_macro']:.4f}")

    per_fold = pd.DataFrame(records)

    agg_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    summary = (
        per_fold.groupby("model")[agg_cols]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
    )
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]

    if verbose:
        print("\n" + "=" * 78)
        print(f"TABLE III -- {n_folds}-Fold Cross-Validation Results (Mean +/- Std, %)")
        print("=" * 78)
        print(_format_table3(summary).to_string())

    return summary, per_fold


def _format_table3(summary: pd.DataFrame) -> pd.DataFrame:
    """Render the summary in the paper's 'mean ± std' style."""
    out = pd.DataFrame(index=summary.index)
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("f1_macro", "F1-score"),
    ]:
        out[label] = [
            f"{100 * m:.2f}% ± {100 * s:.2f}"
            for m, s in zip(summary[f"{metric}_mean"], summary[f"{metric}_std"])
        ]
    return out


def save_cross_validation(summary, per_fold, out_dir=None, verbose=True):
    """Persist Table III and the per-fold scores."""
    from .utils import plots

    out_dir = out_dir or config.METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_dir / "table3_cross_validation_summary.csv")
    _format_table3(summary).to_csv(out_dir / "table3_formatted.csv")
    per_fold.to_csv(out_dir / "cross_validation_per_fold.csv", index=False)

    plots.plot_cv_results(summary)

    if verbose:
        print(f"Cross-validation results saved to {out_dir}")
