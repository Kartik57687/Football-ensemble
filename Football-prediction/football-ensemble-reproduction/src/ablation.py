"""
Ablation study -- Section V-D, Table V.

    "To assess the contribution of each base model, we conducted an ablation
     study systematically by removing one model at a time from the ensemble and
     evaluating the resulting performance."

Variants: Full Ensemble (All), w/o CNN (XGB+LSTM), w/o BiLSTM (XGB+CNN),
w/o XGBoost (LSTM+CNN).

The base learners are trained once and reused; only the meta-feature matrix and
the Layer 3 logistic regression are rebuilt per variant. Retraining the base
models would confound "removing a model from the ensemble" with base-model
seed variance, and the paper describes removal from the ensemble, not
retraining.

Meta-feature dimensionality drops from 9 to 6 for the two-model variants, which
is expected: Eq. 13 concatenates 3 classes per retained base model.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .models import meta_learner
from .utils import metrics as M

ABLATION_VARIANTS = {
    "Full Ensemble (All)": ("XGBoost", "BiLSTM", "CNN"),
    "w/o CNN (XGB+LSTM)": ("XGBoost", "BiLSTM"),
    "w/o BiLSTM (XGB+CNN)": ("XGBoost", "CNN"),
    "w/o XGBoost (LSTM+CNN)": ("BiLSTM", "CNN"),
}


def run_ablation(artifacts, verbose=True) -> pd.DataFrame:
    """
    Refit Layer 3 on each subset of base learners and score on the test split.
    """
    rows = []

    for variant, members in ABLATION_VARIANTS.items():
        Xm_train = meta_learner.build_meta_features(
            *[artifacts.proba_train[m] for m in members]
        )
        Xm_test = meta_learner.build_meta_features(
            *[artifacts.proba_test[m] for m in members]
        )

        meta = meta_learner.train_meta_learner(
            Xm_train, artifacts.y_train, verbose=False
        )
        preds = meta.predict(Xm_test)

        m = M.compute_all_metrics(artifacts.y_test, preds)
        rows.append({
            "Variant": variant,
            "Base learners": " + ".join(members),
            "Meta-features": Xm_train.shape[1],
            "Accuracy": m["accuracy"],
            "Precision": m["precision_macro"],
            "Recall": m["recall_macro"],
            "F1-score": m["f1_macro"],
        })

        if verbose:
            print(f"  {variant:<24} acc={m['accuracy']:.4f}  "
                  f"macroF1={m['f1_macro']:.4f}")

    df = pd.DataFrame(rows).set_index("Variant")

    if verbose:
        full_acc = df.loc["Full Ensemble (All)", "Accuracy"]
        df_out = df.copy()
        df_out["Delta vs full"] = df_out["Accuracy"] - full_acc

        print("\n" + "=" * 78)
        print("TABLE V -- Ablation Study: Removing One Base Model at a Time")
        print("=" * 78)
        print(df_out[["Base learners", "Meta-features", "Accuracy", "Precision",
                      "Recall", "F1-score", "Delta vs full"]].round(4).to_string())
        print("\nA larger negative delta means the removed model contributed more.")

    return df


def save_ablation(df, out_dir=None, verbose=True):
    """Persist Table V."""
    out_dir = out_dir or config.METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "table5_ablation_study.csv")
    if verbose:
        print(f"Ablation results saved to {out_dir}")
