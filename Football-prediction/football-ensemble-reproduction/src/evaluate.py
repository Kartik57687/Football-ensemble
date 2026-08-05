"""
Evaluation on the held-out 20% test split -- Section V, Table VI and Fig. 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .models import meta_learner
from .utils import metrics as M
from .utils import plots


def evaluate_ensemble(artifacts, verbose: bool = True):
    """
    Score the three base learners and the final ensemble on the test split.

    Returns a dict holding the Table VI comparison, per-class breakdown,
    confusion matrix and the meta-learner's interpretable weights.
    """
    y_test = artifacts.y_test

    predictions = {
        name: proba.argmax(axis=1) for name, proba in artifacts.proba_test.items()
    }
    predictions["Ensemble (LR)"] = artifacts.meta.predict(artifacts.X_meta_test)

    results = {name: M.compute_all_metrics(y_test, pred)
               for name, pred in predictions.items()}

    table = M.results_table(results)
    cm = M.get_confusion_matrix(y_test, predictions["Ensemble (LR)"])
    per_class = M.per_class_report(y_test, predictions["Ensemble (LR)"])
    weights = meta_learner.meta_learner_weights(artifacts.meta)

    if verbose:
        print("\n" + "=" * 78)
        print("TABLE VI -- Performance on the held-out test set (%)")
        print("=" * 78)
        print(M.format_percent_table(
            table[["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]]
        ).to_string())
        print("\nSame table, weighted averaging:")
        print(M.format_percent_table(
            table[["Accuracy", "Precision (weighted)", "Recall (weighted)",
                   "F1 (weighted)"]]
        ).to_string())

        print("\n" + "-" * 78)
        print("Per-class breakdown -- final ensemble")
        print("-" * 78)
        print(per_class.round(4).to_string())

        print("\n" + "-" * 78)
        print("Confusion matrix (rows = actual, cols = predicted)")
        print("-" * 78)
        print(pd.DataFrame(cm, index=config.CLASS_NAMES,
                           columns=config.CLASS_NAMES).to_string())

        macro_f1 = results["Ensemble (LR)"]["f1_macro"]
        print(f"\nMacro-averaged F1 (paper reports 0.8540): {macro_f1:.4f}")

        print("\n" + "-" * 78)
        print("Meta-learner weights -- contribution of each base model per class")
        print("(Section IV-D: 'The learned weights reflect the contribution of")
        print(" each base model to each class.')")
        print("-" * 78)
        print(weights.round(3).to_string())

    return {
        "predictions": predictions,
        "results": results,
        "table": table,
        "confusion_matrix": cm,
        "per_class": per_class,
        "meta_weights": weights,
    }


def save_evaluation(evaluation, out_dir=None, verbose=True):
    """Write metrics to outputs/metrics/ and figures to outputs/figures/."""
    out_dir = out_dir or config.METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluation["table"].to_csv(out_dir / "table6_model_performance.csv")
    evaluation["per_class"].to_csv(out_dir / "per_class_report.csv")
    evaluation["meta_weights"].to_csv(out_dir / "meta_learner_weights.csv")
    pd.DataFrame(
        evaluation["confusion_matrix"],
        index=config.CLASS_NAMES,
        columns=config.CLASS_NAMES,
    ).to_csv(out_dir / "confusion_matrix.csv")
    M.save_json(evaluation["results"], out_dir / "test_metrics.json")

    plots.plot_confusion_matrix(evaluation["confusion_matrix"])
    plots.plot_model_comparison(evaluation["table"])

    if verbose:
        print(f"Evaluation artifacts saved to {out_dir} and {config.FIGURES_DIR}")


def error_analysis(evaluation, verbose=True) -> pd.DataFrame:
    """
    Section V: "misclassifications occurred most commonly between 'Draw' and
    'Loss' and/or in ties and low-scoring games."

    Tabulates each off-diagonal confusion cell as a share of all errors so the
    claim can be checked against this run.
    """
    cm = evaluation["confusion_matrix"]
    total_errors = cm.sum() - np.trace(cm)

    rows = []
    for i, actual in enumerate(config.CLASS_NAMES):
        for j, predicted in enumerate(config.CLASS_NAMES):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "Actual": actual,
                    "Predicted": predicted,
                    "Count": int(cm[i, j]),
                    "Share of errors (%)": round(100 * cm[i, j] / total_errors, 2),
                })

    df = pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)

    if verbose:
        print("\n" + "-" * 78)
        print(f"Error analysis -- {total_errors:,} misclassifications")
        print("-" * 78)
        print(df.to_string(index=False))

    return df
