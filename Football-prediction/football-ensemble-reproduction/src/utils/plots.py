"""
Figure reproduction -- Figs. 1, 2, 4 and 5 of the paper.

  Fig. 1  Distribution of match outcomes (bar)
  Fig. 2  Correlation heatmap of match-level numerical features
  Fig. 4  Class distribution of match outcomes (pie)
  Fig. 5  Confusion matrix of the final ensemble model

Fig. 3 is the architecture schematic; `plot_model` is provided to render the
Keras graphs, but the schematic itself is a hand-drawn diagram in the paper and
is not reproduced programmatically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import config

sns.set_theme(style="whitegrid")


def _save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution_bar(y_labels, path=None):
    """Fig. 1 -- Distribution of Match Outcomes (Win, Draw, Loss)."""
    path = path or config.FIGURES_DIR / "fig1_class_distribution_bar.png"
    counts = pd.Series(y_labels).value_counts().reindex(["Loss", "Draw", "Win"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(counts.index, counts.values,
                  color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 40, f"{value:,}",
                ha="center", fontsize=9)

    ax.set_xlabel("Result")
    ax.set_ylabel("Number of Matches")
    ax.set_title("Class Distribution of Match Outcomes")
    return _save(fig, path)


def plot_class_distribution_pie(y_labels, path=None):
    """Fig. 4 -- Class distribution of football match outcomes."""
    path = path or config.FIGURES_DIR / "fig4_class_distribution_pie.png"
    counts = pd.Series(y_labels).value_counts().reindex(["Draw", "Loss", "Win"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=["#d62728", "#ffbf00", "#2ca02c"], startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Class Distribution of Match Outcomes")
    return _save(fig, path)


def plot_correlation_heatmap(df, feature_columns, path=None):
    """Fig. 2 -- Correlation Heatmap of Match-Level Numerical Features."""
    path = path or config.FIGURES_DIR / "fig2_correlation_heatmap.png"
    corr = df[feature_columns].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"label": "Correlation", "shrink": 0.8},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.tick_params(labelsize=6)
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=12)
    return _save(fig, path)


def plot_confusion_matrix(cm, path=None, title="Confusion Matrix of Final Ensemble Model",
                          normalize=False):
    """Fig. 5 -- Confusion matrix for the final ensemble model."""
    path = path or config.FIGURES_DIR / "fig5_confusion_matrix.png"
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(data, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", cbar_kws={"label": "Count"},
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted Outcome")
    ax.set_ylabel("Actual Outcome")
    ax.set_title(title)
    return _save(fig, path)


def plot_model_comparison(results_df, path=None):
    """Grouped bar chart of the Table VI comparison."""
    path = path or config.FIGURES_DIR / "model_comparison.png"
    cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]

    fig, ax = plt.subplots(figsize=(10, 5))
    (results_df[cols] * 100).plot.bar(ax=ax, width=0.78)
    ax.set_ylabel("Score (%)")
    ax.set_title("Base Learners vs. Three-Layer Ensemble")
    ax.legend(loc="lower right", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, path)


def plot_cv_results(cv_df, path=None):
    """Mean +/- std accuracy across the 5 stratified folds (Table III)."""
    path = path or config.FIGURES_DIR / "cross_validation_accuracy.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cv_df.index, cv_df["accuracy_mean"] * 100,
           yerr=cv_df["accuracy_std"] * 100, capsize=6, color="#4c72b0")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5-Fold Stratified Cross-Validation Accuracy")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, path)


def plot_training_history(history, name, path=None):
    """Keras loss/accuracy curves. Not a paper figure; useful for diagnostics."""
    path = path or config.FIGURES_DIR / f"training_history_{name}.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"])
    axes[0].set_title(f"{name} -- training loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(history.history.get("accuracy", []))
    axes[1].set_title(f"{name} -- training accuracy")
    axes[1].set_xlabel("Epoch")
    return _save(fig, path)
