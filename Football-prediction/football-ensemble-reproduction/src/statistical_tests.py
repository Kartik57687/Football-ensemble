"""
Statistical significance testing -- Section V-C, Table IV.

    "we conducted a two-tailed paired t-test and Wilcoxon signed-rank test on
     the 5-fold accuracy scores comparing the ensemble model against each base
     learner. A p-value threshold of 0.01 was selected to ensure strong
     statistical significance."

A caveat worth stating plainly: with only 5 paired observations, the Wilcoxon
signed-rank test has a minimum attainable two-sided p-value of 0.0625. The
p-values below 0.01 that the paper reports in Table IV are therefore not
attainable from a 5-sample Wilcoxon test under the exact null distribution.
SciPy's default `mode='auto'` uses the exact distribution at n=5, so the values
this module produces will differ from the paper's on that test. The normal
approximation is also reported for completeness, since that is the only route to
smaller p-values at this sample size. See docs/unspecified_details.md #10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def _wilcoxon(a, b, how: str, **kwargs):
    """
    Wilcoxon signed-rank test, tolerant of the SciPy API rename.

    SciPy renamed the `mode` keyword to `method` (`mode` still works but is
    deprecated). Both spellings are tried so the module runs unchanged on old and
    new SciPy alike.
    """
    try:
        return stats.wilcoxon(a, b, alternative="two-sided", method=how, **kwargs)
    except TypeError:
        return stats.wilcoxon(a, b, alternative="two-sided", mode=how, **kwargs)


def paired_tests(per_fold: pd.DataFrame, metric: str = "accuracy", verbose=True):
    """
    Compare the ensemble against each base learner on per-fold scores.

    Parameters
    ----------
    per_fold : DataFrame
        Long-form output of cross_validation.run_cross_validation.
    """
    pivot = per_fold.pivot(index="fold", columns="model", values=metric)
    ensemble = pivot["Ensemble (LR)"].to_numpy()

    rows = []
    for base in ("XGBoost", "BiLSTM", "CNN"):
        base_scores = pivot[base].to_numpy()
        diff = ensemble - base_scores

        t_stat, t_p = stats.ttest_rel(ensemble, base_scores)

        if np.allclose(diff, 0):
            w_stat, w_p_exact, w_p_approx = np.nan, 1.0, 1.0
        else:
            w_stat, w_p_exact = _wilcoxon(ensemble, base_scores, "exact")
            _, w_p_approx = _wilcoxon(ensemble, base_scores, "approx",
                                      correction=False)

        rows.append({
            "Compared Models": f"Ensemble vs {base}",
            "Mean difference": float(diff.mean()),
            "t-statistic": float(t_stat),
            "t-test (p)": float(t_p),
            "Wilcoxon (p, exact)": float(w_p_exact),
            "Wilcoxon (p, normal approx.)": float(w_p_approx),
            f"Significant at alpha={config.SIGNIFICANCE_ALPHA}":
                bool(t_p < config.SIGNIFICANCE_ALPHA),
        })

    df = pd.DataFrame(rows)

    if verbose:
        n = len(pivot)
        print("\n" + "=" * 78)
        print(f"TABLE IV -- Paired Statistical Tests on {metric} (n = {n} folds)")
        print("=" * 78)
        print(df.round(5).to_string(index=False))
        if n <= 5:
            print(f"\nNote: with n = {n} paired folds, the smallest two-sided p-value")
            print("attainable by an exact Wilcoxon signed-rank test is 0.0625.")
            print("Exact and normal-approximation p-values are both shown above.")

    return df


def save_statistical_tests(df, out_dir=None, verbose=True):
    """Persist Table IV."""
    out_dir = out_dir or config.METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "table4_statistical_tests.csv", index=False)
    if verbose:
        print(f"Statistical tests saved to {out_dir}")
