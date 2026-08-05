# Reproduction results

Produced by `python run_pipeline.py` with `RANDOM_SEED = 42`, CPU-only, 30 epochs
for both neural base learners — the paper's stated configuration.

Reminder on comparability: the paper reports no random seed, so its exact figures
are not reproducible even by a perfect implementation. The split, the neural
initialisations and XGBoost's 0.8 subsampling are all stochastic. What follows
should be read as "does the implementation land in the same place", not "do the
decimals match".

---

## Table VI — Held-out test set (2,413 matches)

| Model | Accuracy (ours) | Accuracy (paper) | Δ | F1 macro (ours) | F1 (paper) |
|---|---|---|---|---|---|
| XGBoost | **89.97%** | 84.77% | +5.20 | 88.72% | 83.74% |
| BiLSTM | **88.31%** | 88.16% | +0.15 | 86.69% | 87.12% |
| 1D-CNN | **37.84%** | 46.68% | −8.84 | 37.27% | 43.89% |
| **Ensemble (LR)** | **89.64%** | **88.78%** | **+0.86** | **88.45%** | 87.71% |

The headline ensemble figure reproduces within ~0.9 points. BiLSTM matches almost
exactly. XGBoost comes out materially stronger here and the CNN materially
weaker — both discussed below.

### Per-class breakdown, final ensemble

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Loss | 0.918 | 0.909 | 0.913 | 735 |
| Draw | 0.802 | 0.798 | 0.800 | 603 |
| Win | 0.935 | 0.943 | 0.939 | 1,075 |

Macro F1 = **0.8845** (paper quotes 0.8540). Draw is the weakest class by a clear
margin, exactly as the paper reports — and for the reason it gives, that Draw is
the minority outcome at 25%.

### Confusion matrix

| Actual ↓ / Predicted → | Loss | Draw | Win |
|---|---|---|---|
| **Loss** | 668 | 59 | 8 |
| **Draw** | 55 | 484 | 64 |
| **Win** | 1 | 63 | 1,011 |

Note the cell counts are ~22× the paper's Fig. 5, which sums to only 107 matches
— see `unspecified_details.md` #13.

### Error analysis

Section V predicts that *"misclassifications occurred most commonly between
'Draw' and 'Loss'"*. Confirmed, and it generalises: **96.0%** of all 250 errors
involve a Draw on one side or the other.

| Actual | Predicted | Count | Share of errors |
|---|---|---|---|
| Draw | Win | 64 | 25.6% |
| Win | Draw | 63 | 25.2% |
| Loss | Draw | 58 | 23.2% |
| Draw | Loss | 55 | 22.0% |
| Loss | Win | 9 | 3.6% |
| Win | Loss | 1 | 0.4% |

Win↔Loss confusion is nearly absent (10 of 250). The model separates the decisive
outcomes cleanly and struggles only at the Draw boundary.

---

## Table III — 5-fold stratified cross-validation

| Model | Accuracy (ours) | Accuracy (paper) | F1 (ours) | F1 (paper) |
|---|---|---|---|---|
| XGBoost | 90.23% ± 0.45 | 84.25% ± 0.81 | 88.96% ± 0.50 | 83.25% ± 0.79 |
| BiLSTM | 87.78% ± 0.67 | 87.68% ± 0.52 | 86.17% ± 0.70 | 86.81% ± 0.55 |
| 1D-CNN | 39.43% ± 1.34 | 45.95% ± 1.10 | 37.60% ± 0.49 | 43.41% ± 1.21 |
| Ensemble (LR) | 89.65% ± 0.67 | 88.32% ± 0.43 | 88.39% ± 0.72 | 87.60% ± 0.42 |

Fold-to-fold standard deviations are comparable to the paper's, so the stability
claim holds.

---

## Table IV — Statistical significance (n = 5 folds)

| Comparison | Mean diff. | t (p) | Wilcoxon exact (p) | Wilcoxon approx. (p) | Sig. at α=0.01 |
|---|---|---|---|---|---|
| Ensemble vs XGBoost | **−0.0058** | 0.0160 | 0.0625 | 0.0431 | **No** |
| Ensemble vs BiLSTM | +0.0187 | 0.00053 | 0.0625 | 0.0431 | Yes |
| Ensemble vs CNN | +0.5022 | <0.00001 | 0.0625 | 0.0431 | Yes |

### The one result that does not reproduce

The paper's central claim is that the ensemble *"outperformed all base models
individually"*, with Table IV reporting Ensemble vs XGBoost significant at
p = 0.0041.

**In this reproduction the ensemble does not beat XGBoost — it is 0.58 points
worse on average across the five folds**, and the difference is not significant
at the paper's own α = 0.01 threshold. The ensemble does beat the BiLSTM and CNN
significantly, so Layer 3 is doing real work; it just cannot exceed its strongest
member here.

This is not a bug in the implementation. It follows from the base-learner
results: XGBoost lands ~5 points above the paper's figure while the CNN lands
~9 points below. A stacked ensemble is bounded in practice by how much its
members disagree usefully. When one member is far stronger than the others and a
third is barely above chance, the meta-learner's best available strategy is to
lean heavily on the strong member — which is precisely what the learned weights
show (`outputs/metrics/meta_learner_weights.csv`): the XGBoost columns carry
coefficients of magnitude 4.7–6.1, against 0.2–1.5 for the BiLSTM.

Two candidate explanations, neither of which I can settle from the paper alone:

1. **The CNN.** The paper's CNN reaches 46.68%; this one reaches 37.84%. The CNN
   is the component whose specification is thinnest — sequence ordering
   (`unspecified_details.md` #4), short-history padding (#6) and network depth
   (#9) are all underspecified. A better-specified CNN would contribute more
   diversity and could plausibly push the ensemble above XGBoost.
2. **XGBoost being unusually strong.** At 90% it is close to the ceiling the
   post-match rating features permit (#11), leaving little headroom above it for
   any ensemble to occupy.

---

## Table V — Ablation study

| Variant | Meta-features | Accuracy | F1 macro | Δ vs full |
|---|---|---|---|---|
| Full Ensemble (All) | 9 | 0.8964 | 0.8845 | — |
| w/o CNN (XGB+LSTM) | 6 | 0.8939 | 0.8809 | −0.0025 |
| w/o BiLSTM (XGB+CNN) | 6 | 0.8964 | 0.8846 | +0.0000 |
| w/o XGBoost (LSTM+CNN) | 6 | 0.8740 | 0.8574 | **−0.0224** |

Paper's Table V for comparison:

| Variant | Accuracy (paper) |
|---|---|
| Full Ensemble (All) | 0.8906 |
| w/o CNN | 0.8898 |
| w/o BiLSTM | 0.8643 |
| w/o XGBoost | 0.8838 |

**The ordering differs, and it matters.** The paper concludes that removing the
BiLSTM hurts most, *"confirming its importance for capturing temporal
dependencies."* Here, removing XGBoost hurts by far the most (−2.2 points), while
removing the BiLSTM costs nothing measurable.

That divergence is consistent with the significance-test result above and with
`unspecified_details.md` #2: under the paper's own prose, the BiLSTM receives a
single-timestep input, so it has no temporal dependencies to capture. It is a
gated feed-forward model over the same tabular features XGBoost already handles
better. Its contribution being redundant is the expected outcome of that
architecture, not an anomaly.

---

## Summary of what reproduced

| Claim | Status |
|---|---|
| Ensemble accuracy ≈ 88.8% | ✅ 89.6% |
| Ensemble beats BiLSTM significantly | ✅ p = 0.0005 |
| Ensemble beats CNN significantly | ✅ p < 0.00001 |
| BiLSTM ≈ 88% accuracy | ✅ 88.3% |
| CNN is the weakest base learner | ✅ 37.8% |
| Draw is the hardest class | ✅ F1 0.80 vs 0.91/0.94 |
| Errors concentrate around Draw | ✅ 96% of errors |
| Class distribution matches Fig. 4 | ✅ exact |
| Fold-level stability | ✅ std ≤ 1.3% |
| **Ensemble beats XGBoost significantly** | ❌ ensemble is 0.58 pts *worse*, n.s. |
| **BiLSTM is the most important member** | ❌ XGBoost is, by 9× |

Nine of eleven claims reproduce. The two that do not are the two that depend on
the underspecified components, and both failures point the same direction —
toward the CNN and BiLSTM specifications rather than toward the ensembling
machinery itself.
