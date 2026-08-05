# Paper → code traceability map

Every methodological claim in the paper and where it is implemented. Use this to
verify that nothing was skipped, substituted, or quietly modified.

---

## Section III — Dataset and Preprocessing

| Paper statement | Implementation | Location |
|---|---|---|
| Five top European leagues, `combined_data.csv`, ~13,000 matches | Loads that file (12,062 matches; see unspecified #12) | `data_loader.load_raw` |
| ~30 numerical per-match metrics | 31 features survive the drop list | `data_loader.get_feature_columns` |
| Win / Loss / Draw from home-team score comparison | `np.select` on goals scored | `data_loader.build_target` |
| Computed column named `Result` | `config.TARGET_COLUMN = "Result"` | `config.py` |
| Drop team names, league, year, raw scores, goal statistics | 11 columns dropped | `config.DROP_COLUMNS` |
| Median imputation of missing/anomalous values | `fillna(median)` on features | `preprocessing.impute_missing` |
| StandardScaler for XGBoost and BiLSTM | Fitted on train split | `preprocessing.fit_standard_scaler` |
| Min-Max for CNN sequence construction | Fitted on train split | `preprocessing.fit_minmax_scaler` |
| XGBoost: structured features entered directly in tabular form | `(n, d)` matrix | `train.prepare_data` |
| BiLSTM: reshape to `(n, 1, d)` | `reshape_for_bilstm` | `preprocessing.py` |
| CNN: sliding window of last 15 matches, per team | Chronological per-team history | `sequence_builder.build_team_sequences` |
| CNN: separate home and away sequences, each `(15, d)` | Two tensors per match | `sequence_builder.build_team_sequences` |
| CNN: parallel conv components joined at a common decision layer | Two-branch functional model | `models/cnn_model.build_cnn` |
| 80:20 split, stratified sampling | `train_test_split(stratify=y)` | `preprocessing.stratified_split` |
| Class weights computed and used in the CNN loss | `compute_class_weight("balanced")`, CNN only | `preprocessing.compute_cnn_class_weights` |

### Section III-A — Exploratory Data Analysis

| Paper artifact | Location |
|---|---|
| Fig. 1 — outcome distribution (bar) | `utils/plots.plot_class_distribution_bar` |
| Fig. 2 — correlation heatmap | `utils/plots.plot_correlation_heatmap` |
| Fig. 4 — outcome distribution (pie) | `utils/plots.plot_class_distribution_pie` |
| Observed imbalance: Win > Loss > Draw | Verified: 44.5% / 30.5% / 25.0% | — |

---

## Section IV — Proposed Methodology

### IV-A — Layer 1: Base Learners

| Component | Location |
|---|---|
| XGBoost base learner | `models/xgboost_model.py` |
| BiLSTM base learner | `models/bilstm_model.py` |
| 1D-CNN base learner | `models/cnn_model.py` |
| Independent training on the same split | `train.train_base_learners` |

### IV-B — Layer 2: Meta-Feature Construction

| Paper statement | Location |
|---|---|
| Use softmax distributions, not hard predictions | `train.base_probabilities` |
| `x_meta = [p⁽¹⁾, p⁽²⁾, p⁽³⁾] ∈ ℝ⁹` (Eq. 13) | `models/meta_learner.build_meta_features` |
| `X_meta ∈ ℝ^{n×9}` | Shape asserted at construction | `models/meta_learner.py` |

### IV-C — Layer 3: Meta-Learner

| Paper statement | Location |
|---|---|
| Multinomial logistic regression on `X_meta` | `models/meta_learner.build_meta_learner` |
| Softmax probability function (Eq. 14) | `predict_proba_meta` |
| Interpretable per-class weights (§IV-D) | `models/meta_learner.meta_learner_weights` |

### IV-E — Mathematical Formulation

| Equations | Realisation |
|---|---|
| 1–3, XGBoost regularised objective + Taylor expansion | `multi:softprob` + `reg_lambda` (λ of Eq. 2) |
| 4–9, LSTM gates and cell state | `keras.layers.LSTM` (identical formulation) |
| 10–11, 1D convolution and pooling | `Conv1D` + `MaxPooling1D` |
| 12–13, softmax outputs and concatenation | `build_meta_features` |
| 14, multinomial softmax | `LogisticRegression` |

### Algorithm 1 — Three-Layer Ensemble Prediction

| Algorithm line | Location |
|---|---|
| 1–5 Train base models | `train.train_base_learners` |
| 6–10 Meta-feature extraction | `train.base_probabilities` + `build_meta_features` |
| 11–12 Train meta-learner | `models/meta_learner.train_meta_learner` |
| 13–17 Predict | `evaluate.evaluate_ensemble` |

---

## Section V — Results and Evaluation

| Paper artifact | Location |
|---|---|
| Table II — hyperparameters | `config.py` (verbatim) |
| Table III — 5-fold stratified CV, mean ± std | `cross_validation.run_cross_validation` |
| Table IV — paired t-test and Wilcoxon, α = 0.01 | `statistical_tests.paired_tests` |
| Table V — ablation, one model removed at a time | `ablation.run_ablation` |
| Table VI — per-model test performance | `evaluate.evaluate_ensemble` |
| Fig. 5 — ensemble confusion matrix | `utils/plots.plot_confusion_matrix` |
| Precision / Recall / F1 formulae | `utils/metrics.compute_metrics` |
| Macro-averaged F1 (0.8540 quoted) | `compute_all_metrics["f1_macro"]` |
| Error analysis (Draw↔Loss confusion) | `evaluate.error_analysis` |
| Categorical cross-entropy justification | `CNN_PARAMS["loss"]`, `BILSTM_PARAMS["loss"]` |
| Focal Loss explicitly *rejected* by the paper | Not implemented — correctly so |

---

## Deliberately not implemented

The paper mentions these as **future work**, not as part of the method. They are
absent from this repo by design:

- Attention mechanisms / transformer encoders (§VI)
- SHAP or LIME explainability (§VI)
- Focal Loss and SMOTE — explicitly considered and rejected in §V
- Player-level statistics, betting odds, live tracking data (§VI)
- Graph neural networks (§II, noted as requiring unavailable tracking data)

Fig. 3 is a hand-drawn architecture schematic and is not reproduced
programmatically; `keras.utils.plot_model` will render the equivalent graphs from
`build_bilstm` and `build_cnn`.
