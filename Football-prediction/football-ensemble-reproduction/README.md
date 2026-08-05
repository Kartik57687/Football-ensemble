# Three-Layer Ensemble for Football Match Outcome Prediction

A faithful reproduction of **"A Three-Layer Ensemble Model Combining XGBoost,
BiLSTM, and CNN for Football Match Outcome Prediction"** by Kratik Paliwal,
Jitendra Tembhurne and Harsh Goud.

The methodology is implemented exactly as published — no improvements, no
substitutions, no silent fixes. Where the paper is ambiguous or contradicts
itself, the choice made here is documented in
[`docs/unspecified_details.md`](docs/unspecified_details.md) rather than buried
in code.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              LAYER 1: BASE MODELS           │
                    ├─────────────┬─────────────┬─────────────────┤
   match features → │   XGBoost   │   BiLSTM    │     1D-CNN      │
                    │ (structured)│ (sequential)│   (temporal)    │
                    └──────┬──────┴──────┬──────┴────────┬────────┘
                        softmax       softmax         softmax
                           └─────────────┼────────────────┘
                    ┌────────────────────▼────────────────────────┐
                    │  LAYER 2: META-FEATURE VECTOR  x_meta ∈ ℝ⁹  │
                    │        (concatenated probabilities)         │
                    └────────────────────┬────────────────────────┘
                    ┌────────────────────▼────────────────────────┐
                    │  LAYER 3: LOGISTIC REGRESSION META-LEARNER  │
                    └────────────────────┬────────────────────────┘
                                 Win / Draw / Loss
```

| Layer | Component | Paper reference |
|---|---|---|
| 1 | XGBoost, BiLSTM, 1D-CNN trained independently | §IV-A |
| 2 | Softmax outputs concatenated into `x_meta` ∈ ℝ⁹ | §IV-B, Eqs. 12–13 |
| 3 | Multinomial logistic regression | §IV-C, Eq. 14 |

---

## Quick start

### Google Colab

1. Push this folder to GitHub.
2. Open `notebooks/football_ensemble_reproduction.ipynb` in Colab.
3. Uncomment the clone cell at the top, insert your repo URL, and run all.

The dataset ships with the repo, so nothing needs uploading. CPU is sufficient —
the paper itself reports CPU-only training.

### Local

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt

python run_pipeline.py                # full reproduction (~25 min)
python run_pipeline.py --skip-cv      # training + evaluation only (~5 min)
python run_pipeline.py --quick        # 3-epoch smoke test (~1 min)
```

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── run_pipeline.py                 # CLI entry point — runs everything
│
├── data/
│   ├── raw/combined_data.csv       # the paper's named input file
│   └── README.md                   # dataset provenance and schema
│
├── notebooks/
│   └── football_ensemble_reproduction.ipynb   # Colab-ready walkthrough
│
├── src/
│   ├── config.py                   # every hyperparameter, traced to the paper
│   ├── data_loader.py              # loading, target construction, ordering
│   ├── preprocessing.py            # imputation, scaling, stratified split
│   ├── sequence_builder.py         # per-team 15-match windows for the CNN
│   ├── train.py                    # Algorithm 1, end to end
│   ├── evaluate.py                 # Table VI, Fig. 5, error analysis
│   ├── cross_validation.py         # Table III
│   ├── statistical_tests.py        # Table IV
│   ├── ablation.py                 # Table V
│   ├── models/
│   │   ├── xgboost_model.py        # Layer 1
│   │   ├── bilstm_model.py         # Layer 1
│   │   ├── cnn_model.py            # Layer 1 (dual-branch)
│   │   └── meta_learner.py         # Layers 2 and 3
│   └── utils/
│       ├── seed.py, metrics.py, plots.py
│
├── outputs/
│   ├── figures/                    # Figs. 1, 2, 4, 5 + comparisons
│   ├── metrics/                    # Tables III–VI as CSV/JSON
│   └── models/                     # serialised fitted models
│
└── docs/
    ├── unspecified_details.md      # ← read this before trusting any number
    └── paper_to_code_mapping.md    # every paper claim → its code location
```

---

## Dataset

The paper names its input file explicitly:

> *"The file containing the league data to obtain a unified version called
> `combined_data.csv`, where the individual league data are combined, and thus we
> have a dataset that consists of approximately 13,000 matches."*

`combined_data.csv` is therefore the file used. It is the already-merged
five-league file; no per-league merging step is required. You supplied one data
file, and it is the one the paper names — no selection between candidates was
needed.

**Verified against the paper:**

| Property | Paper | This file |
|---|---|---|
| Matches | ~13,000 | 12,062 |
| Leagues | Top 5 European | English, Spanish, Italian, French, German |
| Seasons | multiple | 2014–2020 |
| Win share | 44.5% (Fig. 4) | **44.5%** |
| Loss share | 30.5% (Fig. 4) | **30.5%** |
| Draw share | 25.0% (Fig. 4) | **25.0%** |
| Numerical features | ~30 | 31 |

The class proportions match Fig. 4 to the decimal, which confirms this is the
correct file. See `data/README.md` for the full schema.

---

## Three things to know before reading the results

These are properties of the published method, surfaced here rather than fixed.
Full detail in [`docs/unspecified_details.md`](docs/unspecified_details.md).

**1. Post-match features survive the paper's drop list (#11).**
Section III drops *"team names, league and year, raw match scores, and goal
statistics."* That list does not cover `Match Excitement`, `Home Team Rating` or
`Away Team Rating` — all three are assigned by the data provider *after* a match,
informed by how it went. They carry result information no pre-match model could
have, and are the most plausible explanation for accuracy near 89% against a
literature baseline of 70–81%. They are retained, because the paper's drop list
is explicit. The notebook includes a one-cell sensitivity check that removes
them.

**2. Algorithm 1 uses in-sample meta-features (#5).**
The base models are trained on `D`, then produce probabilities for that same `D`,
and the meta-learner is fitted on those. The meta-learner therefore sees
probabilities the base models have partly memorised, which inflates the apparent
benefit of Layer 3. Implemented as written; set
`config.USE_OUT_OF_FOLD_META_FEATURES = True` for the standard leakage-free
protocol.

**3. Table II and Section III disagree on the BiLSTM (#2).**
Table II lists `Sequence Length: 15`; Section III says the input is reshaped to
`(n, 1, d)` *"despite the fact that the input window size is one."* The prose is
followed. One consequence worth naming: with a single timestep, the
bidirectionality does nothing, and the BiLSTM reduces to a gated feed-forward
layer.

---

## Reproduction results

Full write-up in [`docs/results.md`](docs/results.md). Headline figures from
`python run_pipeline.py` at seed 42:

| Model | Accuracy (this repo) | Accuracy (paper) |
|---|---|---|
| XGBoost | 89.97% | 84.77% |
| BiLSTM | 88.31% | 88.16% |
| 1D-CNN | 37.84% | 46.68% |
| **Ensemble (LR)** | **89.64%** | **88.78%** |

The headline ensemble number reproduces within ~0.9 points, and nine of the
paper's eleven checkable claims hold. **Two do not**, and both are worth knowing
about:

- **The ensemble does not beat XGBoost.** Across 5 folds it is 0.58 points
  *worse*, not significant at the paper's α = 0.01. It does beat the BiLSTM and
  CNN significantly, so Layer 3 works — it just cannot exceed its strongest
  member here.
- **The ablation ordering inverts.** The paper finds the BiLSTM most important;
  here removing XGBoost costs 2.2 points while removing the BiLSTM costs nothing
  measurable. That follows from unspecified detail #2 — under the paper's own
  prose the BiLSTM gets a single timestep, so it has no temporal structure to
  contribute.

Neither is an implementation defect; both trace to the underspecified components.
`docs/results.md` sets out the evidence.

---

## What gets reproduced

| Paper artifact | Module | Output |
|---|---|---|
| Fig. 1 — outcome distribution (bar) | `utils/plots.py` | `outputs/figures/fig1_*.png` |
| Fig. 2 — correlation heatmap | `utils/plots.py` | `outputs/figures/fig2_*.png` |
| Fig. 4 — outcome distribution (pie) | `utils/plots.py` | `outputs/figures/fig4_*.png` |
| Fig. 5 — ensemble confusion matrix | `evaluate.py` | `outputs/figures/fig5_*.png` |
| Table III — 5-fold CV | `cross_validation.py` | `outputs/metrics/table3_*.csv` |
| Table IV — significance tests | `statistical_tests.py` | `outputs/metrics/table4_*.csv` |
| Table V — ablation study | `ablation.py` | `outputs/metrics/table5_*.csv` |
| Table VI — per-model performance | `evaluate.py` | `outputs/metrics/table6_*.csv` |
| Algorithm 1 — three-layer training | `train.py` | `outputs/models/` |
| Eqs. 1–3 — XGBoost objective | `models/xgboost_model.py` | — |
| Eqs. 4–9 — BiLSTM gates | `models/bilstm_model.py` | — |
| Eqs. 10–11 — Conv1D / pooling | `models/cnn_model.py` | — |
| Eqs. 12–14 — fusion + softmax LR | `models/meta_learner.py` | — |

---

## Configuration

All hyperparameters live in `src/config.py`, each annotated with its source in
the paper. Nothing is hard-coded elsewhere.

```python
XGBOOST_PARAMS   = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 150,
                    "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 1.0}
BILSTM_PARAMS    = {"hidden_units": 64, "dropout": 0.3, "activation": "tanh",
                    "epochs": 30, "batch_size": 32}
CNN_PARAMS       = {"filters": 128, "kernel_size": 3, "pool_size": 2,
                    "dropout": 0.3, "window_size": 15, "epochs": 30}
META_LEARNER_PARAMS = {"penalty": "l2", "solver": "lbfgs",
                       "max_iter": 1000, "tol": 1e-4}
```

Useful switches:

| Setting | Default | Effect |
|---|---|---|
| `RANDOM_SEED` | `42` | Not specified by the paper; fixed for reproducibility |
| `USE_OUT_OF_FOLD_META_FEATURES` | `False` | `True` gives leakage-free stacking (not the paper's protocol) |
| `PRIMARY_AVERAGE` | `"macro"` | Averaging for headline precision/recall/F1 |

---

## Reproducibility

The paper reports **no random seed**, so its exact numbers cannot be reproduced
even by a perfect implementation — the split, the neural initialisations and
XGBoost's subsampling are all stochastic. This repo fixes a seed so that *its own*
runs are reproducible, and runs CPU-only, matching the paper's reported setup
(*"no GPU acceleration is utilized"*). Expect run-to-run variation of a few
tenths of a percent when the seed changes.

---

## Requirements

Python 3.9+. See `requirements.txt`. Colab's default environment already
satisfies everything except pinned versions.

---

## Citation

```bibtex
@article{paliwal2025threelayer,
  title  = {A Three-Layer Ensemble Model Combining XGBoost, BiLSTM, and CNN
            for Football Match Outcome Prediction},
  author = {Paliwal, Kratik and Tembhurne, Jitendra and Goud, Harsh},
  year   = {2025}
}
```

Dataset: S. S. Naik, *Football Data Top 5 Leagues*, Kaggle, 2023.
