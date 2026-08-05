"""
Central configuration for the three-layer ensemble.

Every value in this file is traceable to the paper. Where the paper is silent,
the constant is grouped under `UNSPECIFIED_*` and documented in
`docs/unspecified_details.md`. Nothing here is a silent invention.

Paper: "A Three-Layer Ensemble Model Combining XGBoost, BiLSTM, and CNN for
Football Match Outcome Prediction" -- Paliwal, Tembhurne, Goud.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "combined_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

for _d in (FIGURES_DIR, METRICS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Target definition -- Section III
# --------------------------------------------------------------------------
# "Win: Home team has a higher score than the away team.
#  Loss: Home team has a lower score than the away team.
#  Draw: Both teams have the same score."
CLASS_NAMES = ["Loss", "Draw", "Win"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
N_CLASSES = 3
TARGET_COLUMN = "Result"

# --------------------------------------------------------------------------
# Columns dropped to prevent leakage -- Section III
# --------------------------------------------------------------------------
# "the following identifier and result columns are removed from the dataset:
#  team names, league and year, raw match scores, and goal statistics."
IDENTIFIER_COLUMNS = ["Unnamed: 0", "Home Team", "Away Team", "league", "year"]
SCORE_COLUMNS = ["Score", "Half Time Score"]
GOAL_COLUMNS = [
    "Home Team Goals Scored",
    "Away Team Goals Scored",
    "Home Team Goals Conceeded",  # dataset's own spelling
    "Away Team Goals Conceeded",
]
DROP_COLUMNS = IDENTIFIER_COLUMNS + SCORE_COLUMNS + GOAL_COLUMNS

# Columns needed to derive the target and to order matches chronologically,
# retained during loading and dropped before modelling.
ORDERING_COLUMNS = ["year", "league", "Home Team", "Away Team"]

# --------------------------------------------------------------------------
# Split -- Section III
# --------------------------------------------------------------------------
# "The data was split into training and testing sets across all models, with an
#  80:20 ratio. Stratified sampling was considered here..."
TEST_SIZE = 0.20
STRATIFY = True

# --------------------------------------------------------------------------
# Cross-validation -- Section V-B
# --------------------------------------------------------------------------
# "we implemented 5-fold stratified cross-validation across the dataset."
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Statistical testing -- Section V-C
# --------------------------------------------------------------------------
# "a two-tailed paired t-test and Wilcoxon signed-rank test on the 5-fold
#  accuracy scores... A p-value threshold of 0.01"
SIGNIFICANCE_ALPHA = 0.01

# --------------------------------------------------------------------------
# Layer 1 -- XGBoost (Table II)
# --------------------------------------------------------------------------
XGBOOST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 150,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}

# --------------------------------------------------------------------------
# Layer 1 -- BiLSTM (Table II + Section III)
# --------------------------------------------------------------------------
# Section III: "The scaled tabular data is reshaped into a three-dimensional
# format: (n, 1, d) ... despite the fact that the input window size is one
# (non-sequential/sample)."
#
# NOTE: Table II lists "Sequence Length: 15" for the BiLSTM, which contradicts
# the (n, 1, d) reshape described in prose. The prose is explicit and repeated,
# so BILSTM_TIMESTEPS = 1 follows it. See docs/unspecified_details.md #2.
BILSTM_PARAMS = {
    "hidden_units": 64,
    "timesteps": 1,
    "dropout": 0.3,
    "activation": "tanh",
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy",
    "epochs": 30,
    "batch_size": 32,
}

# --------------------------------------------------------------------------
# Layer 1 -- 1D-CNN (Table II + Section III)
# --------------------------------------------------------------------------
# "a fixed-length sliding window of the last 15 matches before each game.
#  Separate sequences for the home and away teams are maintained. Each sequence
#  is shaped (15, d) ... processed concurrently in the same pipeline using
#  parallel 1D convolutional core components before joining into a common
#  decision layer."
CNN_PARAMS = {
    "window_size": 15,
    "filters": 128,
    "kernel_size": 3,
    "pool_size": 2,
    "dropout": 0.3,
    "activation": "relu",
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "epochs": 30,
    "batch_size": 32,
    "use_class_weights": True,  # Section III: "while training CNN, we computed
                                # class weights and used them in the loss function"
}

# --------------------------------------------------------------------------
# Layer 3 -- Logistic Regression meta-learner (Table II)
# --------------------------------------------------------------------------
META_LEARNER_PARAMS = {
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-4,
}

# Layer 2: 3 base models x 3 classes = 9 meta-features (Eq. 13)
N_META_FEATURES = 3 * N_CLASSES

# --------------------------------------------------------------------------
# Meta-feature generation strategy
# --------------------------------------------------------------------------
# Algorithm 1 trains each base model on D, then computes softmax probabilities
# over X and fits the meta-learner on those. That is in-sample (resubstitution)
# stacking. It is what the paper describes, so it is the default.
#
# Set to True for the leakage-free out-of-fold variant, which is the standard
# stacking protocol (Wolpert, 1992) but is NOT what the paper specifies.
# See docs/unspecified_details.md #5.
USE_OUT_OF_FOLD_META_FEATURES = False
OOF_FOLDS = 5

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# UNSPECIFIED: the paper reports no random seed. Fixed here for reproducibility.
UNSPECIFIED_RANDOM_SEED = 42
RANDOM_SEED = UNSPECIFIED_RANDOM_SEED

# --------------------------------------------------------------------------
# Metric averaging
# --------------------------------------------------------------------------
# The paper reports a macro-averaged F1 of 0.8540 in Section V but a
# "F1" of 87.71% in Table VI. Both averages are therefore computed and
# reported side by side rather than picking one.
# See docs/unspecified_details.md #7.
METRIC_AVERAGES = ("macro", "weighted")
PRIMARY_AVERAGE = "macro"

VERBOSE = 1
