# Implementation details the paper does not specify

You asked that unspecified details be flagged rather than guessed. This file is
that list. Each entry states what the paper says, what it omits, what this
implementation does, and where the choice lives in code.

Nothing here is a change to the methodology. Every choice below is either forced
(the paper is silent and *something* must be chosen to run the code) or is a
faithful reading of a passage that admits more than one reading. Where the paper
contradicts itself, both readings are described and the one implemented is
justified.

---

## 1. Random seed

**Paper:** No seed, and no seeding procedure, is reported anywhere.

**Consequence:** Exact reproduction of the paper's specific numbers is not
possible even given a perfect implementation. The BiLSTM and CNN both have
random weight initialisation, the train/test split is random, and XGBoost
subsamples at 0.8.

**Implemented:** `RANDOM_SEED = 42` in `src/config.py`, applied to Python,
NumPy, TensorFlow and scikit-learn via `src/utils/seed.py`. Overridable with
`--seed`.

---

## 2. BiLSTM sequence length — the paper contradicts itself

**Paper, Section III (prose):**

> "The scaled tabular data is reshaped into a three-dimensional format:
> `(n, 1, d)` ... This three-dimensional shape is appropriate for time series
> input into a Bidirectional LSTM, **despite the fact that the input window size
> is one (non-sequential/sample)**."

**Paper, Table II:** `BiLSTM — Sequence Length: 15`

These cannot both hold. A `(n, 1, d)` tensor has one timestep, not fifteen.

**Implemented:** `timesteps = 1`, following the prose. The prose is explicit,
appears in the methodology section rather than a summary table, and explains
*why* the choice was made — it reads as deliberate, whereas the Table II entry
reads as copied from the CNN row (which legitimately uses 15).

**Worth knowing:** with one timestep, the "Bi" in BiLSTM does no work. Forward
and backward passes see the same single vector, so the layer degenerates to a
gated feed-forward transform. This is consistent with the paper's own BiLSTM
accuracy (88.16%) sitting close to XGBoost's — a genuinely sequential model
would be expected to diverge further.

**To try the other reading:** set `BILSTM_PARAMS["timesteps"] = 15` in
`src/config.py` and feed it the CNN's home-team sequence tensor.

---

## 3. Where the scalers are fitted

**Paper:** Names `StandardScaler` (XGBoost, BiLSTM) and Min-Max normalisation
(CNN) but does not say whether they are fitted on the full dataset or on the
training split only.

**Implemented:** Fitted on the training split only, applied to both splits
(`src/preprocessing.py`). Fitting on the full dataset would leak test-set means
and variances into training. Inside cross-validation, scalers are refitted per
fold on that fold's training portion.

---

## 4. Chronological ordering of matches

**Paper:** The CNN uses "the last 15 matches before each game", which requires a
match ordering.

**Dataset:** `combined_data.csv` has a `year` column but **no match date or
matchday column**. Ordering within a season is not recoverable from the file.

**Implemented:** Row order within each `(year, league)` block is used as the
chronological proxy (`data_loader.add_chronological_order`). This is the only
ordering information the file carries.

**Consequence:** If the CSV's row order does not reflect fixture order, the CNN
windows are not true match histories. This is a limitation of the dataset, not
of the implementation, and it plausibly contributes to the CNN's poor standalone
accuracy — which the paper itself reports at 46.68%.

---

## 5. In-sample vs. out-of-fold meta-features

**Paper, Algorithm 1:**

```
1: Layer 1: Train Base Models        (on D)
6: Layer 2: Meta-Feature Extraction  (for each match x in X)
11: Layer 3: Train Meta-Learner      (on {x_meta, y})
```

As written, the base models are trained on `D`, then produce probabilities for
the same data, and the meta-learner is fitted on those. That is **in-sample**
(resubstitution) stacking.

**Implemented:** In-sample, per Algorithm 1. `USE_OUT_OF_FOLD_META_FEATURES =
False` in `src/config.py`.

**The concern, stated once:** in-sample stacking shows the meta-learner
probabilities that the base models have already partly memorised. The
meta-learner then learns to trust whichever base model overfits hardest, and the
measured Layer 3 gain is optimistic. The standard fix — out-of-fold
meta-features — is Wolpert (1992), which the paper cites as reference [5] while
not applying it here.

**Both paths are implemented.** Flip the flag to `True` for the out-of-fold
variant. The paper-faithful path remains the default, because the brief is to
reproduce the paper.

---

## 6. Teams with fewer than 15 prior matches

**Paper:** Specifies a 15-match window but not what happens at the start of a
team's record, where fewer than 15 prior matches exist. Zero-padding is
mentioned for the BiLSTM ("zero-padded (for BiLSTM)") but not for the CNN.

**Implemented:** Front (left) zero-padding, so the most recent prior match always
occupies the final timestep (`src/sequence_builder.py`). No rows are dropped.

**Why not drop them:** Layer 2 concatenates the three base models' outputs
row-by-row (Eq. 13). Dropping short-history matches from the CNN only would
desynchronise it from XGBoost and the BiLSTM and make the concatenation
impossible. The pipeline prints the padded-window count so the effect is visible.

---

## 7. Metric averaging scheme

**Paper:** Reports "Precision", "Recall", "F1" in Tables III, V and VI without
naming an averaging scheme, then separately quotes a **macro-averaged** F1 of
0.8540 in the Fig. 5 discussion. Table VI's F1 of 87.71% does not equal 0.8540,
so the two are not the same quantity.

**Implemented:** Both macro and weighted averages are computed and reported side
by side (`src/utils/metrics.py`), plus a full per-class breakdown. Macro is the
primary figure, since it is the only averaging the paper names explicitly.

---

## 8. BiLSTM architecture beyond the three stated values

**Paper:** Gives `Hidden Units: 64`, `Dropout: 0.3`, `Activation: tanh`. It does
not state the number of recurrent layers, whether any dense layer precedes the
output, or the output activation.

**Implemented:** The minimal architecture consistent with those three values —
`Bidirectional(LSTM(64, tanh)) → Dropout(0.3) → Dense(3, softmax)`. Softmax is
forced by Section IV-B, which requires softmax probability vectors from each base
learner. No hidden dense layer is added, since the paper mentions none.

---

## 9. CNN depth and the shared decision layer

**Paper:** Gives one value each for filters (128), kernel size (3), pool size (2)
and dropout (0.3), which implies a single conv+pool block per branch. Section III
says the two branches are "processed concurrently ... before joining into a
common decision layer", and Section IV-E adds "This output is flattened and
passed to fully connected layers" — plural, with no widths given.

**Implemented:** One `Conv1D(128, 3) → MaxPool(2) → Flatten` per branch,
concatenated, then `Dropout(0.3) → Dense(128, relu) → Dense(3, softmax)`. The
shared dense width is set to 128 (equal to `filters`) so that no new magnitude is
invented; it is exposed as `CNN_PARAMS["dense_units"]` if you want to change it.

---

## 10. Wilcoxon p-values are not attainable at n = 5

**Paper, Table IV:** Reports Wilcoxon signed-rank p-values of 0.0037, 0.0075 and
0.0051, from "the 5-fold accuracy scores".

**Problem:** With 5 paired observations, the exact two-sided Wilcoxon
signed-rank distribution has a **minimum attainable p-value of 0.0625**
(= 2/2⁵). No p-value below 0.01 can be produced by an exact 5-sample Wilcoxon
test. The reported values are only reachable via a normal approximation, which
is not valid at n = 5.

**Implemented:** `src/statistical_tests.py` reports the exact p-value, the
normal-approximation p-value, and the paired t-test, with a printed note. The
paired t-test *can* legitimately fall below 0.01 at n = 5 and is the more
defensible of the two tests here.

---

## 11. Post-match features are retained — and they are the reason accuracy is high

This is the most consequential item in this file, so it is stated plainly.

**Paper, Section III:** Lists exactly what is dropped — "team names, league and
year, raw match scores, and goal statistics" — "to avoid data leakage and ensure
valid generalization."

**What survives that list:** `Match Excitement`, `Home Team Rating`, and
`Away Team Rating`. These are **post-match** quantities. A team rating is
assigned by the data provider *after* the match, informed by how the match went,
including who scored. `Match Excitement` likewise.

**Consequence:** These features carry information about the result that no
pre-match model could have. They are the most plausible explanation for the
paper's ~88.8% accuracy against a literature baseline of 70–81%, and for the
fact that XGBoost alone reaches roughly the same accuracy as the full ensemble in
this reproduction.

**Implemented:** Retained. The paper's drop list is explicit and does not include
them, and the brief is to implement the paper exactly.

**To test the effect:** remove the three columns from the feature set and rerun.
Expect accuracy to fall substantially — toward the 50–60% range that pre-match
football prediction typically achieves. This is a one-line change; see the
"Sensitivity check" section of the README.

---

## 12. Dataset size: 12,062 matches, not 13,000

**Paper:** "approximately 13,000 matches", stated consistently.

**Actual file:** `combined_data.csv` contains **12,062** matches across the five
leagues (English 2,660, Spanish 2,660, Italian 2,658, French 2,554, German 1,530)
over seasons 2014–2020.

**Assessment:** The class distribution matches the paper's Fig. 4 to the decimal
— Win 44.5%, Loss 30.5%, Draw 25.0% — and the raw counts (5,372 / 3,674 / 3,016)
match the bar heights in Fig. 1 (~5,400 / ~3,700 / ~3,000). This is the correct
file; "approximately 13,000" is a rounding of 12,062 on the generous side.

---

## 13. Figure 5's confusion matrix does not match a 20% test split

**Paper, Fig. 5:** The confusion matrix cells sum to **107 matches**
(35+2+0+3+20+3+0+4+40).

**Expected:** A 20% test split of ~12,000 matches is ~2,400 matches.

**Assessment:** Fig. 5 cannot depict the test set the rest of Section V
describes. It is either a subsample, a percentage table rendered as counts, or a
figure from an earlier experiment. This implementation produces the confusion
matrix over the full test split, so its cell counts are ~22× larger than the
paper's by construction. The *shape* of the errors is the comparable quantity.

---

## 14. Three different ensemble accuracies are reported

The paper gives the ensemble's accuracy as:

| Location | Value | Context |
|---|---|---|
| Abstract, Table VI, Table VII | 88.78% | single test split |
| Table III | 88.32% ± 0.43 | 5-fold CV mean |
| Table V | 89.06% | ablation, "Full Ensemble (All)" |

Tables III and VI can coexist (different protocols). Table V's 89.06% is harder
to place, since the ablation's full-ensemble row should be the same model as
Table VI's 88.78%. This implementation reports all three quantities separately
and labels which protocol produced each, rather than reconciling them.
