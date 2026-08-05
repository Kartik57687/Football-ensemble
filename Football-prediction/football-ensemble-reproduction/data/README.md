# Dataset

## Which file, and why

The paper names its input explicitly (Section III):

> *"The file containing the league data to obtain a unified version called
> `combined_data.csv`, where the individual league data are combined, and thus we
> have a dataset that consists of approximately 13,000 matches."*

`raw/combined_data.csv` is therefore the file used. It is already the merged
five-league file, so no per-league merging step is required or performed. No
choice between candidate files was needed — the paper names one and one was
supplied.

**Source:** S. S. Naik, *Football Data Top 5 Leagues*, Kaggle, 2023
(reference [1] in the paper).

---

## Contents

| Property | Value |
|---|---|
| Matches | 12,062 |
| Columns | 42 raw → 31 modelling features |
| Seasons | 2014–2020 |
| Teams | 146 |
| Missing values | none |

**Per league**

| League | Matches |
|---|---|
| English | 2,660 |
| Spanish | 2,660 |
| Italian | 2,658 |
| French | 2,554 |
| German | 1,530 |

**Class distribution** (derived, not stored in the file)

| Outcome | Count | Share | Paper Fig. 4 |
|---|---|---|---|
| Win | 5,372 | 44.5% | 44.5% |
| Loss | 3,674 | 30.5% | 30.5% |
| Draw | 3,016 | 25.0% | 25.0% |

The match to Fig. 4 is exact, confirming this is the file the paper used. The
paper's "approximately 13,000" is a generous rounding of 12,062.

---

## Schema

### Dropped before modelling (Section III)

Removed *"to avoid data leakage and ensure valid generalization"*:

| Column | Category |
|---|---|
| `Unnamed: 0` | index |
| `Home Team`, `Away Team` | identifier |
| `league`, `year` | identifier |
| `Score`, `Half Time Score` | raw match score |
| `Home Team Goals Scored`, `Away Team Goals Scored` | goal statistic |
| `Home Team Goals Conceeded`, `Away Team Goals Conceeded` | goal statistic |

`Home Team` / `Away Team` / `year` are read first, since the target is derived
from goals and the CNN's windows are built per team in chronological order. They
are dropped before any model sees them.

### The 31 modelling features

**Match-level (3)**

`Match Excitement`, `Home Team Rating`, `Away Team Rating`

> ⚠️ These three are **post-match** quantities — the provider assigns them after
> a match, informed by how it went. They survive the paper's stated drop list,
> so they are retained here, but they carry result information no pre-match model
> could have. This is the single most likely explanation for the paper's ~88.8%
> accuracy against a 70–81% literature baseline. See
> `docs/unspecified_details.md` #11.

**Per team, home and away (14 × 2 = 28)**

| Group | Columns |
|---|---|
| Possession | `Possession %`, `Pass Success %` |
| Shooting | `Off Target Shots`, `On Target Shots`, `Total Shots`, `Blocked Shots` |
| Set pieces | `Corners`, `Throw Ins` |
| Duels / defence | `Aerials Won`, `Clearances` |
| Discipline | `Fouls`, `Yellow Cards`, `Second Yellow Cards`, `Red Cards` |

---

## Ordering caveat

The file has a `year` column but **no match date or matchday column**. The CNN's
"last 15 matches" window needs an ordering, so row order within each
`(year, league)` block is used as the chronological proxy — the only ordering
information available. If the CSV's row order does not reflect fixture order, the
CNN windows are not true match histories. See `docs/unspecified_details.md` #4.

---

## Target construction

Computed at load time as the column `Result` (Section III):

```python
Win   if home_goals >  away_goals
Loss  if home_goals <  away_goals
Draw  if home_goals == away_goals
```

Always from the **home team's perspective**.
