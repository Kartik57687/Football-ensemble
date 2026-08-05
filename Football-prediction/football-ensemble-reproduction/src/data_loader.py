"""
Dataset loading and target construction -- Section III of the paper.

Dataset selection
-----------------
The paper names its input file explicitly:

    "The file containing the league data to obtain a unified version called
     combined_data.csv, where the individual league data are combined, and thus
     we have a dataset that consists of approximately 13,000 matches."

`combined_data.csv` is therefore the file used. It is the already-merged
five-league file; no per-league merging step is required or performed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Load the combined five-league match file."""
    path = path or config.RAW_DATA_FILE
    df = pd.read_csv(path)

    # The CSV carries a nameless pandas index column; normalise its name so the
    # drop list in config matches regardless of pandas version.
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "Unnamed: 0"})

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the three-class target from the home team's perspective.

    Section III:
        Win  -> home goals >  away goals
        Loss -> home goals <  away goals
        Draw -> home goals == away goals

    The paper calls this computed column `Result`.
    """
    home = df["Home Team Goals Scored"]
    away = df["Away Team Goals Scored"]

    df = df.copy()
    df[config.TARGET_COLUMN] = np.select(
        [home > away, home < away],
        ["Win", "Loss"],
        default="Draw",
    )
    return df


def add_chronological_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a monotonically increasing `match_order` used to build the CNN's
    "last 15 matches" windows.

    UNSPECIFIED: the dataset has no match-date column -- only `year`. The paper
    does not state how matches are ordered inside a season. Row order within a
    (year, league) block is used as the chronological proxy, which is the only
    ordering information the file carries.
    See docs/unspecified_details.md #4.
    """
    df = df.copy()
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["year", "league", "_row"], kind="mergesort")
    df["match_order"] = np.arange(len(df))
    return df.sort_values("_row", kind="mergesort").drop(columns="_row")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the modelling features: everything except the dropped identifier /
    score / goal columns, the target, and internal bookkeeping columns.

    Section III: "team names, league and year, raw match scores, and goal
    statistics" are removed "to avoid data leakage".
    """
    excluded = set(config.DROP_COLUMNS) | {
        config.TARGET_COLUMN,
        "match_order",
    }
    features = [c for c in df.columns if c not in excluded]

    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns survived the drop: {non_numeric}")

    return features


def load_dataset(path=None, verbose: bool = True):
    """
    Full loading pipeline: read -> build target -> order chronologically.

    Returns
    -------
    df : pd.DataFrame
        Full frame including identifier columns (needed for CNN sequence
        construction, which groups by team).
    feature_columns : list[str]
        The numerical predictors fed to the models.
    """
    df = load_raw(path)
    df = build_target(df)
    df = add_chronological_order(df)
    feature_columns = get_feature_columns(df)

    if verbose:
        counts = df[config.TARGET_COLUMN].value_counts()
        pct = df[config.TARGET_COLUMN].value_counts(normalize=True) * 100
        print(f"Matches loaded            : {len(df):,}")
        print(f"Leagues                   : {df['league'].nunique()} "
              f"({', '.join(sorted(df['league'].unique()))})")
        print(f"Seasons                   : {df['year'].min()}-{df['year'].max()}")
        print(f"Numerical features        : {len(feature_columns)}")
        print("Class distribution:")
        for cls in config.CLASS_NAMES:
            print(f"  {cls:<5} {counts.get(cls, 0):>6,}  ({pct.get(cls, 0):5.2f}%)")

    return df, feature_columns


def encode_labels(y: pd.Series) -> np.ndarray:
    """Map the string target to the integer indices used by the models."""
    return y.map(config.CLASS_TO_INDEX).to_numpy(dtype=np.int64)
