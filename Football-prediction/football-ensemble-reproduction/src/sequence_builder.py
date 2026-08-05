"""
Sliding-window sequence construction for the 1D-CNN -- Section III.

The paper:

    "For convolutional modelling, we defined temporal sequences per team post
     hoc, using a fixed-length sliding window of the last 15 matches before each
     game. Separate sequences for the home and away teams are maintained. Each
     sequence is shaped (15, d), where d is the number of numerical predictors.
     Although separate home and away sequences are splitted, they are processed
     concurrently in the same pipeline using parallel 1D convolutional core
     components before joining into a common decision layer."

So each match yields TWO tensors of shape (15, d): one summarising the home
team's previous 15 fixtures and one summarising the away team's. A team's
"previous fixtures" include games it played in either role.

Two details the paper leaves open, both flagged in docs/unspecified_details.md:

  #4  Chronological ordering. The dataset has no date column, only `year`.
      Row order within a (year, league) block is used as the ordering proxy.

  #6  Short histories. Early-season matches have fewer than 15 prior fixtures.
      The paper mentions zero-padding only for the BiLSTM, not the CNN. Front
      zero-padding is applied here so that the window is left-padded and the
      most recent match always occupies the final timestep. No rows are dropped,
      which keeps the CNN's sample count aligned with XGBoost and the BiLSTM --
      a hard requirement for the row-wise concatenation in Layer 2 (Eq. 13).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import config


def build_team_sequences(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    window_size: int | None = None,
    verbose: bool = True,
):
    """
    Build the paired (home, away) history tensors for every match.

    Parameters
    ----------
    df : DataFrame
        Must contain `Home Team`, `Away Team` and `match_order`. Row order must
        match `features_scaled` row-for-row.
    features_scaled : ndarray, shape (n_matches, d)
        Min-Max scaled numerical predictors.

    Returns
    -------
    home_seq : ndarray, shape (n_matches, window_size, d)
    away_seq : ndarray, shape (n_matches, window_size, d)
    """
    window_size = window_size or config.CNN_PARAMS["window_size"]
    n_matches, d = features_scaled.shape

    # Chronological traversal order.
    order = np.argsort(df["match_order"].to_numpy(), kind="mergesort")

    home_teams = df["Home Team"].to_numpy()
    away_teams = df["Away Team"].to_numpy()

    home_seq = np.zeros((n_matches, window_size, d), dtype=np.float32)
    away_seq = np.zeros((n_matches, window_size, d), dtype=np.float32)

    # team -> list of row positions of that team's matches, chronologically.
    history: dict[str, list[int]] = defaultdict(list)

    short_history = 0

    for pos in order:
        for team, target in ((home_teams[pos], home_seq), (away_teams[pos], away_seq)):
            past = history[team][-window_size:]
            if past:
                # Left-pad: the most recent prior match sits at the last timestep.
                target[pos, window_size - len(past):, :] = features_scaled[past]
            if len(past) < window_size:
                short_history += 1

        # Only after both windows are written does the current match enter the
        # history -- a team's window must contain matches strictly *before* it.
        history[home_teams[pos]].append(pos)
        history[away_teams[pos]].append(pos)

    if verbose:
        total_windows = 2 * n_matches
        print(f"CNN sequences built       : {n_matches:,} matches x 2 branches")
        print(f"Sequence shape            : ({window_size}, {d})")
        print(f"Zero-padded windows       : {short_history:,} / {total_windows:,} "
              f"({100 * short_history / total_windows:.1f}%) -- teams with "
              f"< {window_size} prior fixtures")
        print(f"Distinct teams            : {len(history):,}")

    return home_seq, away_seq


def build_cnn_inputs(df, feature_columns, train_idx, test_idx, verbose: bool = True):
    """
    Full CNN input pipeline: Min-Max scale (fit on train) then window.

    Returns
    -------
    (home_train, away_train), (home_test, away_test), scaler
    """
    from .preprocessing import fit_minmax_scaler

    X = df[feature_columns].to_numpy(dtype=np.float32)
    scaler = fit_minmax_scaler(X[train_idx])
    X_scaled = scaler.transform(X).astype(np.float32)

    home_seq, away_seq = build_team_sequences(df, X_scaled, verbose=verbose)

    return (
        (home_seq[train_idx], away_seq[train_idx]),
        (home_seq[test_idx], away_seq[test_idx]),
        scaler,
    )
