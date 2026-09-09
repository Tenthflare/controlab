"""
Aggregation with honest uncertainty via rliable.

IQM + stratified-bootstrap 95% CIs, and pairwise claims as
probability-of-improvement.

rliable convention: every score array is shape (num_runs, num_tasks). For this study
a "run" is a training seed (5) and a "task" is a held-out dynamics instance
(30), so each (alpha, bucket) becomes a (5, 30) matrix once the 5 episodes per
cell are averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics


def score_matrix( df: pd.DataFrame,
                  alpha: float,
                  bucket: str,
                  value: str = "episode_reward") -> np.ndarray:
    """
    (num_seeds, num_tasks) score matrix for one (alpha, bucket).

    Averages the 5 episodes per (seed, num_tasks) into one cell score -> the
    (n_runs, n_tasks) array rliable expects. Use value="survived" for the
    secondary survival-rate metric.
    """
    sub = df[(df["alpha"] == alpha) & (df["bucket"] == bucket)]
    cell = (
        sub.groupby(["seed", "set_id"])[value]
        .mean()  # 5 episodes -> 1 score per (seed, value_set)
        .unstack("set_id")  # rows = seeds, cols = value_set
        .sort_index()
    )
    return cell.to_numpy()


def aggregate_iqm(x):
    return np.array([metrics.aggregate_iqm(x)])


def iqm_with_ci(scores: np.ndarray,
                reps: int = 5000,
                alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Interquartile mean + (1-alpha) stratified-bootstrap CI.

    `scores`: (n_runs, n_tasks). Returns (iqm, lo, hi).
    IQM = mean of the middle 50% of all runs*tasks scores -> robust to the
    occasional blow-up or total failure, which plain mean/median aren't.
    """
    # get_interval_estimates wants {name: scores} and an aggregate fn that
    # returns an array of metrics; we pass a single metric (IQM).
    point, cis = rly.get_interval_estimates(
        {"_": scores},
        aggregate_iqm,
        reps=reps,
        confidence_interval_size=1.0 - alpha,
    )
    iqm = float(point["_"][0])  # point estimate
    lo, hi = float(cis["_"][0, 0]), float(cis["_"][1, 0])  # cis shape (2, n_metrics)
    return iqm, lo, hi


def probability_of_improvement(scores_x: np.ndarray,
                               scores_y: np.ndarray,
                               reps: int = 5000,
                               alpha: float = 0.05) -> tuple[float, float, float]:
    """
    P(breadth X > breadth Y) averaged over tasks, with bootstrap CI.

    Both arrays: (n_runs, n_tasks). Returns (poi, lo, hi).
    poi > 0.5 means X tends to beat Y on a randomly drawn (run, task); the honest,
    interpretable version of "X is better than Y" for a few-seed sim study.
    """
    # For POI the dict value is a (x, y) PAIR, and the metric is passed directly.
    poi, cis = rly.get_interval_estimates(
        {"x,y": (scores_x, scores_y)},
        metrics.probability_of_improvement,
        reps=reps,
        confidence_interval_size=1.0 - alpha,
    )
    p = float(np.ravel(poi["x,y"])[0])
    lo, hi = (float(v) for v in np.ravel(cis["x,y"])[:2])
    return p, lo, hi


def robustness_curve(df: pd.DataFrame,
                     bucket: str,
                     value: str = "episode_reward",
                     reps: int = 5000) -> pd.DataFrame:
    """
    Per-alpha IQM (+95% CI) for one bucket -> tidy frame for plotting.

    Columns: [alpha, iqm, lo, hi], one row per alpha ascending. For robustness-breadth figure.
    """
    rows = []
    for alpha in sorted(df["alpha"].unique()):
        scores = score_matrix(df, alpha, bucket, value)
        iqm, lo, hi = iqm_with_ci(scores, reps=reps)
        rows.append({"alpha": alpha, "iqm": iqm, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
