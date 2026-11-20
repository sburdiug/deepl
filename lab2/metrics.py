"""Binary counts and an O(n log n) threshold search."""

import numpy as np
import pandas as pd


def validate(y: np.ndarray, scores: np.ndarray) -> None:
    if y.ndim != 1 or scores.shape != y.shape or y.size == 0:
        raise ValueError("Labels and scores must be nonempty vectors of equal length.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be probabilities in [0, 1].")


def counts(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    validate(y, scores)
    if np.isnan(threshold):
        raise ValueError("Threshold cannot be NaN.")
    positive = scores >= threshold
    tp = int(np.count_nonzero(positive & (y == 1)))
    fp = int(np.count_nonzero(positive & (y == 0)))
    fn = int(np.count_nonzero(~positive & (y == 1)))
    tn = int(np.count_nonzero(~positive & (y == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp}


def threshold_table(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    validate(y, scores)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), y.size - 1]
    tp = np.r_[0, np.cumsum(sorted_y, dtype=np.int64)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    fn = int(y.sum()) - tp
    tn = y.size - int(y.sum()) - fp
    return pd.DataFrame({
        "threshold": np.r_[np.inf, sorted_scores[ends]],
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp,
    })


def best_threshold(table: pd.DataFrame) -> float:
    return float(table.loc[table["C"].eq(table["C"].min()), "threshold"].max())
