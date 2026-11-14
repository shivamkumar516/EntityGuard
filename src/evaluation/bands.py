"""Confidence bands (Section 41): high confidence / manual review / likely non-match.

The two boundaries are chosen from VALIDATION data, never guessed and never taken from test:

* T_HIGH = the smallest threshold whose predictions are at least `precision_target` precise
  (default 97 %). Pairs at or above it are accepted without a human.
* T_LOW  = the largest threshold that still keeps at least `recall_keep` (default 99 %) of the
  true matches that blocking found. Pairs below it are rejected without a human.
* Everything in between goes to manual review.

The operating threshold T always lies inside the manual-review band
(T_LOW <= T <= T_HIGH), so the bands never contradict the yes/no prediction.
"""
import numpy as np

from .metrics import threshold_sweep

HIGH, REVIEW, LOW = "High confidence", "Manual review", "Likely non-match"
BAND_NAMES = (HIGH, REVIEW, LOW)
DEFAULT_GRID = np.round(np.arange(0.40, 0.991, 0.01), 2)


def choose_bands(scores, labels, threshold, grid=DEFAULT_GRID,
                 precision_target=0.97, recall_keep=0.99, fallback_high=0.95):
    """Return (t_high, t_low) chosen on validation scores/labels of the candidate pairs.

    scores, labels : arrays for the validation candidate pairs (label 1 = gold match)
    threshold      : the operating threshold T, also chosen on validation
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    grid = np.asarray(grid, dtype=float)

    # precision does not depend on n_gold, so passing labels.sum() is fine here
    sweep = threshold_sweep(scores, labels, int(labels.sum()), grid)
    precise = sweep.threshold[sweep.precision >= precision_target]
    t_high = float(precise.min()) if len(precise) else fallback_high

    n_true = labels.sum()
    kept = np.array([labels[scores >= t].sum() / n_true if n_true else 0.0 for t in grid])
    t_low = float(grid[kept >= recall_keep].max()) if (kept >= recall_keep).any() else float(grid.min())

    return max(t_high, threshold), min(t_low, threshold)


def assign_band(scores, t_high, t_low):
    """Band name for each score."""
    s = np.asarray(scores, dtype=float)
    return np.where(s >= t_high, HIGH, np.where(s >= t_low, REVIEW, LOW))
