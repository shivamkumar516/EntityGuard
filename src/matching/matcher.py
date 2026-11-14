"""Matching engine: weighted record score -> threshold decision -> confidence band.

    S = w_t*S_title + w_a*S_author + w_v*S_venue + w_y*S_year      (weights sum to 1)
    S_title = alpha*S_edit + (1 - alpha)*S_jaccard

A missing field (NaN) is left out and the remaining weights are renormalized, so a pair is
not punished just because Scholar has no year (54 % of Scholar records).

All tuned values (alpha, weights, measures, threshold, band limits) are read from
outputs/summary.json, which the notebook writes after choosing them on the VALIDATION split.
Nothing here is hard-coded, so re-running the experiments automatically updates the matcher.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation.bands import assign_band

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SUMMARY = os.path.join(REPO_ROOT, "outputs", "summary.json")


@dataclass(frozen=True)
class MatchConfig:
    alpha: float
    weights: tuple          # (title, author, venue, year)
    author_measure: str     # feature column used as S_author, e.g. "author_best"
    venue_measure: str      # feature column used as S_venue, e.g. "venue_tok"
    threshold: float
    band_high: float
    band_low: float

    @classmethod
    def from_summary(cls, summary):
        c = summary["chosen"]
        w = tuple(float(x) for x in c["weights_title_author_venue_year"])
        if len(w) != 4 or abs(sum(w) - 1) > 1e-6:
            raise ValueError(f"weights must be 4 numbers summing to 1, got {w}")
        cfg = cls(alpha=float(c["alpha"]), weights=w, author_measure=c["author_measure"],
                  venue_measure=c["venue_measure"], threshold=float(c["threshold"]),
                  band_high=float(c["band_high"]), band_low=float(c["band_low"]))
        if not cfg.band_low <= cfg.threshold <= cfg.band_high:
            raise ValueError(f"expected band_low <= threshold <= band_high, got {cfg}")
        return cfg


def load_config(path=DEFAULT_SUMMARY):
    with open(path, encoding="utf-8") as f:
        return MatchConfig.from_summary(json.load(f))


def title_combined(edit_sim, jaccard_sim, alpha):
    """S_title = alpha*S_edit + (1-alpha)*S_jaccard (works on numbers or arrays)."""
    return alpha * np.asarray(edit_sim) + (1 - alpha) * np.asarray(jaccard_sim)


def weighted_score(df, w, author_col, venue_col, renormalize=True):
    """Record score for every row of a feature table.

    df needs the columns title_combined, <author_col>, <venue_col> and year (each in [0, 1] or NaN).
    With renormalize=True a NaN field is dropped and the other weights are rescaled to sum to 1.
    """
    comps = [(w[0], df["title_combined"].to_numpy(dtype=float)), (w[1], df[author_col].to_numpy(dtype=float)),
             (w[2], df[venue_col].to_numpy(dtype=float)), (w[3], df["year"].to_numpy(dtype=float))]
    num = np.zeros(len(df)); den = np.zeros(len(df))
    for wk, s in comps:
        ok = ~np.isnan(s)
        num += wk * np.where(ok, s, 0)
        den += wk * (ok if renormalize else 1)
    return (num / np.maximum(den, 1e-12)).astype(np.float32)


def predict(scores, threshold):
    """Threshold step: 1 = match, 0 = non-match."""
    return (np.asarray(scores) >= threshold).astype(np.int8)


def match(features, config=None):
    """Score, predict and band a feature table. Returns a copy with score / prediction / band columns.

    If `title_combined` is missing but `title_edit` and `title_jac` are present, it is built with config.alpha.
    """
    config = config or load_config()
    out = features.copy()
    if "title_combined" not in out and {"title_edit", "title_jac"} <= set(out.columns):
        out["title_combined"] = title_combined(out["title_edit"], out["title_jac"], config.alpha)
    out["score"] = weighted_score(out, config.weights, config.author_measure, config.venue_measure)
    out["prediction"] = predict(out["score"], config.threshold)
    out["band"] = assign_band(out["score"], config.band_high, config.band_low)
    return out


def score_one(title, author, venue, year, config):
    """Score a single pair from its four field similarities (any may be None/NaN = missing)."""
    row = pd.DataFrame({"title_combined": [title], config.author_measure: [author],
                        config.venue_measure: [venue], "year": [year]}).astype(float)
    s = float(weighted_score(row, config.weights, config.author_measure, config.venue_measure)[0])
    return {"overall_similarity": s, "prediction": int(s >= config.threshold),
            "band": str(assign_band([s], config.band_high, config.band_low)[0])}
