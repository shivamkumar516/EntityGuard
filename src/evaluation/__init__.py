"""Evaluation module (Student 3): P/R/F1, blocking metrics, threshold sweeps, confidence bands.

`from src.evaluation import prf, choose_bands, ...` keeps working exactly as before the
single file src/evaluation.py was split into this package.
"""
from .metrics import prf, blocking_metrics, threshold_sweep, best_threshold, fast_best_f1
from .bands import choose_bands, assign_band, BAND_NAMES, HIGH, REVIEW, LOW

__all__ = ["prf", "blocking_metrics", "threshold_sweep", "best_threshold", "fast_best_f1",
           "choose_bands", "assign_band", "BAND_NAMES", "HIGH", "REVIEW", "LOW"]
