import numpy as np
import pytest

from src.evaluation import HIGH, LOW, REVIEW, assign_band, choose_bands


def _toy_validation():
    """98 matches at 0.90 and 2 weak matches at 0.45; 3 non-matches at 0.90 and 20 at 0.80.

    precision for 0.80 < t <= 0.90 : 98 / 101 = 0.970  -> first grid value reaching 97 % is 0.81
    matches kept for 0.45 < t      : 98 / 100 = 0.98   -> 99 % is kept only up to t = 0.45
    """
    scores = np.array([0.90] * 98 + [0.45] * 2 + [0.90] * 3 + [0.80] * 20)
    labels = np.array([1] * 100 + [0] * 23)
    return scores, labels


def test_bands_come_from_the_data():
    s, y = _toy_validation()
    t_high, t_low = choose_bands(s, y, threshold=0.60)
    assert t_high == pytest.approx(0.81)
    assert t_low == pytest.approx(0.45)


def test_operating_threshold_is_always_inside_the_review_band():
    s, y = _toy_validation()
    assert choose_bands(s, y, threshold=0.85)[0] == pytest.approx(0.85)   # T_HIGH raised to T
    assert choose_bands(s, y, threshold=0.30)[1] == pytest.approx(0.30)   # T_LOW lowered to T


def test_fallback_when_precision_target_is_never_reached():
    s = np.array([0.9, 0.9, 0.5]); y = np.array([1, 0, 1])
    t_high, _ = choose_bands(s, y, threshold=0.6)
    assert t_high == pytest.approx(0.95)


def test_assign_band_boundaries():
    out = assign_band([0.95, 0.90, 0.89, 0.49, 0.48], t_high=0.90, t_low=0.49)
    assert list(out) == [HIGH, HIGH, REVIEW, REVIEW, LOW]
