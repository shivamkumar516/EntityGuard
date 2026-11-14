import json

import numpy as np
import pandas as pd
import pytest

from src.matching import MatchConfig, load_config, match, predict, score_one, title_combined, weighted_score

CFG = MatchConfig(alpha=0.6, weights=(0.45, 0.30, 0.05, 0.20), author_measure="author_best",
                  venue_measure="venue_tok", threshold=0.65, band_high=0.90, band_low=0.49)


def _summary(**chosen):
    base = {"alpha": 0.6, "weights_title_author_venue_year": [0.45, 0.3, 0.05, 0.2],
            "author_measure": "author_best", "venue_measure": "venue_tok",
            "threshold": 0.65, "band_high": 0.9, "band_low": 0.49}
    base.update(chosen)
    return {"chosen": base}


def test_reads_the_real_summary_json():
    cfg = load_config()
    assert abs(sum(cfg.weights) - 1) < 1e-9
    assert cfg.band_low <= cfg.threshold <= cfg.band_high


def test_config_is_not_hard_coded(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(_summary(threshold=0.8, band_high=0.95)))
    cfg = load_config(str(p))
    assert cfg.threshold == 0.8 and cfg.band_high == 0.95


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        MatchConfig.from_summary(_summary(weights_title_author_venue_year=[0.5, 0.5, 0.5, 0.5]))
    with pytest.raises(ValueError):
        MatchConfig.from_summary(_summary(threshold=0.95, band_high=0.9))


def test_weighted_score_formula_and_renormalization():
    df = pd.DataFrame({"title_combined": [1.0, 1.0], "author_best": [0.5, 0.5],
                       "venue_tok": [0.0, 0.0], "year": [1.0, np.nan]})
    s = weighted_score(df, CFG.weights, "author_best", "venue_tok")
    assert s[0] == pytest.approx(0.45 + 0.15 + 0 + 0.20)
    # year missing -> dropped, the other weights are rescaled to sum to 1
    assert s[1] == pytest.approx((0.45 + 0.15) / 0.80)


def test_title_combined():
    assert title_combined(1.0, 0.0, 0.6) == pytest.approx(0.6)


def test_match_adds_score_prediction_and_band():
    feats = pd.DataFrame({"title_edit": [1.0, 0.2, 0.7], "title_jac": [1.0, 0.1, 0.6],
                          "author_best": [1.0, 0.0, 0.7], "venue_tok": [1.0, 0.0, np.nan],
                          "year": [1.0, 0.0, np.nan]})
    out = match(feats, CFG)
    assert list(out.prediction) == [1, 0, 1]
    assert list(out.band) == ["High confidence", "Likely non-match", "Manual review"]
    assert "score" not in feats  # input not modified


def test_predict_threshold_is_inclusive():
    assert list(predict([0.65, 0.6499], 0.65)) == [1, 0]


def test_score_one_handles_missing_fields():
    r = score_one(1.0, None, None, None, CFG)
    assert r["overall_similarity"] == pytest.approx(1.0) and r["band"] == "High confidence"
