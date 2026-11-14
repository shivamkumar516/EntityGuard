import numpy as np
import pandas as pd

from src.ml import FEATURE_NAMES, load_model, make_X, predict_proba, sample_training_pairs, save_model, train_models


def _fake_features(n, seed):
    """Synthetic candidate pairs: matches have high title/author similarity, ~10 % are matches."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(np.int8)
    noise = lambda: rng.normal(0, 0.12, n)
    df = pd.DataFrame({
        "title_combined": np.clip(0.4 + 0.45 * y + noise(), 0, 1),
        "author_best": np.clip(0.3 + 0.5 * y + noise(), 0, 1),
        "venue_tok": np.clip(0.3 + 0.2 * y + noise(), 0, 1),
        "year": np.where(rng.random(n) < 0.5, np.nan, y * 1.0),
        "len_ratio": np.clip(0.7 + 0.2 * y + noise(), 0, 1),
        "label": y,
    })
    df.loc[rng.random(n) < 0.1, "author_best"] = np.nan
    return df


def test_feature_matrix_flags_missing_values():
    df = _fake_features(50, 0)
    X = make_X(df, "author_best", "venue_tok")
    assert X.shape == (50, len(FEATURE_NAMES))
    assert not np.isnan(X).any()
    assert (X[:, 7] == df.year.isna()).all()


def test_negative_sampling_keeps_all_positives_and_is_reproducible():
    df = _fake_features(3000, 1)
    a = sample_training_pairs(df, neg_per_pos=2)
    b = sample_training_pairs(df, neg_per_pos=2)
    assert (a.label == 1).sum() == (df.label == 1).sum()
    assert (a.label == 0).sum() == 2 * (df.label == 1).sum()
    assert a.index.equals(b.index)


def test_train_save_load_round_trip(tmp_path):
    train, val = _fake_features(4000, 2), _fake_features(1500, 3)
    table, best, info = train_models(train, val, int(val.label.sum()), "author_best", "venue_tok")
    assert len(table) == 10 and set(best) == {"LR", "DT"}
    assert best["LR"]["val_f1"] > 0.8                       # easy synthetic data
    assert info["positives"] == int((train.label == 1).sum())

    path = save_model(best["LR"], "author_best", "venue_tok", path=str(tmp_path / "lr.joblib"))
    bundle = load_model(path)
    assert bundle["feature_names"] == FEATURE_NAMES and 0 < bundle["threshold"] < 1
    p_saved = predict_proba(bundle, val)
    p_orig = best["LR"]["model"].predict_proba(make_X(val, "author_best", "venue_tok"))[:, 1]
    assert np.allclose(p_saved, p_orig)
