"""Tests for src/pipeline.py (the standalone logic behind app.py).

Unit tests use the examples of the problem statement. The integration tests (skipped if data/ or the
app database are missing) check that the pipeline reproduces the notebook's stored results.
"""
import os
import random

import numpy as np
import pandas as pd
import pytest

from src import pipeline as P

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_DATA = os.path.exists(os.path.join(REPO, "data", "Scholar.csv"))
HAVE_DB = os.path.exists(os.path.join(REPO, "outputs", P.APP_DB_NAME))


# ---------------------------------------------------------------- normalization
def test_normalization_examples():
    assert P.normalize_title("An Efficient Method for Record-Linkage!") == "efficient method record linkage"
    assert P.author_tokens("Smith, J.; Kumar, A.") == ["smith", "j", "kumar", "a"]
    assert P.split_author_names("Smith, J.; Kumar, A.") == ["smith j", "kumar a"]
    assert P.split_author_names("A Gorelik, Y Wang") == ["a gorelik", "y wang"]
    assert P.normalize_venue("Proc. ICDE") == "proc icde"
    assert P.parse_year("2005") == 2005 and P.parse_year("11") is None and P.parse_year("") is None
    assert P.parse_year(pd.NA) is None and P.parse_year(float("nan")) is None


# ---------------------------------------------------------------- similarity
def test_levenshtein_dp_and_two_rows():
    assert P.levenshtein("kitten", "sitting") == 3
    assert P.levenshtein("", "abc") == 3 and P.levenshtein("abc", "") == 3
    rng = random.Random(0)
    for _ in range(200):
        a = "".join(rng.choice("abcde ") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("abcde ") for _ in range(rng.randint(0, 12)))
        assert P.levenshtein(a, b) == P.levenshtein_two_rows(a, b)
        assert P.edit_similarity(a, b) == pytest.approx(P.fast_edit_similarity(a, b))


def test_edit_jaccard_ngrams_year():
    assert P.edit_similarity("algorithm", "algorithms") == pytest.approx(0.9)
    assert P.jaccard({"deep", "neural", "network", "classification"}, {"neural", "network", "classification"}) == 0.75
    assert P.char_ngrams("database", 2) == ["da", "at", "ta", "ab", "ba", "as", "se"]
    assert P.char_ngrams("database", 3) == ["dat", "ata", "tab", "aba", "bas", "ase"]
    assert P.year_similarity(2005, 2005) == 1.0 and P.year_similarity(2005, 2006) == 0.5
    assert P.year_similarity(2005, 2008) == 0.0 and np.isnan(P.year_similarity(2005, None))


def test_author_best_match_is_order_independent():
    a, b = ["anil kumar", "raj singh"], ["raj singh", "anil kumar"]
    assert P.author_best_match(a, b) == 1.0
    assert np.isnan(P.author_best_match([], b))


# ---------------------------------------------------------------- score & decision
def test_weighted_score_formula_and_renormalization():
    w = {"title": 0.50, "author": 0.30, "venue": 0.10, "year": 0.10}      # Section 23 example weights
    sims = {"title": 0.9, "author": 0.8, "venue": 0.5, "year": 1.0}
    s, rows = P.weighted_score(sims, w)
    assert s == pytest.approx(0.50 * 0.9 + 0.30 * 0.8 + 0.10 * 0.5 + 0.10 * 1.0)
    assert sum(r["contribution"] for r in rows) == pytest.approx(s)
    s2, rows2 = P.weighted_score({**sims, "year": float("nan")}, w)
    assert s2 == pytest.approx((0.50 * 0.9 + 0.30 * 0.8 + 0.10 * 0.5) / 0.90)       # Section 24
    assert [r["field"] for r in rows2 if r["similarity"] is None] == ["year"]
    df = pd.DataFrame([{"title_combined": 0.9, "author_best": 0.8, "venue_tri": 0.5, "year": np.nan}])
    assert P.weighted_score_frame(df, w)[0] == pytest.approx(s2, abs=1e-6)


def test_app_rule_is_the_validation_choice():
    """The app's weights and bands must come from the experiments (outputs/summary.json), not be hand-picked."""
    w, auto_t, non_t = P.validated_config()
    if os.path.exists(os.path.join(REPO, "outputs", "summary.json")):
        import json
        with open(os.path.join(REPO, "outputs", "summary.json"), encoding="utf-8") as f:
            c = json.load(f)["chosen"]
        assert [w[k] for k in P.FIELDS] == pytest.approx(c["weights_title_author_venue_year"])
        assert (auto_t, non_t) == (c["band_high"], c["band_low"])
    eng = P.EntityGuard(load_data=False)
    assert (eng.weights, eng.auto_t, eng.non_t) == (w, auto_t, non_t)
    assert P.validated_config("no/such/folder") == (P.FALLBACK_WEIGHTS, P.FALLBACK_AUTO_MATCH_T, P.FALLBACK_NON_MATCH_T)


def test_decision_boundaries():
    assert P.decide(0.90, 0.90, 0.49) == P.AUTO_MATCH and P.decide(0.8999, 0.90, 0.49) == P.MANUAL_REVIEW
    assert P.decide(0.49, 0.90, 0.49) == P.MANUAL_REVIEW and P.decide(0.4899, 0.90, 0.49) == P.NON_MATCH
    hi, lo = P.AUTO_MATCH_T, P.NON_MATCH_T
    assert list(P.decide_array([hi, (hi + lo) / 2, lo - 0.01])) == [P.AUTO_MATCH, P.MANUAL_REVIEW, P.NON_MATCH]


def test_score_pair_without_data_explains_missing_year():
    eng = P.EntityGuard(load_data=False)
    d = P.prepare_record("An Efficient Algorithm for Entity Matching", "A. Kumar, R. Singh", "VLDB", 2005)
    s = P.prepare_record("Efficient Algorithms for Entity Matching", "Anil Kumar; Raj Singh", "Proc. VLDB", None)
    res = eng.score_pair(d, s)
    assert res["decision"] in P.DECISION_LABELS
    assert np.isnan(res["sims"]["year"]) and "renormalized" in res["explanation"]
    assert res["blocking"] is None                      # no data -> no blocking check
    for m in res["ml"].values():
        assert 0 <= m["probability"] <= 1


# ---------------------------------------------------------------- review store
def test_review_store_roundtrip(tmp_path):
    path = str(tmp_path / "reviews.csv")
    P.save_review(path, "d1", "s1", "Confirm Match", "same paper")
    P.save_review(path, "d2", "s2", "Reject Match")
    P.save_review(path, "d1", "s1", "Needs Review")           # replaces the first decision
    r = P.load_reviews(path)
    assert len(r) == 2
    assert r.set_index("dblp_id").loc["d1", "decision"] == "Needs Review"
    with pytest.raises(ValueError):
        P.save_review(path, "d3", "s3", "Maybe")


# ---------------------------------------------------------------- integration with the notebook's results
@pytest.fixture(scope="module")
def engine():
    if not HAVE_DATA:
        pytest.skip("data/ not available")
    return P.EntityGuard(load_data=True)


@pytest.fixture(scope="module")
def stored():
    if not HAVE_DB:
        pytest.skip("outputs/entityguard_app.db not available")
    return P.load_stored_candidates()


@pytest.mark.skipif(not (HAVE_DATA and HAVE_DB), reason="needs data/ and the app database")
def test_pair_scores_equal_notebook_match_result(engine, stored):
    df, _ = stored
    sample = df.sample(300, random_state=1)
    lr = engine.models.get("Logistic Regression")
    for row in sample.itertuples():
        d, s = engine.record("dblp", row.dblp_id), engine.record("scholar", row.scholar_id)
        res = engine.score_pair(d, s, with_blocking=False)
        for field, col in P.STORED_COLUMNS.items():
            exp = getattr(row, col)
            got = res["sims"][field]
            assert (np.isnan(got) and pd.isna(exp)) or got == pytest.approx(exp, abs=1e-5), (row.dblp_id, field)
        if lr is not None:
            assert res["ml"]["Logistic Regression"]["probability"] == pytest.approx(row.lr_probability, abs=1e-4)


@pytest.mark.skipif(not HAVE_DATA, reason="needs data/")
def test_single_record_blocking_equals_batch_blocking(engine):
    ix = engine.index()
    union, _, _ = ix.run_all()
    n_s = ix.n_s
    rng = np.random.default_rng(0)
    for di in rng.choice(ix.n_d, 40, replace=False):
        keys = {p: ix.d_keys[p][di] for p in P.PASS_NAMES}
        got = set(ix.candidates(keys))
        exp = set((union[(union // n_s) == di] % n_s).tolist())
        assert got == exp
    for si in rng.choice(n_s, 40, replace=False):
        keys = {p: ix.s_keys[p][si] for p in P.PASS_NAMES}
        got = set(ix.candidates_for_scholar(keys))
        exp = set((union[(union % n_s) == si] // n_s).tolist())
        assert got == exp


@pytest.mark.skipif(not HAVE_DATA, reason="needs data/")
def test_pair_check_agrees_with_candidates(engine):
    ix = engine.index()
    g = engine.gold_df.sample(50, random_state=0)
    for d_id, s_id in zip(g.dblp_id, g.scholar_id):
        di, si = engine.d_idx[d_id], engine.s_idx[s_id]
        dk = {p: ix.d_keys[p][di] for p in P.PASS_NAMES}
        sk = {p: ix.s_keys[p][si] for p in P.PASS_NAMES}
        in_cand = si in ix.candidates(dk)
        assert in_cand == any(ix.pair_check(dk, sk).values())


@pytest.mark.skipif(not HAVE_DATA, reason="needs data/")
def test_resolve_both_directions(engine):
    d_id, s_id = engine.gold_df.iloc[0][["dblp_id", "scholar_id"]]
    info, df = engine.resolve(engine.record("dblp", d_id), "dblp", top_k=1000)
    assert info["candidates"] == len(df) and s_id in set(df.scholar_id)
    assert df.composite.is_monotonic_decreasing
    info2, df2 = engine.resolve(engine.record("scholar", s_id), "scholar", top_k=1000)
    assert d_id in set(df2.dblp_id) and info2["other"] == "dblp"


@pytest.mark.skipif(not (HAVE_DATA and HAVE_DB), reason="needs data/ and the app database")
def test_portable_models_equal_sklearn_models(engine, stored):
    """outputs/*_model.json (used by the app) give the same probabilities as the scikit-learn originals."""
    joblib = pytest.importorskip("joblib")
    df, _ = stored
    rows = df.sample(2000, random_state=3)
    feats = pd.DataFrame({"title_combined": rows.title_similarity, "author_best": rows.author_similarity,
                          "venue_tri": rows.venue_similarity, "year": rows.year_similarity,
                          "len_ratio": np.linspace(0.2, 1, len(rows))})
    for base in P.MODEL_FILES.values():
        path = os.path.join(REPO, "outputs", base + ".joblib")
        if not os.path.exists(path) or not os.path.exists(os.path.join(REPO, "outputs", base + ".json")):
            pytest.skip(f"{base} not exported")
        sk = joblib.load(path)
        portable = P.load_model(os.path.join(REPO, "outputs", base))
        assert isinstance(portable["model"], P.PortableModel) and portable["threshold"] == sk["threshold"]
        np.testing.assert_allclose(P.model_probability(portable, feats), P.model_probability(sk, feats), atol=1e-6)


# ---------------------------------------------------------------- app
def test_app_builds_without_data():
    import gradio as gr
    import app
    eng = P.EntityGuard(load_data=False)
    df, meta = P.load_stored_candidates()
    demo = app.create_demo(eng, df, meta, review_csv=os.path.join(REPO, "outputs", "human_reviews.csv"))
    assert isinstance(demo, gr.Blocks)
