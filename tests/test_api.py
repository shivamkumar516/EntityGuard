import json
import os
import shutil

import pytest

from src.api.store import APP_DB_NAME, DEFAULT_OUT_DIR

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DEFAULT_OUT_DIR, APP_DB_NAME)),
                                reason="outputs/entityguard_app.db not exported yet (run the notebook)")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    csv = tmp_path_factory.mktemp("reviews") / "human_reviews.csv"
    shutil.copy(os.path.join(DEFAULT_OUT_DIR, "human_reviews.csv"), csv)
    return TestClient(create_app(review_csv=str(csv))), csv


@pytest.fixture(scope="module")
def store(env):
    from src.api.store import Store
    return Store(review_csv=str(env[1]))


def test_records(env):
    c, _ = env
    assert len(c.get("/records/dblp?limit=2").json()) == 2
    s = c.get("/records/scholar?offset=5&limit=3").json()
    assert len(s) == 3 and {"record_id", "title", "authors", "venue", "year"} <= set(s[0])


def test_pair_stored_and_by_path_with_slashes(env, store):
    c, _ = env
    d, s = store.sql("SELECT dblp_id, scholar_id FROM MATCH_RESULT WHERE scholar_id LIKE '%/%' LIMIT 1").iloc[0]
    by_query = c.get("/pair", params={"dblp_id": d, "scholar_id": s}).json()
    assert by_query["source"].startswith("stored") and 0 <= by_query["overall_similarity"] <= 1
    by_path = c.get(f"/pair/{d}/{s.replace('%', '%25')}").json()
    assert by_path["overall_similarity"] == by_query["overall_similarity"]


def test_stored_scores_equal_student1_record_similarity(store):
    """The notebook's fast pipeline and Student 1's reference code give the same score."""
    rows = store.sql("SELECT dblp_id, scholar_id, overall_similarity FROM MATCH_RESULT ORDER BY rowid LIMIT 25")
    for d, s, stored in rows.itertuples(index=False):
        live = store._score_live(store.record("dblp", d), store.record("scholar", s))
        assert live["overall_similarity"] == pytest.approx(stored, abs=1e-4)


def test_pair_not_a_candidate_is_computed(env, store):
    c, _ = env
    d = store.sql("SELECT record_id FROM DBLP_RECORD LIMIT 1").record_id[0]
    s = store.sql("SELECT record_id FROM SCHOLAR_RECORD WHERE record_id NOT IN "
                  "(SELECT scholar_id FROM MATCH_RESULT WHERE dblp_id = ?) LIMIT 1", (d,)).record_id[0]
    r = c.get("/pair", params={"dblp_id": d, "scholar_id": s}).json()
    assert r["source"].startswith("computed") and r["candidate"] is False


def test_unknown_pair_is_404(env):
    c, _ = env
    assert c.get("/pair/no/such/dblp/nothing").status_code == 404
    assert c.get("/pair", params={"dblp_id": "x", "scholar_id": "y"}).status_code == 404


def test_evaluation_summary(env):
    c, _ = env
    e = c.get("/evaluation").json()
    assert e["dataset"]["gold"] == 5347 and "human_review" in e and e["chosen"]["threshold"] > 0


def test_matching_run_reproduces_m4(env):
    c, _ = env
    summary = json.load(open(os.path.join(DEFAULT_OUT_DIR, "summary.json"), encoding="utf-8"))
    m4 = next(r for r in summary["test_results"] if r["method"].startswith("M4"))
    r = c.post("/matching/run").json()
    assert r["test"]["TP"] == m4["TP"] and r["test"]["FP"] == m4["FP"] and r["test"]["FN"] == m4["FN"]
    assert c.post("/matching/run?threshold=0.1").status_code == 400      # below the stored scores


def test_matches_limit_and_band(env):
    c, _ = env
    rows = c.get("/matches?band_name=High confidence&limit=3").json()
    assert len(rows) <= 3 and all(r["band"] == "High confidence" for r in rows)


def test_review_round_trip_and_validation(env):
    c, csv = env
    assert c.post("/review", json={"dblp_id": "x", "scholar_id": "y", "decision": "bad"}).status_code == 400
    assert c.post("/review", json={"dblp_id": "x", "scholar_id": "y", "decision": "Needs Review"}).status_code == 404
    d, s = c.get("/matches?limit=1").json()[0]["dblp_id"], c.get("/matches?limit=1").json()[0]["scholar_id"]
    r = c.post("/review", json={"dblp_id": d, "scholar_id": s, "decision": "Needs Review", "reviewer": "api-test"}).json()
    assert r["saved"] and r["reviewer"] == "api-test" and r["note"] == ""
    assert "api-test" in open(csv, encoding="utf-8").read()
    assert c.get("/pair", params={"dblp_id": d, "scholar_id": s}).json()["review"]["decision"] == "Needs Review"


def test_blocking_run_small_sample(env):
    c, _ = env
    r = c.post("/blocking/run?scholar_fraction=0.1&passes=title_prefix,rare_tokens").json()
    assert r["candidates"] > 0 and set(r["per_pass"]) == {"title_prefix", "rare_tokens"}
    assert c.post("/blocking/run?passes=nonsense").status_code == 400


def test_ui_is_mounted(env):
    c, _ = env
    assert c.get("/ui/").status_code == 200
