"""Every callback of the Gradio app (app.py), called directly, including edge cases.

The review tests write to a temporary copy of human_reviews.csv, never to outputs/.
"""
import os
import shutil

import gradio as gr
import pandas as pd
import pytest

from src import pipeline as P

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_ALL = (os.path.exists(os.path.join(REPO, "data", "Scholar.csv"))
            and os.path.exists(os.path.join(REPO, "outputs", P.APP_DB_NAME)))
pytestmark = pytest.mark.skipif(not HAVE_ALL, reason="needs data/ and outputs/entityguard_app.db")
ORDER = "Most uncertain first (LR P(match) closest to 0.5)"
ALL_PASSES = list(P.PASS_NAMES)


def _select(row):
    return gr.SelectData(None, {"index": [0, 0], "value": row[0], "row_value": row})


@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    import app
    csv = str(tmp_path_factory.mktemp("rev") / "human_reviews.csv")
    shutil.copy(os.path.join(REPO, "outputs", "human_reviews.csv"), csv)
    eng = P.EntityGuard(load_data=True)
    stored, meta = P.load_stored_candidates(weights=eng.weights, auto_t=eng.auto_t, non_t=eng.non_t)
    demo = app.create_demo(eng, stored, meta, review_csv=csv)
    return demo.eg_handlers, eng, stored, csv


def test_both_models_load_from_portable_json(app_env):
    _, eng, _, _ = app_env
    assert set(eng.models) == {"Logistic Regression", "Decision Tree"}
    assert all(isinstance(b["model"], P.PortableModel) for b in eng.models.values())


def test_tick_reports_ready(app_env):
    h, *_ = app_env
    md, _ = h["tick"]()
    assert "ready" in md and "Decision Tree" in md


def test_run_pair_typed_and_blank(app_env):
    h, *_ = app_env
    out = h["run_pair"]("", "", "An Efficient Algorithm for Entity Matching", "A. Kumar, R. Singh", "VLDB", "2005",
                        "Efficient Algorithms for Entity Matching", "Anil Kumar; Raj Singh", "Proc. VLDB", "", True)
    assert len(out) == 5 and "composite S" in out[1] and "Why" in out[3] and "would generate" in out[3]
    blank = h["run_pair"]("", "", "", "", "", "", "", "", "", "", False)
    assert "Enter two records" in blank[1]
    junk = h["run_pair"]("", "", "Data", "", "", "abc", "Data", "", "", "19x", False)   # junk years, one-word titles
    assert "composite S" in junk[1]
    unicode = h["run_pair"]("", "", "Über Datenbanken — Teil 1", "J Wäsch", "", "2001",
                            "Uber Datenbanken Teil 1", "J Wasch", "", "2001", False)
    assert "composite S" in unicode[1]


def test_load_ids_gold_label_and_errors(app_env):
    h, eng, *_ = app_env
    d, s = eng.gold_df.iloc[0][["dblp_id", "scholar_id"]]
    out = h["load_ids"](d, s, True)
    assert out[0] == d and out[1] == s and len(out) == 15 and "true match" in out[11]
    # editing a loaded field must drop the id (the typed record is no longer that record)
    edited = h["run_pair"](d, s, "Totally different title", *out[3:10], True)
    assert "Gold mapping" not in edited[1]
    with pytest.raises(gr.Error):
        h["load_ids"]("no-such-id", s, False)
    with pytest.raises(gr.Error):
        h["load_ids"](d, "", False)


def test_random_pairs(app_env):
    h, *_ = app_env
    for kind in ("gold", "hard"):
        out = h["random_pair"](kind, True)
        assert out[0] and out[1] and "composite S" in out[11]


def test_search_and_resolve_both_directions(app_env):
    h, eng, *_ = app_env
    dd, msg = h["search"]("DBLP → find Scholar matches", "association rules")
    assert "found" in msg
    none_dd, none_msg = h["search"]("DBLP → find Scholar matches", "zzzz-no-title-like-this")
    assert "No DBLP record" in none_msg
    d, s = eng.gold_df.iloc[0][["dblp_id", "scholar_id"]]
    out = h["resolve"]("DBLP → find Scholar matches", d, ALL_PASSES, 15, True)
    assert len(out) == 9 and isinstance(out[2], pd.DataFrame) and len(out[2]) <= 15
    assert "gold_match" in out[2].columns and out[3]["side"] == "dblp"
    hidden = h["resolve"]("DBLP → find Scholar matches", d, ALL_PASSES, 15, False)
    assert "gold_match" not in hidden[2].columns
    rev = h["resolve"]("Scholar → find DBLP matches", s, ["rare_tokens"], 5, True)
    assert rev[3]["side"] == "scholar" and "dblp_id" in rev[2].columns
    with pytest.raises(gr.Error):
        h["resolve"]("DBLP → find Scholar matches", d, [], 15, False)          # no pass selected
    with pytest.raises(gr.Error):
        h["resolve"]("DBLP → find Scholar matches", None, ALL_PASSES, 15, False)   # nothing picked


def test_pick_candidate_row(app_env):
    h, eng, *_ = app_env
    d = eng.gold_df.iloc[0].dblp_id
    out = h["resolve"]("DBLP → find Scholar matches", d, ALL_PASSES, 5, False)
    row = out[2].iloc[0].tolist()
    rec, dec, sc, why = h["pick_candidate"](out[3], True, _select(row))
    assert "DBLP RECORD" in rec and "composite S" in dec and "Why" in why
    assert h["pick_candidate"](None, False, _select(row)) == ("", "", "", "")


def test_random_record_both_sides(app_env):
    h, *_ = app_env
    for src in ("DBLP → find Scholar matches", "Scholar → find DBLP matches"):
        out = h["random_record"](src, ALL_PASSES, 10, False)
        assert len(out) == 10 and out[4]


def test_blocking_report_matches_notebook(app_env):
    h, *_ = app_env
    rep = h["blocking_report"](ALL_PASSES)
    union = rep.iloc[-1]
    assert union["candidates"] == 462813 and round(union["blocking_recall"], 4) == 0.9807
    with pytest.raises(gr.Error):
        h["blocking_report"]([])


def test_resolve_if_any_skips_without_query(app_env):
    h, *_ = app_env
    out = h["resolve_if_any"]("DBLP → find Scholar matches", None, ALL_PASSES, 15, True, None)
    assert len(out) == 9


def test_review_queue_label_skip_and_persist(app_env):
    h, eng, stored, csv = app_env
    before = len(P.load_reviews(csv))
    view = h["refresh"]("test", ORDER, [], False)
    assert len(view) == 9 and "unreviewed pairs" in view[1]
    cur = view[6]
    assert cur and "composite S" in view[3]
    out = h["label"]("match", cur, "pytest", "tester", "test", ORDER, [], True)
    assert out[0].startswith("Saved **Confirm Match**") and out[-1] == ""
    r = P.load_reviews(csv)
    assert len(r) == before + 1 and r.iloc[-1].decision == "Confirm Match" and r.iloc[-1].reviewer == "tester"
    assert out[7] != cur                                                 # moved on to the next pair
    nxt = out[7]
    sk = h["skip"](nxt, "test", ORDER, [], False)
    assert sk[0] == [[nxt["dblp_id"], nxt["scholar_id"]]] and sk[7] != nxt
    h["label"]("nonmatch", sk[7], "", "tester", "test", ORDER, sk[0], False)
    h["label"]("needs", None, "", "tester", "test", ORDER, [], False)   # nothing selected: no crash, no write
    assert len(P.load_reviews(csv)) == before + 2
    sel = h["select_queue_row"]("all", "Highest composite first", [], False,
                                _select(view[0].iloc[1].tolist()))
    assert sel[6]["dblp_id"] == view[0].iloc[1].dblp_id
    keep = h["refresh_current"]("all", ORDER, [], True, sel[6])
    assert keep[6] == sel[6]


def test_review_gold_label_correct_before_data_loads(app_env):
    """The review card must not claim 'not a gold match' just because data/ is still loading."""
    import app
    _, _, stored, csv = app_env
    eng = P.EntityGuard(load_data=False)
    meta = P.load_stored_candidates(weights=eng.weights, auto_t=eng.auto_t, non_t=eng.non_t)[1]
    h = app.create_demo(eng, stored, meta, review_csv=csv).eg_handlers
    gold_row = stored[(stored.in_gold == 1) & (stored.decision == P.MANUAL_REVIEW)].iloc[0]
    out = h["select_queue_row"]("all", ORDER, [], True, _select([gold_row.dblp_id, gold_row.scholar_id]))
    assert "true match" in out[3]


def test_queue_empty_is_handled(app_env, tmp_path):
    import app
    h0, eng, stored, _ = app_env
    meta = P.load_stored_candidates(weights=eng.weights, auto_t=eng.auto_t, non_t=eng.non_t)[1]
    h = app.create_demo(eng, stored.iloc[:0], meta, review_csv=str(tmp_path / "r.csv")).eg_handlers
    view = h["refresh"]("all", ORDER, [], False)
    assert view[6] is None and "Queue is empty" in view[2]
    out = h["label"]("match", None, "", "x", "all", ORDER, [], False)
    assert "queue is empty" in out[0].lower()
