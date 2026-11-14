
import argparse
import functools
import math
import os
import sys

if sys.version_info < (3, 11):
    sys.exit(f"EntityGuard needs Python 3.11 or newer; this is Python {sys.version.split()[0]}.")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # no telemetry calls (offline / firewalled laptops)

try:
    import gradio as gr  # noqa: E402
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402

    from src import pipeline as P  # noqa: E402
except ImportError as _e:
    sys.exit(f"Missing package ({_e}). Install the requirements first:\n"
             f"    {sys.executable} -m pip install -r requirements.txt")

DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
REVIEW_CSV = os.path.join(OUT_DIR, "human_reviews.csv")

DECISION_ICON = {P.AUTO_MATCH: "🟢", P.MANUAL_REVIEW: "🟡", P.NON_MATCH: "🔴"}
CSS = ".eg-scores td:not(:first-child), .eg-scores th { white-space: nowrap; }"
REVIEW_BUTTONS = {"match": "Confirm Match", "nonmatch": "Reject Match", "needs": "Needs Review"}
QUEUE_ORDERS = ("Most uncertain first (LR P(match) closest to 0.5)", "Highest composite first",
                "Lowest composite first")
FIELD_LABELS = {
    "title": "Title: S_title = α·S_edit + (1−α)·S_jaccard",
    "author": "Author: avg. best-match name edit similarity",
    "venue": "Venue: character-trigram Jaccard",
    "year": "Year: 1 same · 0.5 one apart · 0 otherwise",
}


# ====================================================================== formatting helpers
def _missing(v):
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _num(v, nd=3):
    return "—" if _missing(v) else f"{float(v):.{nd}f}"


def _cell(v):
    """Text safe for a markdown table cell."""
    if _missing(v):
        return "—"
    return str(v).replace("|", "\\|").replace("\n", " ").strip() or "—"


def _year(v):
    return "—" if _missing(v) else str(int(float(v))) if str(v).replace(".", "", 1).isdigit() else str(v)


def read_csv(name, **kw):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame({"missing file": [f"outputs/{name}"]})
    return pd.read_csv(path, **kw)


def read_text(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return f"*outputs/{name} not found.*"
    with open(path, encoding="utf-8") as f:
        return f.read()


def image(name):
    path = os.path.join(OUT_DIR, name)
    return path if os.path.exists(path) else None


def rounded(df, nd=4):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):      # float32 -> float64 first, or 0.961 shows as 0.96100002527
            out[c] = out[c].astype("float64").round(nd)
    return out


def weights_text(w):
    return (f"S = {w['title']:.2f}·S_title + {w['author']:.2f}·S_author + {w['venue']:.2f}·S_venue + "
            f"{w['year']:.2f}·S_year")


# ====================================================================== pair rendering (Section 40 layout)
def records_md(d, s):
    rows = [("ID", d.get("id"), s.get("id")), ("Title", d.get("title"), s.get("title")),
            ("Authors", d.get("authors"), s.get("authors")), ("Venue", d.get("venue"), s.get("venue")),
            ("Year", _year(d.get("year")), _year(s.get("year"))),
            ("Normalized title", d.get("t_norm"), s.get("t_norm"))]
    return ("| | **DBLP RECORD** | **SCHOLAR RECORD** |\n|---|---|---|\n" +
            "".join(f"| **{k}** | {_cell(a)} | {_cell(b)} |\n" for k, a, b in rows))


def decision_md(res, engine, show_gold, gold=None):
    dec = res["decision"]
    md = (f"## {DECISION_ICON[dec]} {dec}: composite S = {res['score']:.3f}\n"
          f"Rule: **Auto-Match** S ≥ {engine.auto_t:.2f} · **Manual Review** {engine.non_t:.2f} ≤ S < "
          f"{engine.auto_t:.2f} · **Non-Match** S < {engine.non_t:.2f}")
    if show_gold and res["dblp"].get("id") and res["scholar"].get("id"):
        gold = gold if gold is not None else engine.gold
        md += ("\n\n**Gold mapping (evaluation only, never used for the decision):** " +
               ("✅ true match" if (res["dblp"]["id"], res["scholar"]["id"]) in gold else "❌ not a gold match"))
    return md


def scores_md(res, engine):
    lines = ["| Field similarity | S_k | Weight w_k | Effective weight | Contribution |",
             "|---|---|---|---|---|"]
    for r in res["contributions"]:
        lines.append(f"| {FIELD_LABELS[r['field']]} | **{_num(r['similarity'])}** | {r['weight']:.2f} | "
                     f"{r['effective_weight']:.3f} | {r['contribution']:.3f} |")
    lines.append(f"| **Weighted composite S** | **{res['score']:.3f}** | 1.00 | | |")
    md = "\n".join(lines)
    md += (f"\n\n`{weights_text(engine.weights)}`, with α = {engine.alpha}. Missing fields are dropped and the "
           "remaining weights renormalized (Section 24).")
    if res["ml"]:
        md += "\n\n| ML classifier | P(match) | Validated threshold | Verdict |\n|---|---|---|---|\n"
        for name, m in res["ml"].items():
            p = m["probability"]
            verdict = "—" if p is None else ("match" if p >= m["threshold"] else "non-match")
            md += f"| {name} ({m['param']}) | **{_num(p)}** | {m['threshold']:.2f} | {verdict} |\n"
    else:
        md += "\n\n*No ML model found in outputs/ (lr_model.json / dt_model.json).*"
    return md


def measures_md(res):
    f = res["features"]
    rows = [("Title exact (normalized)", "title_exact"), ("Title edit similarity (Levenshtein DP)", "title_edit"),
            ("Title token Jaccard", "title_jac"), ("Title trigram Jaccard", "title_tri"),
            ("Title length ratio", "len_ratio"), ("Author token Jaccard", "author_tok"),
            ("Author best-match names", "author_best"), ("Venue exact (normalized)", "venue_exact"),
            ("Venue trigram Jaccard", "venue_tri"), ("Year", "year")]
    md = "| Measure | Value |\n|---|---|\n" + "".join(f"| {k} | {_num(f.get(c))} |\n" for k, c in rows)
    blk = res.get("blocking")
    if blk is None:
        md += "\n*Blocking check: index not ready yet (or data/ missing).*"
    else:
        md += "\n| Blocking pass | Shared keys |\n|---|---|\n" + "".join(
            f"| {P.PASS_LABELS[p]} | {_cell(', '.join(k)) if k else '— (no shared key)'} |\n" for p, k in blk.items())
    return md


def render(res, engine, show_gold=False, gold=None):
    """(records, decision, scores, explanation, all measures) markdown for one scored pair."""
    return (records_md(res["dblp"], res["scholar"]), decision_md(res, engine, show_gold, gold),
            scores_md(res, engine), "### Why\n" + res["explanation"], measures_md(res))


# ====================================================================== the application
def create_demo(engine, stored, meta, review_csv=REVIEW_CSV):
    """Build the Gradio Blocks app around an EntityGuard engine and the stored candidate table."""
    gold_db = meta.get("gold", set()) if isinstance(meta, dict) else set()

    def gold_set():
        return engine.gold if engine.data_loaded else gold_db

    # ------------------------------------------------------------------ shared
    def status_md():
        n_models = ", ".join(f"{k} ({b.get('param')})" for k, b in engine.models.items()) or "none"
        return (f"**Engine:** {engine.status()} · **Models:** {n_models} · "
                f"**Review queue source:** {len(stored):,} stored candidate pairs · "
                f"**Reviews file:** `outputs/{os.path.basename(review_csv)}`")

    def tick():
        return status_md(), gr.Timer(active=not engine.index_ready)

    # ------------------------------------------------------------------ tab 1: pair matching
    def _keep_id(source, rid, rec):
        """Keep the typed id only if the typed fields are that record (so gold / blocking keys stay valid)."""
        rid = (rid or "").strip()
        if not rid or not engine.data_loaded:
            return None
        orig = engine.record(source, rid)
        if orig is None:
            return None
        same = all(orig[k] == rec[k] for k in ("t_norm", "a_names", "v_norm", "yr"))
        return rid if same else None

    def run_pair(dblp_id, scholar_id, dt, da, dv, dy, st, sa, sv, sy, show_gold):
        d = P.prepare_record(dt, da, dv, dy)
        s = P.prepare_record(st, sa, sv, sy)
        if not d["t_norm"] and not s["t_norm"] and not d["a_names"] and not s["a_names"]:
            return ("", "### Enter two records (at least titles), or load a pair by id.", "", "", "")
        d["id"], s["id"] = _keep_id("dblp", dblp_id, d), _keep_id("scholar", scholar_id, s)
        return render(engine.score_pair(d, s), engine, show_gold, gold_set())

    def _fields(rec):
        return [rec["title"] or "", rec["authors"] or "", rec["venue"] or "",
                "" if _missing(rec["year"]) else str(rec["year"])]

    def load_ids(dblp_id, scholar_id, show_gold):
        try:
            d, s = engine.record("dblp", dblp_id), engine.record("scholar", scholar_id)
        except RuntimeError as e:
            raise gr.Error(str(e))
        if d is None or s is None:
            raise gr.Error(f"Unknown {'DBLP' if d is None else 'Scholar'} id: "
                           f"{(dblp_id if d is None else scholar_id)!r}")
        return [d["id"], s["id"], *_fields(d), *_fields(s), *render(engine.score_pair(d, s), engine, show_gold, gold_set())]

    def random_pair(kind, show_gold):
        rng = np.random.default_rng()
        if kind == "gold":
            pool = (engine.gold_df[["dblp_id", "scholar_id"]] if engine.data_loaded
                    else stored.loc[stored.in_gold == 1, ["dblp_id", "scholar_id"]])
        else:                     # look-alike non-match: high composite but not in the gold mapping
            pool = stored.loc[(stored.in_gold == 0) & (stored.composite >= 0.6), ["dblp_id", "scholar_id"]]
        if not len(pool):
            raise gr.Error("No pairs available (data/ or outputs/entityguard_app.db missing).")
        r = pool.iloc[int(rng.integers(len(pool)))]
        return load_ids(r.dblp_id, r.scholar_id, show_gold)

    # ------------------------------------------------------------------ tab 2: entity resolution
    def _side(src):
        return "dblp" if src.startswith("DBLP") else "scholar"

    def search(src, text):
        try:
            hits = engine.search(_side(src), text)
        except RuntimeError as e:
            raise gr.Error(str(e))
        choices = [(f"{r.id}: {str(r.title)[:100]}", r.id) for r in hits.itertuples()]
        if not choices:
            return gr.Dropdown(choices=[], value=None), f"No {_side(src).upper()} record matches {text!r}."
        return gr.Dropdown(choices=choices, value=choices[0][1]), f"{len(choices)} record(s) found; pick one and press Resolve."

    def resolve(src, rid, passes, top_k, show_gold):
        side = _side(src)
        if not passes:
            raise gr.Error("Select at least one blocking pass.")
        try:
            rec = engine.record(side, rid or "")
        except RuntimeError as e:
            raise gr.Error(str(e))
        if rec is None:
            raise gr.Error("Pick a record first (search by title words or id, or press Random).")
        info, df = engine.resolve(rec, side, passes, int(top_k))
        other = info["other"]
        dec = info["decisions"]
        summary = (f"**Query ({side.upper()})**: `{rec['id']}`, {_cell(rec['title'])} "
                   f"({_cell(rec['authors'])}; {_cell(rec['venue'])}; {_year(rec['year'])})\n\n"
                   f"**Multi-pass blocking** ({', '.join(info['keys'])}): {info['candidates']:,} candidates out of "
                   f"{info['other_records']:,} {other.upper()} records, a reduction ratio of "
                   f"{info['reduction_ratio']:.4%} ({info['blocking_s']*1000:.0f} ms). Scoring took "
                   f"{info['matching_s']:.2f} s.\n\n"
                   f"**Decisions:** {DECISION_ICON[P.AUTO_MATCH]} {dec[P.AUTO_MATCH]} Auto-Match · "
                   f"{DECISION_ICON[P.MANUAL_REVIEW]} {dec[P.MANUAL_REVIEW]} Manual Review · "
                   f"{DECISION_ICON[P.NON_MATCH]} {dec[P.NON_MATCH]} Non-Match")
        if show_gold:
            summary += (f"\n\n**Gold mapping (evaluation only):** {info['gold_matches']} true match(es); "
                        f"{info['gold_in_candidates']} found by blocking; {info.get('gold_auto_matched', 0)} auto-matched.")
        keys_md = "| Pass | Keys of the query record |\n|---|---|\n" + "".join(
            f"| {P.PASS_LABELS[p]} | {_cell(', '.join(k)) if k else '—'} |\n" for p, k in info["keys"].items())
        if len(df) and not show_gold:
            df = df.drop(columns=["gold_match"])
        if len(df):
            df = rounded(df.drop(columns=[f"{other}_venue"]), 3)
            first = [f"{other}_id", "composite", "decision"] + [c for c in df.columns if c.startswith("P(")]
            df = df[first + [c for c in df.columns if c not in first]]     # id first: row clicks read it
        state = {"side": side, "id": rec["id"], "other": other}
        hint = "Click a row to see its full score card and explanation." if len(df) else "No candidates."
        return summary, keys_md, df, state, hint, "", "", "", ""

    def random_record(src, passes, top_k, show_gold):
        side = _side(src)
        try:
            engine.wait_for_data()
        except RuntimeError as e:
            raise gr.Error(str(e))
        if side == "dblp":
            rid = engine.dblp["id"].iloc[int(np.random.default_rng().integers(len(engine.dblp)))]
        else:                     # a Scholar record that has a DBLP match, so the demo is informative
            ids = engine.gold_df["scholar_id"].unique()
            rid = ids[int(np.random.default_rng().integers(len(ids)))]
        rec = engine.record(side, rid)
        dd = gr.Dropdown(choices=[(f"{rid}: {str(rec['title'])[:100]}", rid)], value=rid)
        return (dd, *resolve(src, rid, passes, top_k, show_gold))

    def pick_candidate(state, show_gold, evt: gr.SelectData):
        if not state or evt.row_value is None:
            return "", "", "", ""
        other_id = str(evt.row_value[0])
        q = engine.record(state["side"], state["id"])
        o = engine.record(state["other"], other_id)
        if q is None or o is None:
            return "", "", "", ""
        d, s = (q, o) if state["side"] == "dblp" else (o, q)
        rec, dec, sc, why, _ = render(engine.score_pair(d, s), engine, show_gold, gold_set())
        return rec, dec, sc, why

    def blocking_report(passes):
        if not passes:
            raise gr.Error("Select at least one blocking pass.")
        try:
            rep = engine.blocking_report(passes)
        except RuntimeError as e:
            raise gr.Error(str(e))
        return rounded(rep, 5)

    # ------------------------------------------------------------------ tab 3: human review queue
    def reviewed_pairs():
        r = P.load_reviews(review_csv)
        return set(zip(r.dblp_id, r.scholar_id)), r

    def queue(split, order, skipped):
        done, _ = reviewed_pairs()
        q = stored[stored.decision == P.MANUAL_REVIEW]
        if split != "all":
            q = q[q.split == split]
        skip = set(map(tuple, skipped or []))
        keep = np.array([k not in done and k not in skip for k in zip(q.dblp_id, q.scholar_id)], dtype=bool)
        q = q[keep] if len(q) else q          # an empty list would select zero *columns*, so use a bool array
        if order == QUEUE_ORDERS[0]:
            q = q.assign(_u=(q.lr_probability - 0.5).abs()).sort_values(["_u", "dblp_id"])
        elif order == QUEUE_ORDERS[1]:
            q = q.sort_values(["composite", "dblp_id"], ascending=[False, True])
        else:
            q = q.sort_values(["composite", "dblp_id"])
        return q

    def queue_view(q):
        shown = q.head(50)[["dblp_id", "scholar_id", "composite", "lr_probability", "dblp_title", "scholar_title",
                            "split"]].rename(columns={"composite": "S (composite)", "lr_probability": "P(LR)"})
        info = (f"**{len(q):,} unreviewed pairs** in the Manual Review band ({engine.non_t:.2f} ≤ S < "
                f"{engine.auto_t:.2f}); the first 50 are listed. Click a row to review it.")
        return rounded(shown, 3), info

    def review_stats():
        _, r = reviewed_pairs()
        gold = gold_set()
        counts = r.decision.value_counts()
        dec = r[r.decision != "Needs Review"]
        agree = (((dec.decision == "Confirm Match").to_numpy() ==
                  np.array([(a, b) in gold for a, b in zip(dec.dblp_id, dec.scholar_id)], dtype=bool)).mean()
                 if len(dec) else float("nan"))
        md = (f"**{len(r)} saved decisions**: {counts.get('Confirm Match', 0)} Match · "
              f"{counts.get('Reject Match', 0)} Non-Match · {counts.get('Needs Review', 0)} Needs Review · "
              f"agreement of Match/Non-Match decisions with the gold mapping: "
              f"**{'—' if math.isnan(agree) else f'{agree:.1%}'}**")
        log = r.iloc[::-1].head(25).reset_index(drop=True)
        return md, log

    def show_pair(dblp_id, scholar_id, show_gold):
        hit = stored[(stored.dblp_id == dblp_id) & (stored.scholar_id == scholar_id)]
        if not len(hit):
            return ("*Queue is empty.*", "", "", "", None)
        row = hit.iloc[0]
        d, s = P.stored_record(row, "dblp"), P.stored_record(row, "scholar")
        rec, dec, sc, why, _ = render(engine.score_pair(d, s), engine, show_gold, gold_set())
        return rec, dec, sc, why, {"dblp_id": dblp_id, "scholar_id": scholar_id}

    def refresh(split, order, skipped, show_gold, current=None):
        q = queue(split, order, skipped)
        qdf, qinfo = queue_view(q)
        if current is None:
            current = (q.iloc[0].dblp_id, q.iloc[0].scholar_id) if len(q) else (None, None)
        card = show_pair(*current, show_gold)
        stats, log = review_stats()
        return (qdf, qinfo, *card, stats, log)

    def label(kind, cur, note, reviewer, split, order, skipped, show_gold):
        if not cur:
            return ("Nothing to label: the queue is empty.", *refresh(split, order, skipped, show_gold), note)
        decision = REVIEW_BUTTONS[kind]
        try:
            P.save_review(review_csv, cur["dblp_id"], cur["scholar_id"], decision, note, reviewer)
        except OSError as e:          # e.g. the CSV is open in Excel on Windows
            raise gr.Error(f"Could not write {review_csv}: {e}. Close the file if it is open elsewhere.")
        msg = (f"Saved **{decision}** for `{cur['dblp_id']}` / `{cur['scholar_id']}` → "
               f"`outputs/{os.path.basename(review_csv)}`")
        return (msg, *refresh(split, order, skipped, show_gold), "")

    def skip(cur, split, order, skipped, show_gold):
        skipped = list(skipped or [])
        if cur:
            skipped.append([cur["dblp_id"], cur["scholar_id"]])
        return (skipped, *refresh(split, order, skipped, show_gold))

    def select_queue_row(split, order, skipped, show_gold, evt: gr.SelectData):
        if evt.row_value is None:
            return refresh(split, order, skipped, show_gold)
        return refresh(split, order, skipped, show_gold, current=(str(evt.row_value[0]), str(evt.row_value[1])))

    def refresh_current(split, order, skipped, show_gold, cur):
        """Re-render the queue without moving away from the pair being reviewed."""
        current = (cur["dblp_id"], cur["scholar_id"]) if cur else None
        return refresh(split, order, skipped, show_gold, current=current)

    def resolve_if_any(src, rid, passes, top_k, show_gold, state):
        """Re-run the last query (e.g. when the gold checkbox changes); do nothing if there is none."""
        if not state or not rid or not passes:
            return tuple(gr.skip() for _ in range(9))
        return resolve(src, rid, passes, top_k, show_gold)

    # ------------------------------------------------------------------ static dashboard content
    summary = engine.summary
    chosen = summary.get("chosen", {})
    w_val = chosen.get("weights_title_author_venue_year", [None] * 4)
    config_df = pd.DataFrame([
        {"setting": "weights (title, author, venue, year)",
         "validation choice (outputs/summary.json)": " / ".join(_num(x, 2) for x in w_val),
         "this app": " / ".join(f"{engine.weights[k]:.2f}" for k in P.FIELDS)},
        {"setting": "α in S_title = α·S_edit + (1−α)·S_jaccard",
         "validation choice (outputs/summary.json)": str(chosen.get("alpha", "—")), "this app": str(engine.alpha)},
        {"setting": "author / venue measure",
         "validation choice (outputs/summary.json)": f"{chosen.get('author_measure', '—')} / {chosen.get('venue_measure', '—')}",
         "this app": f"{P.AUTHOR_MEASURE} / {P.VENUE_MEASURE}"},
        {"setting": "decision boundaries",
         "validation choice (outputs/summary.json)":
             f"match if S ≥ {chosen.get('threshold', '—')}; bands ≥ {chosen.get('band_high', '—')} high confidence, "
             f"< {chosen.get('band_low', '—')} likely non-match",
         "this app": f"Auto-Match ≥ {engine.auto_t:.2f}; Manual Review {engine.non_t:.2f}–{engine.auto_t:.2f}; "
                     f"Non-Match < {engine.non_t:.2f}"},
    ])
    rule_df = rounded(P.decision_rule_check(stored, meta), 4) if len(stored) else pd.DataFrame(
        {"note": [meta.get("error", "no stored candidates")]})
    ml_df = pd.DataFrame([{"model": k, "hyper-parameter": b.get("param"), "validation F1": round(b.get("val_f1", np.nan), 4),
                           "probability threshold (val)": b.get("threshold")} for k, b in engine.models.items()])
    errors = read_csv("error_analysis_40_cases.csv", dtype=str, keep_default_na=False)
    err_cols = [c for c in ["kind", "title_d", "title_s", "author_d", "author_s", "year_d", "year_s", "S_title",
                            "S_author", "score", "in_candidates", "auto_causes", "manual_cause", "manual_note"]
                if c in errors.columns]
    if "manual_cause" in errors.columns:
        causes = (errors.assign(primary_cause=errors.manual_cause.str.split("; ").str[0]
                                        .replace("", "(not labelled by hand)"))
                  .groupby(["kind", "primary_cause"]).size().rename("cases").reset_index()
                  .sort_values(["kind", "cases"], ascending=[True, False]))
    else:
        causes = pd.DataFrame()
    bullets = "\n".join(f"- {line}" for line in read_text("results_summary.txt").splitlines() if line.strip())
    ds = summary.get("dataset", {})

    def img(name, label):
        return gr.Image(value=image(name), label=label, type="filepath", interactive=False,
                        buttons=["download", "fullscreen"])

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title="EntityGuard") as demo:
        gr.Markdown("# EntityGuard: Entity Resolution across DBLP and Google Scholar\n"
                    "Normalization → multi-pass blocking → field similarities → weighted score → "
                    "Logistic Regression / Decision Tree → decision → human review.")
        status = gr.Markdown(status_md())
        timer = gr.Timer(1.5)
        timer.tick(tick, outputs=[status, timer])
        show_gold = gr.Checkbox(value=False, label="Show gold-mapping labels (evaluation only; they never "
                                                   "influence a decision)")

        # ---------------- tab 1
        with gr.Tab("1 · Pair Matching"):
            gr.Markdown("Compare a **DBLP** record with a **Scholar** record. Type the fields, or load a pair by id. "
                        "The pair is normalized, compared field by field, scored, classified, and checked against "
                        "the multi-pass blocking.")
            with gr.Row():
                p_did = gr.Textbox(label="DBLP id", placeholder="e.g. conf/sigmod/AgrawalS94", scale=3)
                p_sid = gr.Textbox(label="Scholar id", scale=3)
                p_load = gr.Button("Load by ids", scale=1)
                p_rand_g = gr.Button("Random gold pair", scale=1)
                p_rand_h = gr.Button("Random look-alike non-match", scale=1)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**DBLP record**")
                    d_t = gr.Textbox(label="Title", value="An Efficient Algorithm for Entity Matching")
                    d_a = gr.Textbox(label="Authors", value="A. Kumar, R. Singh")
                    d_v = gr.Textbox(label="Venue", value="VLDB")
                    d_y = gr.Textbox(label="Year", value="2005")
                with gr.Column():
                    gr.Markdown("**Scholar record**")
                    s_t = gr.Textbox(label="Title", value="Efficient Algorithms for Entity Matching")
                    s_a = gr.Textbox(label="Authors", value="Anil Kumar; Raj Singh")
                    s_v = gr.Textbox(label="Venue", value="Proc. VLDB")
                    s_y = gr.Textbox(label="Year", value="")
            p_run = gr.Button("Match these two records", variant="primary")
            p_dec = gr.Markdown()
            with gr.Row():
                p_scores = gr.Markdown(elem_classes="eg-scores", min_width=560)
                p_why = gr.Markdown()
            p_rec = gr.Markdown()
            with gr.Accordion("All similarity measures and blocking keys", open=False):
                p_meas = gr.Markdown()
            pair_out = [p_rec, p_dec, p_scores, p_why, p_meas]
            fields_in = [d_t, d_a, d_v, d_y, s_t, s_a, s_v, s_y]
            p_run.click(run_pair, [p_did, p_sid, *fields_in, show_gold], pair_out)
            loaded_out = [p_did, p_sid, *fields_in, *pair_out]
            p_load.click(load_ids, [p_did, p_sid, show_gold], loaded_out)
            p_rand_g.click(lambda g: random_pair("gold", g), [show_gold], loaded_out)
            p_rand_h.click(lambda g: random_pair("hard", g), [show_gold], loaded_out)

        # ---------------- tab 2
        with gr.Tab("2 · Entity Resolution (query)"):
            gr.Markdown("Pick one record from `data/DBLP.csv` or `data/Scholar.csv`. Multi-pass blocking selects "
                        "candidates from the other source; each candidate is then scored and classified.")
            with gr.Row():
                q_src = gr.Radio(["DBLP → find Scholar matches", "Scholar → find DBLP matches"],
                                 value="DBLP → find Scholar matches", label="Query source", scale=2)
                q_text = gr.Textbox(label="Search title words or exact id", placeholder="e.g. association rules",
                                    scale=3)
                q_search = gr.Button("Search", scale=1)
                q_rand = gr.Button("Random record", scale=1)
            with gr.Row():
                q_pick = gr.Dropdown(label="Record", choices=[], allow_custom_value=True, scale=4)
                q_k = gr.Slider(5, 100, value=15, step=5, label="Show top-k candidates", scale=2)
            q_passes = gr.CheckboxGroup(choices=[(P.PASS_LABELS[p], p) for p in P.PASS_NAMES],
                                        value=list(P.PASS_NAMES), label="Blocking configuration (passes in the union)")
            q_run = gr.Button("Resolve", variant="primary")
            q_msg = gr.Markdown()
            q_sum = gr.Markdown()
            with gr.Accordion("Blocking keys of the query record", open=False):
                q_keys = gr.Markdown()
            q_df = gr.Dataframe(label="Ranked candidates", interactive=False, max_height=420, max_chars=60)
            q_state = gr.State(None)
            q_dec = gr.Markdown()
            with gr.Row():
                q_scores = gr.Markdown(elem_classes="eg-scores", min_width=560)
                q_why = gr.Markdown()
            q_rec = gr.Markdown()
            with gr.Accordion("Run the blocking configuration on all 2,616 DBLP records "
                              "(candidates, reduction ratio, blocking recall)", open=False):
                b_run = gr.Button("Run multi-pass blocking")
                b_df = gr.Dataframe(interactive=False)
            res_out = [q_sum, q_keys, q_df, q_state, q_msg, q_rec, q_dec, q_scores, q_why]
            q_search.click(search, [q_src, q_text], [q_pick, q_msg])
            q_text.submit(search, [q_src, q_text], [q_pick, q_msg])
            q_run.click(resolve, [q_src, q_pick, q_passes, q_k, show_gold], res_out)
            q_rand.click(random_record, [q_src, q_passes, q_k, show_gold], [q_pick, *res_out])
            q_df.select(pick_candidate, [q_state, show_gold], [q_rec, q_dec, q_scores, q_why])
            b_run.click(blocking_report, [q_passes], [b_df])

        # ---------------- tab 3
        with gr.Tab("3 · Human Review Queue"):
            gr.Markdown(f"Pairs that blocking produced, with composite score {engine.non_t:.2f} ≤ S < "
                        f"{engine.auto_t:.2f}, are not merged automatically. Label them below; every decision is "
                        f"written to `outputs/{os.path.basename(review_csv)}` right away.")
            with gr.Row():
                r_split = gr.Dropdown(["all", "train", "val", "test"], value="all", label="Split", scale=1)
                r_order = gr.Dropdown(list(QUEUE_ORDERS), value=QUEUE_ORDERS[0], label="Order", scale=3)
                r_refresh = gr.Button("Refresh queue", scale=1)
            r_info = gr.Markdown()
            r_df = gr.Dataframe(interactive=False, max_height=300, label="Review queue", max_chars=60)
            r_cur, r_skipped = gr.State(None), gr.State([])
            r_dec = gr.Markdown()
            r_rec = gr.Markdown()
            with gr.Row():
                r_scores = gr.Markdown(elem_classes="eg-scores", min_width=560)
                r_why = gr.Markdown()
            with gr.Row():
                r_reviewer = gr.Textbox(label="Reviewer", value="student3", scale=1)
                r_note = gr.Textbox(label="Note (optional)", scale=3)
            with gr.Row():
                b_match = gr.Button("✅ Match (Confirm Match)", variant="primary")
                b_non = gr.Button("❌ Non-Match (Reject Match)", variant="stop")
                b_needs = gr.Button("⏸ Needs Review")
                b_skip = gr.Button("Skip for now")
            r_msg = gr.Markdown()
            r_stats = gr.Markdown()
            r_log = gr.Dataframe(label="Latest decisions (outputs/human_reviews.csv)", interactive=False, wrap=True)
            view_out = [r_df, r_info, r_rec, r_dec, r_scores, r_why, r_cur, r_stats, r_log]
            ctl = [r_split, r_order, r_skipped, show_gold]
            for b, kind in ((b_match, "match"), (b_non, "nonmatch"), (b_needs, "needs")):
                b.click(functools.partial(label, kind), [r_cur, r_note, r_reviewer, *ctl], [r_msg, *view_out, r_note])
            b_skip.click(skip, [r_cur, *ctl], [r_skipped, *view_out])
            r_refresh.click(refresh, ctl, view_out)
            r_split.change(refresh, ctl, view_out)
            r_order.change(refresh, ctl, view_out)
            r_df.select(select_queue_row, ctl, view_out)
            demo.load(refresh, ctl, view_out)

        # ---------------- tab 4
        with gr.Tab("4 · Evaluation Dashboard"):
            gr.Markdown("Precomputed by `experiments/EntityGuard_Student3.ipynb` and loaded from `outputs/`. "
                        "All numbers are on the untouched **test** split; thresholds and weights were chosen on "
                        "validation.")
            with gr.Tab("Results summary"):
                gr.Markdown(bullets)
                gr.Markdown("**Configuration: validation choice vs. this app's decision rule**")
                gr.Markdown("| " + " | ".join(config_df.columns) + " |\n|" + "---|" * len(config_df.columns) + "\n" +
                            "".join("| " + " | ".join(_cell(v) for v in row) + " |\n"
                                    for row in config_df.itertuples(index=False)))
                gr.Markdown("**This app's decision rule on the test split** (stored candidates; gold mapping used "
                            "for evaluation only). Non-Match counts the true matches the rule does not surface, "
                            "including those blocking missed.")
                gr.Dataframe(rule_df, interactive=False)
            with gr.Tab("Method comparison"):
                gr.Dataframe(rounded(read_csv("method_comparison.csv")), interactive=False, wrap=True)
                img("method_comparison.png", "Precision / recall / F1 per method (test split)")
            with gr.Tab("Blocking"):
                gr.Dataframe(rounded(read_csv("blocking_passes.csv"), 5), interactive=False)
                img("blocking_tradeoff.png", "Blocking trade-off: candidates vs. blocking recall")
                gr.Markdown("**Scalability (Section 43)**")
                gr.Dataframe(rounded(read_csv("scalability.csv")), interactive=False)
                img("scalability.png", "Scalability on 10–100 % of Scholar")
            with gr.Tab("ML models & thresholds"):
                gr.Dataframe(ml_df, interactive=False)
                img("ml_models.png", "Logistic Regression coefficients and Decision Tree")
                gr.Markdown("**Mandatory thresholds on the test split (Section 33)**")
                gr.Dataframe(rounded(read_csv("threshold_sweep_test.csv")), interactive=False)
                img("threshold_curves.png", "Threshold vs precision / recall / F1")
                img("weight_sensitivity.png", "Sensitivity of validation F1 to each weight")
            with gr.Tab("Error analysis"):
                gr.Markdown("20 false positives and 20 false negatives of Method 4 on the test split, each read by "
                            "hand (Section 45).")
                gr.Dataframe(causes, interactive=False, label="Primary manual cause")
                img("error_causes.png", "Error causes")
                gr.Dataframe(errors[err_cols], interactive=False, max_height=500, max_chars=60,
                             label="outputs/error_analysis_40_cases.csv")
            with gr.Tab("Similarity measures"):
                for name, title in (("compare_title_measures.csv", "Title measures"),
                                    ("compare_author_measures.csv", "Author measures"),
                                    ("compare_venue_measures.csv", "Venue measures")):
                    gr.Markdown(f"**{title}** (each measure used alone; threshold chosen on validation)")
                    gr.Dataframe(rounded(read_csv(name)), interactive=False)

        # ---------------- tab 5
        with gr.Tab("5 · Dataset Statistics"):
            if ds:
                gr.Markdown(f"**Search space:** |DBLP| × |Scholar| = {ds.get('dblp', 0):,} × {ds.get('scholar', 0):,} "
                            f"= **{ds.get('cartesian_pairs', 0):,}** cross-source pairs; "
                            f"**{ds.get('gold', 0):,}** gold matches.")
            gr.Dataframe(read_csv("dataset_statistics.csv"), interactive=False)
            img("year_distribution.png", "Records per year")
            with gr.Row():
                gr.Dataframe(read_csv("common_venues.csv"), interactive=False, label="Most common venues")
                gr.Dataframe(read_csv("punctuation_frequency.csv"), interactive=False, label="Punctuation in titles")

        demo.load(run_pair, [p_did, p_sid, *fields_in, show_gold], pair_out)
        # the gold checkbox re-renders what is on screen in every tab
        show_gold.change(run_pair, [p_did, p_sid, *fields_in, show_gold], pair_out)
        show_gold.change(resolve_if_any, [q_src, q_pick, q_passes, q_k, show_gold, q_state], res_out)
        show_gold.change(refresh_current, [*ctl, r_cur], view_out)

    # handlers, so tests can call every callback directly (tests/test_app.py)
    demo.eg_handlers = {f.__name__: f for f in (
        run_pair, load_ids, random_pair, search, resolve, random_record, pick_candidate, blocking_report,
        queue, refresh, label, skip, select_queue_row, refresh_current, resolve_if_any, review_stats, tick)}
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser(description="EntityGuard web app")
    # Hosting platforms set GRADIO_SERVER_NAME / GRADIO_SERVER_PORT (Hugging Face Spaces) or PORT (Render etc.)
    env_port = os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT")
    env_host = os.environ.get("GRADIO_SERVER_NAME") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    ap.add_argument("--host", default=env_host, help="use 0.0.0.0 to serve on the network")
    ap.add_argument("--port", type=int, default=int(env_port) if env_port else None,
                    help="default: 7860 or the next free port")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args(argv)

    required = ["data/Scholar.csv", "data/DBLP-Scholar_perfectMapping.csv", "outputs/summary.json",
                "outputs/entityguard_app.db", "outputs/lr_model.json", "outputs/dt_model.json"]
    missing = [p for p in required if not os.path.exists(os.path.join(ROOT, p))]
    if not any(os.path.exists(os.path.join(DATA_DIR, n)) for n in ("DBLP.csv", "DBLP1.csv")):
        missing.insert(0, "data/DBLP.csv")
    if missing:
        print("WARNING: missing files (the related features will be unavailable): " + ", ".join(missing))

    engine = P.EntityGuard(DATA_DIR, OUT_DIR, load_data="background")     # data + blocking index load in background
    stored, meta = P.load_stored_candidates(OUT_DIR, engine.weights, engine.auto_t, engine.non_t)
    print(f"EntityGuard: {len(engine.models)} model(s), {len(stored):,} stored candidate pairs; "
          "loading data/ and the blocking index in the background.")
    demo = create_demo(engine, stored, meta)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=args.open,
                allowed_paths=[OUT_DIR], css=CSS)


if __name__ == "__main__":
    main()
