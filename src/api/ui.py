"""Gradio interface (Section 46 layout, Section 40 review screen), built only from a Store.

Tabs: Dataset Statistics · Blocking Configuration · Run Matching · Candidate Pairs ·
      Review Pair · Evaluation · Performance Dashboard
"""
import os
import re

import gradio as gr
import pandas as pd

from ..evaluation import BAND_NAMES
from ..matching.blocking_reference import PASS_NAMES

ERROR_QUEUES = ("Error analysis: false positives", "Error analysis: false negatives")


def _csv(store, name):
    p = os.path.join(store.out_dir, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame({"missing file": [name]})


def _img(store, name):
    p = os.path.join(store.out_dir, name)
    return p if os.path.exists(p) else None


def _fmt(v, digits=2):
    if v is None or (isinstance(v, float) and v != v):
        return "— (missing)"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def pair_card(store, dblp_id, scholar_id, show_gold=False):
    """(records markdown, scores markdown) for the review screen."""
    p = store.pair((dblp_id or "").strip(), (scholar_id or "").strip())
    if p is None:
        return "**Pair not found.** Check the ids.", ""
    a, b = p["dblp"], p["scholar"]
    rec = "| | **DBLP record** | **Scholar record** |\n|---|---|---|\n" + "".join(
        f"| **{label}** | {_fmt(a[k]) if a[k] is not None else '—'} | {_fmt(b[k]) if b[k] is not None else '—'} |\n"
        for label, k in (("ID", "record_id"), ("Title", "title"), ("Authors", "authors"), ("Venue", "venue"),
                         ("Year", "year")))
    sc = ("| Similarity | Score |\n|---|---|\n"
          f"| Title Similarity | **{_fmt(p['title_similarity'])}** |\n"
          f"| Author Similarity | **{_fmt(p['author_similarity'])}** |\n"
          f"| Venue Similarity | **{_fmt(p['venue_similarity'])}** |\n"
          f"| Year Similarity | **{_fmt(p['year_similarity'])}** |\n"
          f"| **Overall Score** | **{_fmt(p['overall_similarity'])}** (threshold {p['threshold']}) |\n"
          f"| Logistic Regression P(match) | {_fmt(p['lr_probability'])} |\n"
          f"| Confidence band | {p['band']} |\n"
          f"| Prediction | {'match' if p['prediction'] else 'non-match'} |\n")
    sc += f"\n*Scores: {p['source']}.*"
    if show_gold:
        sc += f"\n\n**Gold mapping (evaluation only):** {'✅ true match' if (a['record_id'], b['record_id']) in store.gold else '❌ not a match'}"
    if p["review"]:
        r = p["review"]
        sc += f"\n\n*Already reviewed:* **{r['decision']}** by {r['reviewer'] or '—'}{(' — ' + r['note']) if r['note'] else ''}"
    return rec, sc


def build_ui(store):
    cfg, s = store.cfg, store.summary
    blk = s.get("blocking", {})
    errors = _csv(store, "error_analysis_40_cases.csv")

    def error_queue(kind):
        if "kind" not in errors:
            return []
        k = "FP" if kind.endswith("positives") else "FN"
        done = set(zip(*[store.sql("SELECT dblp_id, scholar_id FROM REVIEW")[c] for c in ("dblp_id", "scholar_id")]))
        return [(d, sc) for d, sc in zip(errors[errors.kind == k].dblp_id, errors[errors.kind == k].scholar_id)
                if (d, sc) not in done]

    def next_pair(kind, show_gold):
        if kind in ERROR_QUEUES:
            q = error_queue(kind)
            n = len(q)
            first = q[0] if q else None
        else:
            df = store.review_queue(kind, limit=1)
            n = int(store.sql("SELECT COUNT(*) AS n FROM MATCH_RESULT m LEFT JOIN REVIEW r ON r.dblp_id=m.dblp_id "
                              "AND r.scholar_id=m.scholar_id WHERE m.band=? AND r.dblp_id IS NULL", (kind,)).n[0])
            first = (df.dblp_id[0], df.scholar_id[0]) if len(df) else None
        if first is None:
            return "🎉 Nothing left to review in this queue.", "", "", "", f"0 pairs left in '{kind}'"
        rec, sc = pair_card(store, first[0], first[1], show_gold)
        return rec, sc, first[0], first[1], f"{n:,} unreviewed pairs left in '{kind}'"

    def decide(decision, d, sc, note, reviewer, kind, show_gold):
        msg = ""
        if d and sc:
            try:
                store.save_review(d, sc, decision, note, reviewer)
            except (ValueError, KeyError) as e:
                msg = f"⚠️ not saved: {e}"
        rec, card, nd, ns, status = next_pair(kind, show_gold)
        return rec, card, nd, ns, (msg + "  \n" + status) if msg else status, ""

    def lookup(d, sc, show_gold):
        rec, card = pair_card(store, d, sc, show_gold)
        return rec, card, (d or "").strip(), (sc or "").strip()

    def reviews_table():
        r, st = store.reviews()
        agree = "—" if st["agreement_with_gold"] is None else f"{st['agreement_with_gold']:.1%}"
        return r, (f"**{st['reviews']}** reviews · Confirm {st['Confirm Match']} · Reject {st['Reject Match']} · "
                   f"Needs review {st['Needs Review']}  \nAgreement of human decisions with the gold mapping: **{agree}**")

    def candidates(text, band, limit):
        text = (text or "").strip().lower()
        ids = None
        if text:
            d = store.sql("SELECT record_id FROM DBLP_RECORD WHERE lower(title) LIKE ? OR lower(record_id) = ? LIMIT 20",
                          (f"%{text}%", text))
            ids = d.record_id.tolist()
        df = store.matches(None if band == "All" else band, int(limit), dblp_ids=ids)
        return df.round(3)

    def run_matching(threshold):
        try:
            r = store.run_matching(threshold)
        except ValueError as e:
            return f"⚠️ {e}", pd.DataFrame()
        m = r["test"]
        md = (f"**T = {r['threshold']:.2f}** → {r['predicted_matches']:,} predicted matches "
              f"({r['predicted_matches_test']:,} in the test split). On the **test split**: precision **{m['precision']:.3f}**, "
              f"recall **{m['recall']:.3f}**, F1 **{m['f1']:.3f}** (TP {m['TP']}, FP {m['FP']}, FN {m['FN']}).  \n"
              f"The operating threshold chosen on validation is **{cfg.threshold}**.")
        top = store.matches(None, 30, min_score=threshold).round(3)
        return md, top

    def run_blocking(fraction, passes):
        if not passes:
            return "⚠️ choose at least one pass", pd.DataFrame()
        try:
            r = store.run_blocking(fraction / 100, passes)
        except Exception as e:                           # e.g. dataset files missing
            return f"⚠️ {e}", pd.DataFrame()
        md = (f"**{r['candidates']:,}** candidate pairs out of {r['cartesian_pairs']:,} "
              f"({r['scholar_records']:,} Scholar records) · reduction ratio **{r['reduction_ratio']:.4%}** · "
              f"blocking recall **{r['blocking_recall']:.2%}** · {r['seconds']} s")
        return md, pd.DataFrame({"pass": list(r["per_pass"]), "candidates": list(r["per_pass"].values())})

    with gr.Blocks(title="EntityGuard") as demo:
        gr.Markdown(f"# 🛡️ EntityGuard\nEntity resolution between **DBLP** ({store.n_dblp:,}) and **Google Scholar** "
                    f"({store.n_scholar:,}) · {store.n_dblp * store.n_scholar:,} possible pairs → "
                    f"**{blk.get('candidates', store.n_candidates):,}** candidates after blocking")

        with gr.Tab("Dataset Statistics"):
            gr.Dataframe(_csv(store, "dataset_statistics.csv").astype(str), label="Profile")
            gr.Image(_img(store, "year_distribution.png"), show_label=False)
            gr.Dataframe(_csv(store, "common_venues.csv").astype(str), label="Most common venues")

        with gr.Tab("Blocking Configuration"):
            gr.Markdown(f"Saved run (all passes, 100 % of Scholar): **{blk.get('candidates', 0):,}** candidates · "
                        f"reduction ratio **{blk.get('reduction_ratio', 0):.4%}** · blocking recall "
                        f"**{blk.get('blocking_recall', 0):.2%}**. Re-run with another configuration below "
                        "(reference blocking; the first run loads the dataset, ~5 s).")
            with gr.Row():
                b_passes = gr.CheckboxGroup(list(PASS_NAMES), value=list(PASS_NAMES), label="Blocking passes")
                b_frac = gr.Slider(10, 100, value=100, step=5, label="% of Scholar records")
            b_run = gr.Button("Run blocking", variant="primary")
            b_out = gr.Markdown(); b_tbl = gr.Dataframe()
            b_run.click(run_blocking, [b_frac, b_passes], [b_out, b_tbl])
            gr.Dataframe(_csv(store, "blocking_passes.csv").round(5).astype(str), label="Saved run: passes and their union")
            gr.Image(_img(store, "blocking_tradeoff.png"), show_label=False)

        with gr.Tab("Run Matching"):
            gr.Markdown(f"Weighted score S = {cfg.weights[0]}·title + {cfg.weights[1]}·author + {cfg.weights[2]}·venue + "
                        f"{cfg.weights[3]}·year (α = {cfg.alpha}, missing fields renormalized). "
                        "Choose a threshold; precision / recall / F1 are measured on the test split.")
            m_T = gr.Slider(max(store.min_score, 0.40), 0.99, value=cfg.threshold, step=0.01, label="Threshold T")
            m_run = gr.Button("Run matching", variant="primary")
            m_out = gr.Markdown(); m_tbl = gr.Dataframe(label="Highest-scoring predicted matches")
            m_run.click(run_matching, m_T, [m_out, m_tbl])
            gr.Image(_img(store, "threshold_curves.png"), show_label=False)

        with gr.Tab("Candidate Pairs"):
            gr.Markdown(f"Stored candidate pairs with score ≥ {store.min_score} "
                        f"({store.sql('SELECT COUNT(*) n FROM MATCH_RESULT').n[0]:,} of {store.n_candidates:,} candidates).")
            with gr.Row():
                c_text = gr.Textbox(label="DBLP title contains / DBLP id")
                c_band = gr.Dropdown(["All", *BAND_NAMES], value="All", label="Band")
                c_lim = gr.Slider(10, 500, value=50, step=10, label="Rows")
            c_btn = gr.Button("Show candidates")
            c_out = gr.Dataframe()
            c_btn.click(candidates, [c_text, c_band, c_lim], c_out)
            c_text.submit(candidates, [c_text, c_band, c_lim], c_out)

        with gr.Tab("Review Pair"):
            with gr.Row():
                kind = gr.Dropdown([BAND_NAMES[1], BAND_NAMES[0], BAND_NAMES[2], *ERROR_QUEUES],
                                   value=BAND_NAMES[1], label="Review queue")
                show_gold = gr.Checkbox(False, label="Show gold label (evaluation only)")
                reviewer = gr.Textbox("student3", label="Reviewer")
            status = gr.Markdown()
            gr.HTML("<style>.eg-card table td:first-child, .eg-card table th:first-child {white-space: nowrap;}</style>")
            with gr.Row():
                rec_md = gr.Markdown(elem_classes=["eg-card"])
                score_md = gr.Markdown(elem_classes=["eg-card"])
            cur_d, cur_s = gr.Textbox(visible=False), gr.Textbox(visible=False)
            note = gr.Textbox(label="Note (optional)", placeholder="e.g. journal version of the conference paper")
            with gr.Row():
                b_yes = gr.Button("✅ Confirm Match", variant="primary")
                b_no = gr.Button("❌ Reject Match", variant="stop")
                b_maybe = gr.Button("🤔 Needs Review")
                b_skip = gr.Button("⏭ Next pair")
            with gr.Accordion("Look up any pair", open=False):
                with gr.Row():
                    q_d = gr.Textbox(label="DBLP id"); q_s = gr.Textbox(label="Scholar id")
                gr.Button("Show pair").click(lookup, [q_d, q_s, show_gold], [rec_md, score_md, cur_d, cur_s])
            outs = [rec_md, score_md, cur_d, cur_s, status]
            b_skip.click(next_pair, [kind, show_gold], outs)
            kind.change(next_pair, [kind, show_gold], outs)
            for btn, dec in ((b_yes, "Confirm Match"), (b_no, "Reject Match"), (b_maybe, "Needs Review")):
                btn.click(lambda d, sc, n, r, k, g, _dec=dec: decide(_dec, d, sc, n, r, k, g),
                          [cur_d, cur_s, note, reviewer, kind, show_gold], outs + [note])
            demo.load(next_pair, [kind, show_gold], outs)
            with gr.Accordion("All reviews", open=False):
                rev_msg = gr.Markdown(); rev_tbl = gr.Dataframe()
                gr.Button("Refresh").click(reviews_table, None, [rev_tbl, rev_msg])
                demo.load(reviews_table, None, [rev_tbl, rev_msg])

        with gr.Tab("Evaluation"):
            gr.Markdown(f"Chosen on validation: α = **{cfg.alpha}**, weights (title, author, venue, year) = "
                        f"**{cfg.weights}**, author measure `{cfg.author_measure}`, venue measure `{cfg.venue_measure}`, "
                        f"threshold **T = {cfg.threshold}**, bands **{cfg.band_low} / {cfg.band_high}**. "
                        "All metrics below are on the untouched test split.")
            gr.Dataframe(_csv(store, "method_comparison.csv").round(4).astype(str), label="Method comparison (Section 39)")
            gr.Image(_img(store, "method_comparison.png"), show_label=False)
            gr.Dataframe(_csv(store, "threshold_sweep_test.csv").round(4).astype(str), label="Mandatory thresholds (Section 33)")
            gr.Dataframe(_csv(store, "compare_title_measures.csv").round(4).astype(str), label="Title measures (Section 44)")
            gr.Dataframe(_csv(store, "compare_author_measures.csv").round(4).astype(str), label="Author measures")
            gr.Dataframe(_csv(store, "student1_settings_in_pipeline.csv").round(4).astype(str),
                         label="Student 1's settings in the full pipeline")
            gr.Image(_img(store, "student1_vs_pipeline.png"), show_label=False)
            gr.Image(_img(store, "ml_models.png"), show_label=False)
            gr.Image(_img(store, "error_causes.png"), show_label=False)

        with gr.Tab("Performance Dashboard"):
            gr.Markdown(f"Blocking time **{blk.get('time_s', 0):.1f} s** for {store.n_dblp * store.n_scholar:,} possible pairs. "
                        "Scalability on 10–100 % of Scholar (Section 43):")
            gr.Dataframe(_csv(store, "scalability.csv").round(4).astype(str))
            gr.Image(_img(store, "scalability.png"), show_label=False)
            gr.Image(_img(store, "weight_sensitivity.png"), show_label=False)
    return demo
