"""Student 1's sample-based similarity study (src/compare_similarity.py), saved as tables.

Student 1's script scores every gold pair against one random non-match and one
"same first title word" non-match, then prints the tables. This module runs exactly
his code (loading, sampling, scoring, threshold choice) and returns the tables as
DataFrames so they can be saved to outputs/ and compared with the full pipeline.

One change of protocol, for the problem statement's rule that the test split stays
untouched: only gold pairs whose DBLP record is in TRAIN or VALIDATION are used
(Student 1's script used all gold pairs).

Note: the sample is ~50 % matches, while the real candidate set is ~1 % matches,
so precision and F1 here are much higher than in the full pipeline.

Run from the repo root:  python -m src.evaluation.student1_study
"""
import contextlib
import io
import os

import pandas as pd

from .. import compare_similarity as S1C   # Student 1's script (not modified)
from .split import split_of_ids

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "outputs")


def load_student1_data():
    """DBLP / Scholar records and gold pairs, loaded with Student 1's own loaders."""
    dblp = S1C.load_records(S1C.read_table(S1C.find_file("DBLP.csv")))
    scholar = S1C.load_records(S1C.read_table(S1C.find_file("Scholar.csv")))
    gold_table = S1C.read_table(S1C.find_file("DBLP-Scholar_perfectMapping.csv"))
    gold = list(dict.fromkeys((str(r.idDBLP), str(r.idScholar)) for r in gold_table.itertuples(index=False)
                              if str(r.idDBLP) in dblp and str(r.idScholar) in scholar))
    return dblp, scholar, gold


def _scores(rows, key):
    return [r["titles"][key] for r in rows] if key in S1C.TITLE_MEASURES else [r[key] for r in rows]


def _title_combined(rows, alpha):
    return [None if r["edit"] is None else alpha * r["edit"] + (1 - alpha) * r["jaccard"] for r in rows]


def run_study(dblp, scholar, gold_pairs):
    """Run Student 1's study on the given gold pairs. Returns {table name: DataFrame}."""
    with contextlib.redirect_stdout(io.StringIO()):          # his scorer prints progress every 500 pairs
        gold, rand, hard = S1C.sample_pairs(gold_pairs, dblp, scholar, None)
        g_rows, _ = S1C.score_pairs(gold, dblp, scholar)
        r_rows, _ = S1C.score_pairs(rand, dblp, scholar)
        h_rows, _ = S1C.score_pairs(hard, dblp, scholar)

    def row(match, other, **label):
        T, P, R, F = S1C.best_threshold(match, other)
        return {**label, "gold_avg": S1C.average(match), "nonmatch_avg": S1C.average(other),
                "best_T": T, "precision": P, "recall": R, "f1": F}

    t = {}
    t["sample"] = pd.DataFrame([{"gold_pairs": len(gold), "random_nonmatches": len(rand),
                                 "same_first_word_nonmatches": len(hard)}])
    t["title_measures"] = pd.DataFrame(
        [row(_scores(g_rows, m), _scores(o_rows, m), nonmatches=name, measure=m)
         for name, o_rows in (("random", r_rows), ("same first word", h_rows)) for m in S1C.TITLE_MEASURES])
    t["alpha"] = pd.DataFrame([row(_title_combined(g_rows, a), _title_combined(h_rows, a), alpha=a)
                               for a in S1C.ALPHAS])
    t["author_methods"] = pd.DataFrame(
        [row(_scores(g_rows, k), _scores(h_rows, k), method=label)
         for label, k in (("token jaccard", "author_jaccard"), ("best name match", "author_names"))])
    t["weight_trials"] = pd.DataFrame(
        [row(S1C.record_scores(g_rows, w, "author_jaccard"), S1C.record_scores(h_rows, w, "author_jaccard"),
             setting=label, w_title=w["title"], w_author=w["author"], w_venue=w["venue"], w_year=w["year"])
         for label, w in S1C.WEIGHT_TRIALS])
    curves = []
    for name, match, other in (
            ("combined title (alpha 0.4)", _scores(g_rows, "combined"), _scores(h_rows, "combined")),
            ("record score, example weights", S1C.record_scores(g_rows, S1C.WEIGHT_TRIALS[0][1], "author_jaccard"),
             S1C.record_scores(h_rows, S1C.WEIGHT_TRIALS[0][1], "author_jaccard"))):
        for T in S1C.THRESHOLDS:
            P, R, F = S1C.precision_recall_f1(match, other, T)
            curves.append({"score": name, "threshold": T, "precision": P, "recall": R, "f1": F})
    t["threshold_curves"] = pd.DataFrame(curves)
    return t


def non_test_gold(dblp, gold, seed=42):
    """Gold pairs whose DBLP record is in TRAIN or VALIDATION (same split as the pipeline)."""
    split = split_of_ids(dblp.keys(), seed)
    return [p for p in gold if split[p[0]] != "test"]


def main(out_dir=OUT_DIR):
    dblp, scholar, gold = load_student1_data()
    tables = run_study(dblp, scholar, non_test_gold(dblp, gold))
    os.makedirs(out_dir, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"student1_{name}.csv"), index=False)
    return tables


if __name__ == "__main__":
    for name, df in main().items():
        print(f"\n== {name}\n{df.round(3).to_string(index=False)}")
