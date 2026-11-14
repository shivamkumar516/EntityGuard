"""Dataset loading, dataset health summary and the exact-title baseline (B1).

Replaces the old script-style files baseline.py, eda_summary.py and error_analysis.py.

Encodings (important):
    DBLP.csv     -> Latin-1
    Scholar.csv  -> UTF-8 with BOM ("utf-8-sig")
Reading Scholar as Latin-1 silently corrupts names such as "Wäsch".

Run from the repo root:
    python -m src.data_loading            # health summary + B1 on all gold pairs
    python -m src.data_loading --audit    # also writes reports/baseline_error_audit.csv

Note: the gold mapping is used here ONLY to evaluate B1, never to build predictions.
"""
import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from normalize import normalize_title  # Student 1's normalization  # noqa: E402

REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")

# The original benchmark zip ships "DBLP1.csv"; this repo stores it as "DBLP.csv".
DBLP_NAMES = ("DBLP.csv", "DBLP1.csv")
SCHOLAR_NAME = "Scholar.csv"
GOLD_NAME = "DBLP-Scholar_perfectMapping.csv"


def find_data_file(names, data_dir=DEFAULT_DATA_DIR):
    """Return the path of the first file in `names` found under data_dir (case-insensitive, searches sub-folders)."""
    if isinstance(names, str):
        names = (names,)
    wanted = [n.lower() for n in names]
    found = {}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower() in wanted and f.lower() not in found:
                found[f.lower()] = os.path.join(root, f)
    for n in wanted:  # respect the preference order of `names`
        if n in found:
            return found[n]
    raise FileNotFoundError(f"None of {list(names)} found inside {data_dir}")


def _clean_source(df):
    """Numeric year (junk -> <NA>) and blank text fields -> None, same as the notebook."""
    df = df.copy()
    df.columns = [c.strip().replace('"', "") for c in df.columns]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ("title", "authors", "venue"):
        df[c] = df[c].where(df[c].notna() & (df[c].astype(str).str.strip() != ""), None)
    return df


def load_datasets(data_dir=DEFAULT_DATA_DIR):
    """Load (dblp, scholar, gold). gold has columns dblp_id, scholar_id."""
    dblp = pd.read_csv(find_data_file(DBLP_NAMES, data_dir), encoding="latin-1")
    scholar = pd.read_csv(find_data_file(SCHOLAR_NAME, data_dir), encoding="utf-8-sig")
    gold = pd.read_csv(find_data_file(GOLD_NAME, data_dir), encoding="utf-8-sig")
    gold.columns = [c.strip().replace('"', "") for c in gold.columns]
    gold = gold.rename(columns={"idDBLP": "dblp_id", "idScholar": "scholar_id"})
    return _clean_source(dblp), _clean_source(scholar), gold


def gold_pairs(gold):
    return set(zip(gold.dblp_id, gold.scholar_id))


def dataset_health(dblp, scholar):
    """One row per source: record count, missing values per field, title length."""
    rows = []
    for name, df in (("DBLP", dblp), ("Scholar", scholar)):
        titles = df.title.dropna()
        rows.append({
            "source": name,
            "records": len(df),
            "missing title": int(df.title.isna().sum()),
            "missing authors": int(df.authors.isna().sum()),
            "missing venue": int(df.venue.isna().sum()),
            "missing year": int(df.year.isna().sum()),
            "missing year %": round(100 * df.year.isna().mean(), 1),
            "avg title length (chars)": round(titles.str.len().mean(), 1),
        })
    return pd.DataFrame(rows).set_index("source")


def exact_title_baseline(dblp, scholar):
    """B1: predict a match when the normalized titles are identical.

    A hash join on the normalized title, so it covers all 168M pairs in O(|DBLP| + |Scholar|).
    Records with an empty normalized title are skipped. Returns a set of (dblp_id, scholar_id).
    """
    d = pd.DataFrame({"dblp_id": dblp.id, "t": dblp.title.map(normalize_title)})
    s = pd.DataFrame({"scholar_id": scholar.id, "t": scholar.title.map(normalize_title)})
    m = d[d.t != ""].merge(s[s.t != ""], on="t")
    return set(zip(m.dblp_id, m.scholar_id))


def baseline_error_samples(pred_pairs, gold_set, dblp, scholar, n=10, seed=42):
    """n false positives + n false negatives of a prediction set, with titles, chosen reproducibly."""
    d_title = dblp.set_index("id").title
    s_title = scholar.set_index("id").title
    out = []
    for kind, pairs in (("False positive", pred_pairs - gold_set), ("False negative", gold_set - pred_pairs)):
        pairs = pd.DataFrame(sorted(pairs), columns=["dblp_id", "scholar_id"])
        pairs = pairs.sample(min(n, len(pairs)), random_state=seed)
        pairs.insert(0, "kind", kind)
        pairs["dblp_title"] = pairs.dblp_id.map(d_title)
        pairs["scholar_title"] = pairs.scholar_id.map(s_title)
        out.append(pairs)
    return pd.concat(out, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dataset health summary and exact-title baseline (B1).")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--audit", action="store_true", help="write reports/baseline_error_audit.csv")
    args = ap.parse_args(argv)

    from src.evaluation import prf

    dblp, scholar, gold = load_datasets(args.data_dir)
    print(dataset_health(dblp, scholar).T.to_string())
    print(f"\nGold matches: {len(gold):,}   Cartesian pairs: {len(dblp) * len(scholar):,}")

    pred = exact_title_baseline(dblp, scholar)
    gs = gold_pairs(gold)
    m = prf(pred, gs)
    print("\nB1 exact normalized title (all gold pairs, not just the test split):")
    print(f"  TP {m['TP']}  FP {m['FP']}  FN {m['FN']}  "
          f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")

    if args.audit:
        out = os.path.join(REPO_ROOT, "reports", "baseline_error_audit.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        baseline_error_samples(pred, gs, dblp, scholar).to_csv(out, index=False)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
