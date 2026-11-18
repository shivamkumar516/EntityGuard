"""EntityGuard pipeline: the working logic of experiments/EntityGuard_Student3.ipynb in one module.

The scoring and decision logic is maintained here. Candidate generation is provided by the
Student 2 module in src/blocking and imported below, so the web application and batch pipeline
share one reusable blocking implementation.

    data/DBLP.csv, data/Scholar.csv, data/DBLP-Scholar_perfectMapping.csv
        -> load_datasets()             encodings: DBLP latin-1, Scholar utf-8 with BOM
        -> normalization               Sections 11-13 (normalize_title, author tokens/names, venue, year)
        -> BlockingIndex               Sections 26-30: title prefix, rare title tokens, character prefix,
                                       rare character 6-grams; union of passes = candidate set
        -> pair_features()             Sections 15-22: DP Levenshtein, edit similarity, Jaccard, n-grams,
                                       title / author / venue / year similarity
        -> weighted_score()            Sections 23-24: S = sum w_k S_k / sum w_k over available fields
        -> decide() / explain()        decision band + human-readable reason
        -> ML probability              Sections 36-38: Logistic Regression / Decision Tree bundles

Command line (from the repo root):
    python -m src.pipeline verify      rebuild all candidates, compare with outputs/summary.json
    python -m src.pipeline train       same, then train LR + DT and save outputs/dt_model.joblib
"""
import argparse
import json
import math
import os
import re
import threading
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from .evaluation import best_threshold, prf, threshold_sweep
from .evaluation.split import split_labels
from .blocking import BlockingIndex, PASS_CAPS, PASS_LABELS, PASS_NAMES

try:                                    # C implementation of the same edit distance, used for bulk scoring
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein
except ImportError:                     # pragma: no cover - rapidfuzz is in requirements.txt
    _RFLevenshtein = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
SEED = 42

# ====================================================================== 1. configuration
FIELDS = ("title", "author", "venue", "year")

AUTO_MATCH, MANUAL_REVIEW, NON_MATCH = "Auto-Match", "Manual Review", "Non-Match"
DECISION_LABELS = (AUTO_MATCH, MANUAL_REVIEW, NON_MATCH)

# Decision rule of the application. The weights and band limits are NOT chosen by hand: they are the
# values the notebook selected on the VALIDATION split (Sections 12 and 41) and saved in
# outputs/summary.json ("chosen"), so re-running the experiments updates the app automatically.
# The fallback below is only used when summary.json is missing; it equals the current validation result.
FALLBACK_WEIGHTS = {"title": 0.45, "author": 0.30, "venue": 0.00, "year": 0.25}
FALLBACK_AUTO_MATCH_T = 0.95  # band_high: smallest threshold with >= 97 % precision on validation
FALLBACK_NON_MATCH_T = 0.49   # band_low: keeps >= 99 % of the true matches found by blocking (validation)


def validated_config(out_dir=None):
    """(weights dict, auto-match threshold, non-match threshold) chosen on validation (outputs/summary.json)."""
    path = os.path.join(out_dir or os.path.join(REPO_ROOT, "outputs"), "summary.json")
    try:
        with open(path, encoding="utf-8") as f:
            c = json.load(f)["chosen"]
        w = dict(zip(("title", "author", "venue", "year"), (float(x) for x in c["weights_title_author_venue_year"])))
        auto_t, non_t = float(c["band_high"]), float(c["band_low"])
        if abs(sum(w.values()) - 1) > 1e-6 or not 0 <= non_t <= auto_t <= 1:
            raise ValueError(f"invalid chosen configuration in {path}: {c}")
        return w, auto_t, non_t
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return dict(FALLBACK_WEIGHTS), FALLBACK_AUTO_MATCH_T, FALLBACK_NON_MATCH_T


APP_WEIGHTS, AUTO_MATCH_T, NON_MATCH_T = validated_config()

# Measures chosen on validation in the notebook (Section 12) and used by the saved ML models.
DEFAULT_ALPHA = 0.6          # S_title = alpha * S_edit + (1 - alpha) * S_jaccard
AUTHOR_MEASURE = "author_best"
VENUE_MEASURE = "venue_tri"
FIELD_COLUMNS = {"title": "title_combined", "author": AUTHOR_MEASURE, "venue": VENUE_MEASURE, "year": "year"}


def load_summary(out_dir=OUT_DIR):
    """outputs/summary.json written by the notebook ({} if it does not exist)."""
    path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ====================================================================== 2. normalization
TITLE_STOP_WORDS = {"a", "an", "the", "for", "of", "and", "in", "on", "to"}
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


def is_missing(value):
    """True for None, NaN, pd.NA and blank / 'nan' / 'none' / 'null' strings."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    try:
        if value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}


def _basic_clean(text):
    """Lowercase, punctuation -> space, squeeze whitespace."""
    return " ".join(_NON_ALNUM.sub(" ", str(text).lower()).split())


def normalize_title(text):
    """'An Efficient Method for Record-Linkage!' -> 'efficient method record linkage' (Section 11)."""
    if is_missing(text):
        return ""
    return " ".join(w for w in _basic_clean(text).split() if w not in TITLE_STOP_WORDS)


def author_tokens(text):
    """'Smith, J.; Kumar, A.' -> ['smith', 'j', 'kumar', 'a'] (Section 12)."""
    return [] if is_missing(text) else _basic_clean(text).split()


def split_author_names(text):
    """Whole author names: ';' separates people, otherwise ','. 'Smith, J.; Kumar, A.' -> ['smith j', 'kumar a']."""
    if is_missing(text):
        return []
    raw = str(text)
    parts = raw.split(";") if ";" in raw else raw.split(",")
    return [c for c in (_basic_clean(p) for p in parts) if c]


def normalize_venue(text):
    """'Proc. ICDE' -> 'proc icde' (Section 13)."""
    return "" if is_missing(text) else _basic_clean(text)


def parse_year(value):
    """First run of 4 digits as an int; missing or junk years (e.g. '11') -> None."""
    if is_missing(value):
        return None
    digits = ""
    for ch in str(value).strip():
        if ch.isdigit():
            digits += ch
            if len(digits) == 4:
                break
        elif digits:
            break
    return int(digits) if len(digits) == 4 else None


# ====================================================================== 3. similarity measures
def levenshtein(a, b):
    """Section 15: edit distance by dynamic programming over the full (m+1) x (n+1) table.

    D(i,0) = i ; D(0,j) = j ; D(i,j) = min(D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+[a_i != b_j])
    """
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        D[i][0] = i
    for j in range(len(b) + 1):
        D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + c)
    return D[len(a)][len(b)]


def levenshtein_two_rows(a, b):
    """Section 17: the same distance in O(n) memory (only the previous row is kept)."""
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i]
        for j in range(1, len(b) + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1])))
        prev = cur
    return prev[len(b)]


def edit_similarity(a, b):
    """Section 16: S_edit = 1 - D(A,B) / max(|A|,|B|)  (1.0 for two empty strings)."""
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - levenshtein_two_rows(a, b) / longest


def fast_edit_similarity(a, b):
    """Identical value to edit_similarity, computed by rapidfuzz's C code (for scoring many pairs)."""
    if _RFLevenshtein is None:
        return edit_similarity(a, b)
    return _RFLevenshtein.normalized_similarity(a, b)


def jaccard(a, b):
    """Section 18: |A ∩ B| / |A ∪ B| on sets (1.0 when both are empty)."""
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def char_ngrams(text, n):
    """Section 19: overlapping character n-grams; 'database', 2 -> da at ta ab ba as se."""
    text = "" if text is None else str(text)
    if n < 1 or text == "":
        return []
    if len(text) <= n:
        return [text]
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def ngram_jaccard(a, b, n):
    return jaccard(char_ngrams(a, n), char_ngrams(b, n))


def author_best_match(names_a, names_b, edit=edit_similarity):
    """Section 21, method 2: average best name-to-name edit similarity, averaged over both directions."""
    if not names_a or not names_b:
        return float("nan")
    m = [[edit(x, y) for y in names_b] for x in names_a]
    fwd = sum(max(row) for row in m) / len(names_a)
    bwd = sum(max(m[i][j] for i in range(len(names_a))) for j in range(len(names_b))) / len(names_b)
    return (fwd + bwd) / 2


def _year_missing(y):
    return y is None or (isinstance(y, float) and math.isnan(y))


def year_similarity(y1, y2):
    """Section 22: 1 same year, 0.5 one year apart, 0 otherwise, NaN if either year is missing."""
    if _year_missing(y1) or _year_missing(y2):
        return float("nan")
    d = abs(float(y1) - float(y2))
    return 1.0 if d == 0 else 0.5 if d == 1 else 0.0


# ====================================================================== 4. records
def _text_or_none(value):
    return None if is_missing(value) else str(value).strip()


def prepare_record(title=None, authors=None, venue=None, year=None, record_id=None):
    """Raw fields + every normalized form the matcher and the blocking need."""
    title, authors, venue = _text_or_none(title), _text_or_none(authors), _text_or_none(venue)
    t_norm = normalize_title(title)
    tokens = t_norm.split()
    v_norm = normalize_venue(venue)
    yr = parse_year(year)
    return {"id": record_id, "title": title, "authors": authors, "venue": venue,
            "year": None if is_missing(year) else year,
            "t_norm": t_norm, "t_tokens": tokens, "t_squash": "".join(tokens),
            "a_names": split_author_names(authors), "a_tokens": frozenset(author_tokens(authors)),
            "v_norm": v_norm, "yr": yr}


def add_normalized_columns(df):
    """Add the normalized columns used by blocking and features to a DBLP / Scholar table (in place)."""
    df["t_norm"] = df["title"].map(normalize_title)
    df["t_tokens"] = df["t_norm"].str.split()
    df["t_squash"] = df["t_tokens"].map("".join)       # robust to glued words in Scholar titles
    df["a_names"] = df["authors"].map(split_author_names)
    df["a_tokens"] = df["authors"].map(lambda a: frozenset(author_tokens(a)))
    df["v_norm"] = df["venue"].map(normalize_venue)
    df["yr"] = df["year"].map(parse_year).astype(float)  # NaN = missing or junk
    return df


def _find(data_dir, names):
    for n in names:
        p = os.path.join(data_dir, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {names} found in {data_dir}")


def load_datasets(data_dir=DATA_DIR):
    """DBLP, Scholar (normalized, DBLP with its train/val/test split) and the gold mapping."""
    dblp = pd.read_csv(_find(data_dir, ("DBLP.csv", "DBLP1.csv")), encoding="latin-1")
    scholar = pd.read_csv(_find(data_dir, ("Scholar.csv",)), encoding="utf-8-sig")
    gold = pd.read_csv(_find(data_dir, ("DBLP-Scholar_perfectMapping.csv",))).rename(
        columns={"idDBLP": "dblp_id", "idScholar": "scholar_id"})
    for df in (dblp, scholar):
        df["id"] = df["id"].astype(str)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        for c in ("title", "authors", "venue"):
            df[c] = df[c].astype(object).where(df[c].notna() & (df[c].astype(str).str.strip() != ""), None)
        add_normalized_columns(df)
    gold["dblp_id"] = gold["dblp_id"].astype(str)
    gold["scholar_id"] = gold["scholar_id"].astype(str)
    dblp["split"] = split_labels(len(dblp), seed=SEED)  # 60 / 20 / 20 by DBLP record, as in the notebook
    gold["split"] = gold["dblp_id"].map(dict(zip(dblp["id"], dblp["split"])))
    return dblp, scholar, gold


def record_from_row(row):
    """Prepared record dict from a row of a table produced by load_datasets()."""
    yr = row["yr"]
    return {"id": row["id"], "title": row["title"], "authors": row["authors"], "venue": row["venue"],
            "year": None if is_missing(row["year"]) else int(row["year"]),
            "t_norm": row["t_norm"], "t_tokens": row["t_tokens"], "t_squash": row["t_squash"],
            "a_names": row["a_names"], "a_tokens": row["a_tokens"], "v_norm": row["v_norm"],
            "yr": None if _year_missing(yr) else int(yr)}


# ====================================================================== 5. pair features
NAN = float("nan")


def pair_features(d, s, alpha=DEFAULT_ALPHA, edit=edit_similarity, extended=True):
    """Every attribute similarity for one DBLP record d and one Scholar record s (prepared dicts).

    NaN = the field is missing on at least one side (it is then left out of the weighted score).
    extended=False skips the measures that are only displayed (title_exact / title_tri / author_tok / venue_exact).
    """
    f = {}
    dt, st = d["t_norm"], s["t_norm"]
    if dt and st:
        f["title_edit"] = edit(dt, st)
        f["title_jac"] = jaccard(dt.split(), st.split())
        f["len_ratio"] = min(len(dt), len(st)) / max(len(dt), len(st))
        f["title_combined"] = alpha * f["title_edit"] + (1 - alpha) * f["title_jac"]
        if extended:
            f["title_exact"] = float(dt == st)
            f["title_tri"] = ngram_jaccard(dt, st, 3)
    else:
        f.update(title_edit=NAN, title_jac=NAN, len_ratio=NAN, title_combined=NAN)
        if extended:
            f.update(title_exact=NAN, title_tri=NAN)
    f["author_best"] = author_best_match(d["a_names"], s["a_names"], edit)
    dv, sv = d["v_norm"], s["v_norm"]
    f["venue_tri"] = ngram_jaccard(dv, sv, 3) if dv and sv else NAN
    f["year"] = year_similarity(d["yr"], s["yr"])
    if extended:
        f["author_tok"] = jaccard(d["a_tokens"], s["a_tokens"]) if d["a_tokens"] and s["a_tokens"] else NAN
        f["venue_exact"] = float(dv == sv) if dv and sv else NAN
    return f


def field_similarities(features):
    """{title, author, venue, year} -> the four field similarities S_title, S_author, S_venue, S_year."""
    return {k: features[c] for k, c in FIELD_COLUMNS.items()}


def features_for_pairs(dblp, scholar, di, si, alpha=DEFAULT_ALPHA):
    """Float32 feature table for many pairs (row positions di / si), using the fast edit distance."""
    cols = ["t_norm", "a_names", "v_norm", "yr"]
    D = dblp[cols].to_dict("records")
    S = scholar[cols].to_dict("records")
    rows = []
    for a, b in zip(np.asarray(di), np.asarray(si)):
        rows.append(pair_features(D[a], S[b], alpha, fast_edit_similarity, extended=False))
    F = pd.DataFrame(rows, columns=["title_edit", "title_jac", "len_ratio", "title_combined",
                                    "author_best", "venue_tri", "year"]).astype(np.float32)
    # the notebook builds title_combined from the float32 columns; do the same so scores agree bit for bit
    F["title_combined"] = (alpha * F["title_edit"] + (1 - alpha) * F["title_jac"]).astype(np.float32)
    F.insert(0, "si", np.asarray(si, dtype=np.int64))
    F.insert(0, "di", np.asarray(di, dtype=np.int64))
    return F


# ====================================================================== 6. weighted score & decision
def weighted_score(sims, weights=APP_WEIGHTS):
    """Section 23-24 for one pair. Returns (score, rows) where rows explain every field's contribution.

    A missing field is dropped and the remaining weights are renormalized:
        S = sum_{k in K} w_k S_k / sum_{k in K} w_k
    """
    avail = {k: v for k, v in sims.items() if not _year_missing(v)}
    den = sum(weights[k] for k in avail)
    num = sum(weights[k] * v for k, v in avail.items())
    score = num / den if den > 0 else 0.0
    rows = []
    for k in FIELDS:
        v = sims.get(k, NAN)
        missing = k not in avail
        eff = 0.0 if missing or den == 0 else weights[k] / den
        rows.append({"field": k, "similarity": None if missing else float(v), "weight": weights[k],
                     "effective_weight": eff, "contribution": 0.0 if missing else eff * float(v)})
    return float(score), rows


def weighted_score_frame(df, weights=APP_WEIGHTS, columns=None):
    """Vectorized weighted score for a feature table (NaN fields renormalized away), float32."""
    columns = columns or FIELD_COLUMNS
    num = np.zeros(len(df)); den = np.zeros(len(df))
    for k in FIELDS:
        s = df[columns[k]].to_numpy(dtype=float)
        ok = ~np.isnan(s)
        num += weights[k] * np.where(ok, s, 0)
        den += weights[k] * ok
    return (num / np.maximum(den, 1e-12)).astype(np.float32)


def decide(score, auto_t=AUTO_MATCH_T, non_t=NON_MATCH_T):
    """Auto-Match for S >= auto_t, Manual Review for non_t <= S < auto_t, Non-Match below non_t."""
    if score >= auto_t:
        return AUTO_MATCH
    if score >= non_t:
        return MANUAL_REVIEW
    return NON_MATCH


def decide_array(scores, auto_t=AUTO_MATCH_T, non_t=NON_MATCH_T):
    s = np.asarray(scores, dtype=float)
    return np.where(s >= auto_t, AUTO_MATCH, np.where(s >= non_t, MANUAL_REVIEW, NON_MATCH))


# ====================================================================== 7. machine learning
ML_FEATURES = ["S_title", "S_author", "S_venue", "S_year", "TitleLengthRatio",
               "author_missing", "venue_missing", "year_missing"]
LR_C_GRID = (0.01, 0.1, 1, 10)
DT_DEPTH_GRID = (3, 4, 5, 6, 8, 10)
PROB_GRID = np.round(np.arange(0.05, 0.996, 0.01), 2)
MODEL_FILES = {"Logistic Regression": "lr_model", "Decision Tree": "dt_model"}   # .json preferred, .joblib fallback


def make_X(df, author_col=AUTHOR_MEASURE, venue_col=VENUE_MEASURE):
    """X = [S_title, S_author, S_venue, S_year, TitleLengthRatio] + missing flags (missing -> 0, flag 1)."""
    a, v, y = df[author_col].astype(float), df[venue_col].astype(float), df["year"].astype(float)
    return np.column_stack([df["title_combined"].astype(float), np.nan_to_num(a), np.nan_to_num(v),
                            np.nan_to_num(y), np.nan_to_num(df["len_ratio"].astype(float)),
                            a.isna(), v.isna(), y.isna()]).astype(np.float32)


class PortableModel:
    """A trained LR / DT stored as plain numbers (JSON), so it works with any library version.

    predict_proba reproduces scikit-learn exactly: the StandardScaler + LogisticRegression pipeline
    (float32 scaling, then the logistic function) and the decision-tree walk (x[f] <= threshold -> left).
    """

    def __init__(self, spec):
        self.spec = spec
        if spec["kind"] == "logistic_regression":
            self.mean = np.asarray(spec["mean"], dtype=np.float64)
            self.scale = np.asarray(spec["scale"], dtype=np.float64)
            self.coef = np.asarray(spec["coef"], dtype=np.float64)
            self.intercept = float(spec["intercept"])
        elif spec["kind"] == "decision_tree":
            self.left = np.asarray(spec["children_left"], dtype=np.int64)
            self.right = np.asarray(spec["children_right"], dtype=np.int64)
            self.feature = np.asarray(spec["feature"], dtype=np.int64)
            self.threshold = np.asarray(spec["threshold"], dtype=np.float64)
            self.p_match = np.asarray(spec["p_match"], dtype=np.float64)
        else:
            raise ValueError(f"unknown model kind {spec['kind']!r}")

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.spec["kind"] == "logistic_regression":
            Z = (X.astype(np.float64) - self.mean).astype(np.float32)
            Z = (Z.astype(np.float64) / self.scale).astype(np.float32)
            z = Z.astype(np.float64) @ self.coef + self.intercept
            p = np.exp(-np.logaddexp(0.0, -z))                       # numerically stable 1 / (1 + e^-z)
        else:
            node = np.zeros(len(X), dtype=np.int64)
            rows = np.arange(len(X))
            while True:
                inner = self.left[node] != -1
                if not inner.any():
                    break
                r, n = rows[inner], node[inner]
                go_left = X[r, self.feature[n]].astype(np.float64) <= self.threshold[n]
                node[inner] = np.where(go_left, self.left[n], self.right[n])
            p = self.p_match[node]
        return np.column_stack([1 - p, p])

    @classmethod
    def from_sklearn(cls, model):
        """Convert a fitted StandardScaler+LogisticRegression pipeline or DecisionTreeClassifier."""
        if hasattr(model, "tree_"):
            t = model.tree_
            v = t.value[:, 0, :].astype(np.float64)
            classes = list(model.classes_)
            p = v[:, classes.index(1)] / v.sum(axis=1)
            return cls({"kind": "decision_tree", "children_left": t.children_left.tolist(),
                        "children_right": t.children_right.tolist(), "feature": t.feature.tolist(),
                        "threshold": t.threshold.tolist(), "p_match": p.tolist()})
        scaler, lr = model[0], model[-1]
        if list(lr.classes_) != [0, 1]:
            raise ValueError("expected a binary model with classes [0, 1]")
        return cls({"kind": "logistic_regression", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                    "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])})


def save_model_json(bundle, path):
    """Write a model bundle as portable JSON (model numbers + threshold + feature recipe)."""
    model = bundle["model"]
    spec = model.spec if isinstance(model, PortableModel) else PortableModel.from_sklearn(model).spec
    out = {k: bundle[k] for k in ("threshold", "param", "val_f1") if k in bundle}
    out.update(feature_names=ML_FEATURES, author_col=bundle.get("author_col", AUTHOR_MEASURE),
               venue_col=bundle.get("venue_col", VENUE_MEASURE), model=spec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return path


def load_model(base_path):
    """Model bundle {model, threshold, param, val_f1, author_col, venue_col} or None.

    Prefers <base>.json (portable, any library version); falls back to the scikit-learn <base>.joblib.
    """
    base = base_path[:-7] if base_path.endswith(".joblib") else base_path[:-5] if base_path.endswith(".json") else base_path
    bundle = None
    if os.path.exists(base + ".json"):
        with open(base + ".json", encoding="utf-8") as f:
            bundle = json.load(f)
        bundle["model"] = PortableModel(bundle["model"])
    elif os.path.exists(base + ".joblib"):
        try:
            import joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # e.g. a scikit-learn minor-version warning on unpickling
                bundle = joblib.load(base + ".joblib")
        except Exception as e:                       # incompatible pickle: run without this model
            warnings.warn(f"could not load {base}.joblib ({type(e).__name__}: {e}); run "
                          "'python -m src.pipeline export-models' where it was trained")
            return None
    if bundle is None:
        return None
    used = (bundle.get("author_col", AUTHOR_MEASURE), bundle.get("venue_col", VENUE_MEASURE))
    if used != (AUTHOR_MEASURE, VENUE_MEASURE):
        raise ValueError(f"{base}: model was trained on {used}; this pipeline computes {(AUTHOR_MEASURE, VENUE_MEASURE)}")
    return bundle


def model_probability(bundle, features_df):
    """P(match) for every row of a feature table; NaN where the title is missing (no ML input)."""
    out = np.full(len(features_df), np.nan)
    ok = ~features_df["title_combined"].astype(float).isna().to_numpy()
    if ok.any():
        X = make_X(features_df[ok], bundle.get("author_col", AUTHOR_MEASURE), bundle.get("venue_col", VENUE_MEASURE))
        out[ok] = bundle["model"].predict_proba(X)[:, 1]
    return out


def sample_training_pairs(train, neg_per_pos=20, seed=SEED):
    """All positives + a reproducible sample of negatives (random_state=42), shuffled."""
    pos = train[train["label"] == 1]
    neg = train[train["label"] == 0]
    neg_s = neg.sample(min(len(neg), neg_per_pos * len(pos)), random_state=seed)
    return pd.concat([pos, neg_s]).sample(frac=1, random_state=seed)


def train_models(train, val, n_gold_val, neg_per_pos=20, seed=SEED):
    """Fit LR (C grid) and DT (depth grid) on sampled TRAIN pairs; pick each model and its threshold on VALIDATION."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    tr = sample_training_pairs(train, neg_per_pos, seed)
    X_tr, y_tr = make_X(tr), tr["label"].to_numpy()
    X_va, y_va = make_X(val), val["label"].to_numpy()
    candidates = [("Logistic Regression", f"C={C}",
                   make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed)))
                  for C in LR_C_GRID]
    candidates += [("Decision Tree", f"max_depth={d}",
                    DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=seed))
                   for d in DT_DEPTH_GRID]
    rows, best = [], {}
    for name, param, model in candidates:
        model.fit(X_tr, y_tr)
        T, f = best_threshold(model.predict_proba(X_va)[:, 1], y_va, n_gold_val, grid=PROB_GRID)
        rows.append({"model": name, "param": param, "val F1": f, "threshold": T})
        if name not in best or f >= best[name]["val_f1"]:          # ties keep the last setting, as in the notebook
            best[name] = {"model": model, "threshold": float(T), "param": param, "val_f1": float(f),
                          "feature_names": ML_FEATURES, "author_col": AUTHOR_MEASURE, "venue_col": VENUE_MEASURE}
    return pd.DataFrame(rows), best


# ====================================================================== 8. multi-pass blocking
# BlockingIndex and pass configuration are implemented in src.blocking (Student 2).
# The application imports that implementation directly so notebook and app logic
# no longer maintain separate candidate-generation implementations.


# ====================================================================== 9. explanation
def _fmt(v, nd=3):
    return "missing" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def explain(result, auto_t=AUTO_MATCH_T, non_t=NON_MATCH_T):
    """Markdown explanation of why a pair was auto-matched, sent to review, or rejected."""
    score, decision, sims, rows = result["score"], result["decision"], result["sims"], result["contributions"]
    d, s = result["dblp"], result["scholar"]
    lines = []
    if decision == AUTO_MATCH:
        lines.append(f"**{AUTO_MATCH}:** S = {score:.3f} ≥ {auto_t:.2f}. The weighted evidence is strong enough to "
                     "merge the two records without a human.")
    elif decision == MANUAL_REVIEW:
        lines.append(f"**{MANUAL_REVIEW}:** {non_t:.2f} ≤ S = {score:.3f} < {auto_t:.2f}. The records look related, "
                     "but not certain enough to merge automatically, so a human should decide.")
    else:
        lines.append(f"**{NON_MATCH}:** S = {score:.3f} < {non_t:.2f}. The records disagree on too much to be "
                     "the same publication.")

    present = [r for r in rows if r["similarity"] is not None and r["weight"] > 0]
    if present:
        strong = sorted(present, key=lambda r: -r["contribution"])[0]
        weak = sorted(present, key=lambda r: r["similarity"])[0]
        lines.append(f"- Largest contribution: **{strong['field']}** (similarity {strong['similarity']:.3f} × "
                     f"effective weight {strong['effective_weight']:.3f} = {strong['contribution']:.3f}).")
        if weak["similarity"] < 0.75:
            lost = weak["effective_weight"] * (1 - weak["similarity"])
            lines.append(f"- Weakest field: **{weak['field']}** (similarity {weak['similarity']:.3f}) "
                         f"costs {lost:.3f} of the possible score.")
    missing = [r["field"] for r in rows if r["similarity"] is None and r["weight"] > 0]
    if missing:
        used = sum(r["weight"] for r in rows if r["similarity"] is not None)
        lines.append(f"- Missing on at least one side: **{', '.join(missing)}**. Those weights were dropped and the "
                     f"remaining weights renormalized (they sum to {used:.2f}, rescaled to 1), so a missing value is "
                     "not treated as a disagreement.")
    zero = [r["field"] for r in rows if r["weight"] == 0]
    if zero:
        lines.append(f"- **{', '.join(zero)}** has weight 0: validation found it does not help (Scholar venues are "
                     "noisy or missing), so it is shown for information only and does not change S.")

    t = sims["title"]
    if not _year_missing(t):
        if result["features"].get("title_exact") == 1.0:
            lines.append("- Titles are **identical after normalization** (case, punctuation and stop words removed).")
        elif t >= 0.9:
            lines.append(f"- Titles are near-identical (S_title {t:.3f}).")
        elif t < 0.5:
            lines.append(f"- Titles differ substantially (S_title {t:.3f}); typical causes are citation text glued "
                         "onto Scholar titles, glued words or a different paper.")
    a = sims["author"]
    if not _year_missing(a) and a < 0.5:
        lines.append(f"- Author lists differ (S_author {a:.3f}): different people, reversed name order or truncated lists.")
    y = sims["year"]
    if not _year_missing(y) and y < 1:
        lines.append("- Years differ by one (S_year 0.5)." if y == 0.5 else "- Years differ by more than one (S_year 0).")
    if len(d.get("t_tokens") or []) <= 3 and d.get("t_norm"):
        lines.append("- Short title (≤ 3 significant words). Short generic titles (editorials, columns) were a main "
                     "false-positive cause in the error analysis.")
    if _year_missing(y) and not _year_missing(t) and t >= 0.8 and not _year_missing(a) and a >= 0.7:
        lines.append("- Same authors and a similar title but no year to compare: a related publication by the same "
                     "group (tutorial, demo, journal version) is the most common false positive in the error analysis.")

    for name, m in result.get("ml", {}).items():
        if m["probability"] is None:
            continue
        verdict = "match" if m["probability"] >= m["threshold"] else "non-match"
        rule = "match" if decision == AUTO_MATCH else "non-match" if decision == NON_MATCH else None
        agree = "" if rule is None else (" (agrees with the score rule)" if verdict == rule else
                                         " (**disagrees** with the score rule, so treat this pair with care)")
        lines.append(f"- {name}: P(match) = {m['probability']:.3f}, validated threshold {m['threshold']:.2f} → "
                     f"**{verdict}**{agree}.")

    blk = result.get("blocking")
    if blk is not None:
        hits = {p: k for p, k in blk.items() if k}
        if hits:
            parts = [f"{p} (`{', '.join(k[:3])}`)" for p, k in hits.items()]
            lines.append(f"- Multi-pass blocking **would generate** this pair: {'; '.join(parts)}.")
        else:
            lines.append("- Multi-pass blocking **would not generate** this pair (no shared key in any pass), so the "
                         "batch pipeline never scores it, and if it were a true match it would count as a blocking miss.")
    return "\n".join(lines)


# ====================================================================== 10. engine
class EntityGuard:
    """Everything the application needs: data, blocking index, scoring, ML models, decisions."""

    def __init__(self, data_dir=DATA_DIR, out_dir=OUT_DIR, weights=None, auto_t=None, non_t=None, load_data=True):
        """weights / auto_t / non_t default to the validation choice in <out_dir>/summary.json.
        load_data: True = load data/ and build the blocking index now; "background" = do it in a
        thread (scoring typed-in pairs works immediately); False = no data (dashboard / typed pairs only)."""
        self.data_dir, self.out_dir = data_dir, out_dir
        v_w, v_auto, v_non = validated_config(out_dir)
        self.weights = dict(weights or v_w)
        if abs(sum(self.weights.values()) - 1) > 1e-9:
            raise ValueError(f"weights must sum to 1, got {self.weights}")
        self.auto_t = v_auto if auto_t is None else auto_t
        self.non_t = v_non if non_t is None else non_t
        self.summary = load_summary(out_dir)
        self.alpha = float(self.summary.get("chosen", {}).get("alpha", DEFAULT_ALPHA))
        self.models = {}
        for name, fname in MODEL_FILES.items():
            b = load_model(os.path.join(out_dir, fname))
            if b is not None:
                self.models[name] = b
        self.dblp = self.scholar = self.gold_df = None
        self.d_idx, self.s_idx, self.gold = {}, {}, set()
        self.data_error, self._index, self._index_error = None, None, None
        self._data_ready, self._index_ready = threading.Event(), threading.Event()
        if load_data == "background":
            threading.Thread(target=self._load_all, daemon=True).start()
        elif load_data:
            self._load_all()
        else:
            self.data_error = self._index_error = "data loading disabled"
            self._data_ready.set(); self._index_ready.set()

    def _load_all(self):
        try:
            self.dblp, self.scholar, self.gold_df = load_datasets(self.data_dir)
            self.d_idx = {k: i for i, k in enumerate(self.dblp["id"])}
            self.s_idx = {k: i for i, k in enumerate(self.scholar["id"])}
            self.gold = set(zip(self.gold_df["dblp_id"], self.gold_df["scholar_id"]))
        except Exception as e:                              # surfaced in the UI instead of killing the thread
            self.dblp = None
            self.data_error = self._index_error = f"{type(e).__name__}: {e}"
        finally:
            self._data_ready.set()
        if self.dblp is not None:
            try:
                self._index = BlockingIndex(self.dblp, self.scholar)
            except Exception as e:
                self._index_error = f"{type(e).__name__}: {e}"
        self._index_ready.set()

    # ---------------------------------------------------------------- data access
    @property
    def has_data(self):
        return self.dblp is not None

    @property
    def data_loaded(self):
        """True once data/ has been loaded successfully (never blocks)."""
        return self._data_ready.is_set() and self.dblp is not None

    @property
    def index_ready(self):
        """True once the background work has finished (successfully or not; never blocks)."""
        return self._index_ready.is_set()

    def status(self):
        if not self._data_ready.is_set():
            return "loading data/ …"
        if not self.has_data:
            return f"data/ not available ({self.data_error})"
        if not self._index_ready.is_set():
            return "data loaded; building the blocking index …"
        return "ready" if self._index is not None else f"blocking index failed ({self._index_error})"

    def wait_for_data(self, timeout=60):
        self._data_ready.wait(timeout)
        if not self.has_data:
            raise RuntimeError(f"data/ not available: {self.data_error or 'still loading'}")

    def index(self, timeout=120):
        """The blocking index (waits for the background build, normally a few seconds)."""
        self._index_ready.wait(timeout)
        if self._index is None:
            raise RuntimeError(f"blocking index unavailable: {self._index_error or 'still building'}")
        return self._index

    def record(self, source, record_id):
        """Prepared record by id from 'dblp' or 'scholar' (None if unknown)."""
        self.wait_for_data()
        table, idx = (self.dblp, self.d_idx) if source == "dblp" else (self.scholar, self.s_idx)
        record_id = str(record_id).strip()
        if record_id not in idx:
            return None
        return record_from_row(table.iloc[idx[record_id]])

    def search(self, source, text, limit=25):
        """Records of 'dblp' / 'scholar' whose title contains `text` (case-insensitive) or whose id equals it."""
        self.wait_for_data()
        table = self.dblp if source == "dblp" else self.scholar
        cols = ["id", "title", "authors", "venue", "year"]
        text = str(text or "").strip()
        if not text:
            return table.iloc[:0][cols]
        hit = (table["id"] == text) | table["title"].fillna("").str.contains(re.escape(text), case=False)
        return table.loc[hit, cols].head(limit)

    # ---------------------------------------------------------------- scoring one pair
    def score_pair(self, d, s, with_blocking=True):
        """Full result for one DBLP record d and one Scholar record s (prepared dicts)."""
        f = pair_features(d, s, self.alpha)
        sims = field_similarities(f)
        score, rows = weighted_score(sims, self.weights)
        ml = {}
        fdf = pd.DataFrame([f])
        for name, bundle in self.models.items():
            p = float(model_probability(bundle, fdf)[0])
            ml[name] = {"probability": None if math.isnan(p) else p, "threshold": float(bundle["threshold"]),
                        "param": bundle.get("param")}
        blocking = None
        if with_blocking and self._index_ready.is_set() and self._index is not None:
            ix = self._index
            d_in = d.get("id") in self.d_idx
            s_in = s.get("id") in self.s_idx
            dk = {p: ix.d_keys[p][self.d_idx[d["id"]]] for p in PASS_NAMES} if d_in else ix.record_keys(d)
            sk = {p: ix.s_keys[p][self.s_idx[s["id"]]] for p in PASS_NAMES} if s_in else ix.record_keys(s)
            blocking = ix.pair_check(dk, sk, s_external=not s_in)
        result = {"dblp": d, "scholar": s, "features": f, "sims": sims, "score": score, "contributions": rows,
                  "decision": decide(score, self.auto_t, self.non_t), "ml": ml, "blocking": blocking,
                  "in_gold": (d.get("id"), s.get("id")) in self.gold}
        result["explanation"] = explain(result, self.auto_t, self.non_t)
        return result

    # ---------------------------------------------------------------- query -> candidates
    def resolve(self, rec, side="dblp", passes=PASS_NAMES, top_k=20):
        """Entity resolution for one record: blocking -> features -> score -> ML -> decision.

        side="dblp": rec is a DBLP-side record, candidates come from Scholar; side="scholar": the reverse.
        Returns (info dict, DataFrame of the top_k candidates of the other source ranked by composite score).
        """
        ix = self.index()
        passes = [p for p in PASS_NAMES if p in set(passes)]
        own_idx, own_keys = (self.d_idx, ix.d_keys) if side == "dblp" else (self.s_idx, ix.s_keys)
        in_corpus = rec.get("id") in own_idx
        keys = {p: own_keys[p][own_idx[rec["id"]]] for p in PASS_NAMES} if in_corpus else ix.record_keys(rec)
        t0 = time.time()
        if side == "dblp":
            cand = ix.candidates(keys, passes)
            other, n_other = "scholar", ix.n_s
            gold_ids = {s for d, s in self.gold if d == rec.get("id")} if in_corpus else set()
        else:
            cand = ix.candidates_for_scholar(keys, passes, external=not in_corpus)
            other, n_other = "dblp", ix.n_d
            gold_ids = {d for d, s in self.gold if s == rec.get("id")} if in_corpus else set()
        t_block = time.time() - t0
        info = {"side": side, "other": other, "keys": {p: keys[p] for p in passes}, "candidates": len(cand),
                "other_records": n_other, "reduction_ratio": 1 - len(cand) / n_other, "blocking_s": t_block,
                "gold_matches": len(gold_ids)}
        if not cand:
            info.update(matching_s=0.0, decisions={k: 0 for k in DECISION_LABELS}, gold_in_candidates=0,
                        gold_auto_matched=0)
            return info, pd.DataFrame()
        t0 = time.time()
        rows = np.fromiter(cand.keys(), dtype=np.int64, count=len(cand))
        one = pd.DataFrame([{k: rec[k] for k in ("t_norm", "a_names", "v_norm", "yr")}])
        zeros = np.zeros(len(rows), dtype=np.int64)
        if side == "dblp":
            F = features_for_pairs(one, self.scholar, zeros, rows, self.alpha)
        else:
            F = features_for_pairs(self.dblp, one, rows, zeros, self.alpha)
        F["score"] = weighted_score_frame(F, self.weights)
        F["decision"] = decide_array(F["score"], self.auto_t, self.non_t)
        for name, bundle in self.models.items():
            F[name] = model_probability(bundle, F)
        info["matching_s"] = time.time() - t0
        tab = (self.scholar if side == "dblp" else self.dblp).iloc[rows]
        out = pd.DataFrame({
            f"{other}_id": tab["id"].to_numpy(), f"{other}_title": tab["title"].to_numpy(),
            f"{other}_authors": tab["authors"].to_numpy(), f"{other}_venue": tab["venue"].to_numpy(),
            f"{other}_year": tab["year"].astype(object).where(tab["year"].notna(), None).to_numpy(),
            "S_title": F["title_combined"].to_numpy(), "S_author": F[AUTHOR_MEASURE].to_numpy(),
            "S_venue": F[VENUE_MEASURE].to_numpy(), "S_year": F["year"].to_numpy(),
            "composite": F["score"].to_numpy()})
        for name in self.models:
            out[f"P({'LR' if name.startswith('Logistic') else 'DT'})"] = F[name].to_numpy()
        out["decision"] = F["decision"].to_numpy()
        out["blocking_passes"] = [", ".join(cand[r]) for r in rows]
        out["gold_match"] = [i in gold_ids for i in out[f"{other}_id"]]
        out = out.sort_values(["composite", f"{other}_id"], ascending=[False, True]).reset_index(drop=True)
        info["decisions"] = {k: int((out["decision"] == k).sum()) for k in DECISION_LABELS}
        info["gold_in_candidates"] = int(out["gold_match"].sum())
        info["gold_auto_matched"] = int((out["gold_match"] & (out["decision"] == AUTO_MATCH)).sum())
        return info, out.head(int(top_k))

    def blocking_report(self, passes=PASS_NAMES):
        """Run the chosen passes for all DBLP records: candidates, reduction ratio and blocking recall per pass."""
        ix = self.index()
        passes = [p for p in PASS_NAMES if p in set(passes)]
        t0 = time.time()
        union, per_pass, times = ix.run_all(passes)
        total = time.time() - t0
        gold_keys = self.gold_keys()
        p_all = ix.n_d * ix.n_s
        rows = []
        for p in passes:
            k = per_pass[p]
            rows.append({"pass": p, "candidates": len(k), "reduction_ratio": 1 - len(k) / p_all,
                         "blocking_recall": float(np.isin(gold_keys, k).mean()), "time_s": times[p]})
        rows.append({"pass": "UNION (" + " + ".join(passes) + ")", "candidates": len(union),
                     "reduction_ratio": 1 - len(union) / p_all,
                     "blocking_recall": float(np.isin(gold_keys, union).mean()) if len(union) else 0.0,
                     "time_s": total})
        return pd.DataFrame(rows)

    def gold_keys(self):
        n_s = len(self.scholar)
        return np.array(sorted(self.d_idx[d] * n_s + self.s_idx[s] for d, s in self.gold
                               if d in self.d_idx and s in self.s_idx), dtype=np.int64)


# ====================================================================== 11. stored candidates (review queue)
APP_DB_NAME = "entityguard_app.db"
STORED_COLUMNS = {"title": "title_similarity", "author": "author_similarity", "venue": "venue_similarity",
                  "year": "year_similarity"}


def load_stored_candidates(out_dir=OUT_DIR, weights=APP_WEIGHTS, auto_t=AUTO_MATCH_T, non_t=NON_MATCH_T):
    """Blocked candidate pairs scored by the notebook (MATCH_RESULT in outputs/entityguard_app.db),
    re-scored with `weights` and given a decision label.

    The app database keeps every candidate whose notebook score is >= 0.40, so every candidate at or above
    the 0.49 review boundary is in the table (`python -m src.pipeline verify` checks this against all
    462,813 candidates; it also holds for other weights within ~0.07 of the validated ones).
    Returns (DataFrame, meta dict); the DataFrame is empty if the database is missing.
    """
    import sqlite3
    path = os.path.join(out_dir, APP_DB_NAME)
    if not os.path.exists(path):
        cols = ["dblp_id", "scholar_id", *STORED_COLUMNS.values(), "lr_probability", "split", "in_gold",
                *(f"{side}_{f}" for side in ("dblp", "scholar") for f in ("title", "authors", "venue", "year")),
                "composite", "decision"]
        return pd.DataFrame(columns=cols), {"error": f"{path} not found (run the notebook to export it)"}
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql(
            "SELECT m.dblp_id, m.scholar_id, m.title_similarity, m.author_similarity, m.venue_similarity, "
            "m.year_similarity, m.lr_probability, m.split, "
            "d.title AS dblp_title, d.authors AS dblp_authors, d.venue AS dblp_venue, d.year AS dblp_year, "
            "s.title AS scholar_title, s.authors AS scholar_authors, s.venue AS scholar_venue, s.year AS scholar_year, "
            "CASE WHEN g.dblp_id IS NULL THEN 0 ELSE 1 END AS in_gold "
            "FROM MATCH_RESULT m JOIN DBLP_RECORD d ON d.record_id = m.dblp_id "
            "JOIN SCHOLAR_RECORD s ON s.record_id = m.scholar_id "
            "LEFT JOIN GOLD_MATCH g ON g.dblp_id = m.dblp_id AND g.scholar_id = m.scholar_id", con)
        meta = dict(con.execute("SELECT key, value FROM APP_META").fetchall())
        meta["test_gold"] = con.execute(
            "SELECT COUNT(*) FROM GOLD_MATCH g JOIN DBLP_RECORD d ON d.record_id = g.dblp_id "
            "WHERE d.split = 'test'").fetchone()[0]
        meta["gold"] = set(con.execute("SELECT dblp_id, scholar_id FROM GOLD_MATCH").fetchall())
    finally:
        con.close()
    df["composite"] = weighted_score_frame(df, weights, STORED_COLUMNS)
    df["decision"] = decide_array(df["composite"], auto_t, non_t)
    return df, meta


def stored_record(row, side):
    """Prepared record from a row of load_stored_candidates() ('dblp' or 'scholar' side)."""
    return prepare_record(row[f"{side}_title"], row[f"{side}_authors"], row[f"{side}_venue"],
                          row[f"{side}_year"], record_id=row[f"{side}_id"])


def decision_rule_check(stored, meta, split="test"):
    """How the decision bands behave on one split: pairs and true matches per band (gold = evaluation only)."""
    sub = stored[stored["split"] == split]
    n_gold = int(meta.get("test_gold", 0)) if split == "test" else int(sub["in_gold"].sum())
    rows = []
    for lab in (AUTO_MATCH, MANUAL_REVIEW):
        b = sub[sub["decision"] == lab]
        rows.append({"decision": lab, "pairs": len(b), "true matches": int(b["in_gold"].sum()),
                     "match rate (precision)": float(b["in_gold"].mean()) if len(b) else float("nan"),
                     "share of all true matches": int(b["in_gold"].sum()) / n_gold if n_gold else float("nan")})
    found = sum(r["true matches"] for r in rows)
    rows.append({"decision": NON_MATCH, "pairs": None, "true matches": n_gold - found,
                 "match rate (precision)": float("nan"),
                 "share of all true matches": (n_gold - found) / n_gold if n_gold else float("nan")})
    return pd.DataFrame(rows)


# ====================================================================== 12. human review store
REVIEW_COLS = ["dblp_id", "scholar_id", "decision", "note", "reviewer", "reviewed_at"]
REVIEW_DECISIONS = ("Confirm Match", "Reject Match", "Needs Review")   # Section 40 wording
_review_lock = threading.Lock()


def load_reviews(path):
    """All saved review decisions (latest decision per pair)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=REVIEW_COLS)
    r = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in REVIEW_COLS:
        if c not in r:
            r[c] = ""
    return r[REVIEW_COLS].drop_duplicates(["dblp_id", "scholar_id"], keep="last").reset_index(drop=True)


def save_review(path, dblp_id, scholar_id, decision, note="", reviewer="student3"):
    """Insert or replace one decision and rewrite the CSV atomically. Returns the saved row."""
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of {REVIEW_DECISIONS}")
    row = {"dblp_id": str(dblp_id), "scholar_id": str(scholar_id), "decision": decision, "note": note or "",
           "reviewer": reviewer or "student3", "reviewed_at": datetime.now().isoformat(timespec="seconds")}
    with _review_lock:
        r = load_reviews(path)
        r = r[~((r.dblp_id == row["dblp_id"]) & (r.scholar_id == row["scholar_id"]))]
        r = pd.concat([r, pd.DataFrame([row])], ignore_index=True)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        r.to_csv(tmp, index=False)
        os.replace(tmp, path)          # never leaves a half-written file behind
    return row


# ====================================================================== 12. command line: verify / train
def build_candidate_features(eng, verbose=True):
    """All blocked candidate pairs with features, score, label and split (as the notebook's FEAT table)."""
    ix = eng.index()
    t0 = time.time()
    union, per_pass, _ = ix.run_all()
    t_block = time.time() - t0
    n_s = len(eng.scholar)
    di, si = union // n_s, union % n_s
    t0 = time.time()
    F = features_for_pairs(eng.dblp, eng.scholar, di, si, eng.alpha)
    t_feat = time.time() - t0
    gk = eng.gold_keys()
    F["key"] = union
    F["label"] = np.isin(union, gk).astype(np.int8)       # evaluation / ML training only
    F["split"] = eng.dblp["split"].to_numpy()[di]
    if verbose:
        print(f"blocking: {len(union):,} candidates in {t_block:.1f}s | features in {t_feat:.1f}s")
    return F, per_pass, {"blocking_s": t_block, "features_s": t_feat}


def _report(name, got, expected):
    ok = "OK  " if expected is None or got == expected else "DIFF"
    print(f"  [{ok}] {name}: {got}" + ("" if expected is None or got == expected else f"  (notebook: {expected})"))
    return expected is None or got == expected


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild the EntityGuard candidate set and check / train models.")
    ap.add_argument("command", choices=["verify", "train", "export-models"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)

    if args.command == "export-models":          # scikit-learn .joblib -> portable .json (no data needed)
        import joblib
        for name, base in MODEL_FILES.items():
            src = os.path.join(args.out_dir, base + ".joblib")
            if not os.path.exists(src):
                print(f"skip {name}: {src} not found")
                continue
            print(f"{name}: wrote {save_model_json(joblib.load(src), os.path.join(args.out_dir, base + '.json'))}")
        return 0

    t0 = time.time()
    eng = EntityGuard(args.data_dir, args.out_dir, load_data=True)
    if not eng.has_data:
        raise SystemExit(eng.data_error)
    print(f"loaded + indexed {len(eng.dblp):,} DBLP / {len(eng.scholar):,} Scholar records in {time.time()-t0:.1f}s")
    F, per_pass, _ = build_candidate_features(eng)
    summ = eng.summary
    gk = eng.gold_keys()
    ok = True
    print("Checks against outputs/summary.json and outputs/method_comparison.csv:")
    ok &= _report("candidate pairs", len(F), summ.get("blocking", {}).get("candidates"))
    ok &= _report("blocking recall", round(float(np.isin(gk, F["key"]).mean()), 6),
                  round(summ["blocking"]["blocking_recall"], 6) if "blocking" in summ else None)

    n_gold = {s: int((eng.gold_df["split"] == s).sum()) for s in ("train", "val", "test")}
    tests = {r["method"].split()[0]: r for r in summ.get("test_results", [])}
    test = F[F.split == "test"].reset_index(drop=True)
    chosen = summ.get("chosen", {})
    if chosen:
        w = dict(zip(FIELDS, chosen["weights_title_author_venue_year"]))
        sc = weighted_score_frame(test, w)
        m = threshold_sweep(sc, test["label"], n_gold["test"], [float(chosen["threshold"])]).iloc[0]
        exp = tests.get("M4")
        ok &= _report("M4 weighted + blocking (TP, FP, FN)", (int(m.TP), int(m.FP), int(m.FN)),
                      (exp["TP"], exp["FP"], exp["FN"]) if exp else None)
    for name, key in (("Logistic Regression", "M5"), ("Decision Tree", "M5b")):
        if name in eng.models:
            b = eng.models[name]
            m = threshold_sweep(model_probability(b, test), test["label"], n_gold["test"], [b["threshold"]]).iloc[0]
            exp = tests.get(key)
            kind = "portable JSON" if isinstance(b["model"], PortableModel) else "joblib"
            ok &= _report(f"{key} saved {name} ({kind}) (TP, FP, FN)", (int(m.TP), int(m.FP), int(m.FN)),
                          (exp["TP"], exp["FP"], exp["FN"]) if exp else None)

    app = weighted_score_frame(F, eng.weights)
    app_dec = decide_array(app, eng.auto_t, eng.non_t)
    print(f"App decision rule {eng.weights} on the TEST split:")
    for lab in DECISION_LABELS:
        sel = (app_dec == lab) & (F["split"] == "test").to_numpy()
        print(f"  {lab:<14} pairs {int(sel.sum()):>7,} | true matches {int(F['label'].to_numpy()[sel].sum()):>5,}")
    stored, _ = load_stored_candidates(args.out_dir, eng.weights, eng.auto_t, eng.non_t)
    if len(stored):
        full = int((app_dec != NON_MATCH).sum())
        ok &= _report("app database holds every candidate with app score >= 0.49",
                      int((stored["decision"] != NON_MATCH).sum()), full)

    if args.command == "train":
        import joblib
        train = F[F.split == "train"].reset_index(drop=True)
        val = F[F.split == "val"].reset_index(drop=True)
        table, best = train_models(train, val, n_gold["val"])
        print(table.round(4).to_string(index=False))
        lr_saved = eng.models.get("Logistic Regression")
        if lr_saved:
            ok &= _report("retrained LR validation F1 equals saved bundle",
                          round(best["Logistic Regression"]["val_f1"], 6), round(lr_saved["val_f1"], 6))
        dt = best["Decision Tree"]
        m = threshold_sweep(model_probability(dt, test), test["label"], n_gold["test"], [dt["threshold"]]).iloc[0]
        exp = tests.get("M5b")
        ok &= _report(f"M5b Decision Tree {dt['param']} T={dt['threshold']} (TP, FP, FN)",
                      (int(m.TP), int(m.FP), int(m.FN)), (exp["TP"], exp["FP"], exp["FN"]) if exp else None)
        base = os.path.join(args.out_dir, MODEL_FILES["Decision Tree"])
        joblib.dump(dt, base + ".joblib")
        save_model_json(dt, base + ".json")
        print(f"saved {base}.joblib and {base}.json")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS DIFFER (see DIFF lines)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
