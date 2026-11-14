"""Data layer of the standalone web app.

The app loads ONLY saved artifacts (no notebook globals):
  outputs/entityguard_app.db   records, gold mapping, stored match results, experiment summary
  outputs/summary.json         alpha, weights, measures, threshold, bands (chosen on validation)
  outputs/lr_model.joblib      Logistic Regression bundle (optional)
  outputs/human_reviews.csv    review decisions -> imported into the REVIEW table at start-up

The app database is a slim copy of the full experiment database (export_app_db): the full one
(~180 MB, 462k candidate pairs) is too big for GitHub. MATCH_RESULT keeps the pairs with an
overall score >= min_score (default 0.40, which covers every review band and the whole threshold
range of the experiments). Any other pair is scored on demand with Student 1's record_similarity.

The database file is copied into memory at start-up and never modified; review decisions are
saved to human_reviews.csv, which is the permanent record.
"""
import json
import os
import sqlite3
import threading
import urllib.parse
from datetime import datetime

import numpy as np
import pandas as pd

from .. import similarity as S1                       # Student 1 (reference scoring)
from ..evaluation import BAND_NAMES, assign_band, prf
from ..matching import MatchConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "outputs")
APP_DB_NAME = "entityguard_app.db"
DECISIONS = ("Confirm Match", "Reject Match", "Needs Review")
REVIEW_COLS = ["dblp_id", "scholar_id", "decision", "note", "reviewer", "reviewed_at"]

APP_SCHEMA = """
CREATE TABLE DBLP_RECORD    (record_id TEXT PRIMARY KEY, title TEXT, authors TEXT, venue TEXT, year INTEGER, split TEXT);
CREATE TABLE SCHOLAR_RECORD (record_id TEXT PRIMARY KEY, title TEXT, authors TEXT, venue TEXT, year INTEGER);
CREATE TABLE GOLD_MATCH     (dblp_id TEXT REFERENCES DBLP_RECORD, scholar_id TEXT REFERENCES SCHOLAR_RECORD,
                             PRIMARY KEY (dblp_id, scholar_id));
CREATE TABLE MATCH_RESULT   (dblp_id TEXT REFERENCES DBLP_RECORD, scholar_id TEXT REFERENCES SCHOLAR_RECORD,
                             title_similarity REAL, author_similarity REAL, venue_similarity REAL, year_similarity REAL,
                             overall_similarity REAL, lr_probability REAL, prediction INTEGER, band TEXT, split TEXT,
                             PRIMARY KEY (dblp_id, scholar_id));
CREATE TABLE REVIEW         (dblp_id TEXT, scholar_id TEXT,
                             decision TEXT CHECK (decision IN ('Confirm Match','Reject Match','Needs Review')),
                             note TEXT, reviewer TEXT, reviewed_at TEXT, PRIMARY KEY (dblp_id, scholar_id));
CREATE TABLE EXPERIMENT     (name TEXT PRIMARY KEY, result_json TEXT, created_at TEXT);
CREATE TABLE APP_META       (key TEXT PRIMARY KEY, value TEXT);
"""


def export_app_db(full_db_path, app_db_path, min_score=0.40):
    """Write the slim app database from the full experiment database. Returns the file size in MB."""
    if os.path.exists(app_db_path):
        os.remove(app_db_path)
    con = sqlite3.connect(app_db_path)
    con.executescript(APP_SCHEMA)
    con.execute("ATTACH DATABASE ? AS full", (full_db_path,))
    for t in ("DBLP_RECORD", "SCHOLAR_RECORD", "GOLD_MATCH", "EXPERIMENT"):
        con.execute(f"INSERT INTO {t} SELECT * FROM full.{t}")
    con.execute("INSERT INTO MATCH_RESULT SELECT * FROM full.MATCH_RESULT WHERE overall_similarity >= ?", (min_score,))
    n_all = con.execute("SELECT COUNT(*) FROM full.MATCH_RESULT").fetchone()[0]
    con.executemany("INSERT INTO APP_META VALUES (?, ?)", [
        ("min_score", str(min_score)), ("candidate_pairs_total", str(n_all)),
        ("created_at", datetime.now().isoformat(timespec="seconds"))])
    con.execute("CREATE INDEX idx_match_band ON MATCH_RESULT(band, overall_similarity)")
    con.execute("CREATE INDEX idx_match_dblp ON MATCH_RESULT(dblp_id)")
    con.commit()
    con.execute("DETACH DATABASE full")
    con.execute("VACUUM")
    con.close()
    return os.path.getsize(app_db_path) / 1e6


def _clean(v):
    """JSON-safe value: NaN/None -> None, numpy -> python."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    return v


class Store:
    def __init__(self, out_dir=DEFAULT_OUT_DIR, db_path=None, review_csv=None, model_path=None):
        self.out_dir = out_dir
        self.db_path = db_path or os.path.join(out_dir, APP_DB_NAME)
        self.review_csv = review_csv or os.path.join(out_dir, "human_reviews.csv")
        model_path = model_path or os.path.join(out_dir, "lr_model.joblib")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"{self.db_path} not found. Run the experiment notebook (it exports the app database), "
                "or export it from a full database with src.api.store.export_app_db().")
        self.lock = threading.RLock()

        disk = sqlite3.connect(self.db_path)
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        disk.backup(self.con)                      # work on an in-memory copy; the file stays unchanged
        disk.close()

        with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as f:
            self.summary = json.load(f)
        self.cfg = MatchConfig.from_summary(self.summary)
        self.model = None
        if os.path.exists(model_path):
            from ..ml import load_model
            self.model = load_model(model_path)

        meta = dict(self.con.execute("SELECT key, value FROM APP_META").fetchall())
        self.min_score = float(meta.get("min_score", 0.0))
        self.n_candidates = int(meta.get("candidate_pairs_total", 0))
        self.dblp_ids = {r[0] for r in self.con.execute("SELECT record_id FROM DBLP_RECORD")}
        self.scholar_ids = {r[0] for r in self.con.execute("SELECT record_id FROM SCHOLAR_RECORD")}
        self.gold = set(self.con.execute("SELECT dblp_id, scholar_id FROM GOLD_MATCH").fetchall())
        self.split = dict(self.con.execute("SELECT record_id, split FROM DBLP_RECORD").fetchall())
        self.n_dblp, self.n_scholar = len(self.dblp_ids), len(self.scholar_ids)
        self._import_reviews()
        self._blocking_data = None

    # ------------------------------------------------------------------ basics
    def sql(self, query, params=()):
        with self.lock:
            return pd.read_sql(query, self.con, params=params)

    def record(self, source, record_id):
        table = "DBLP_RECORD" if source == "dblp" else "SCHOLAR_RECORD"
        with self.lock:
            cur = self.con.execute(f"SELECT * FROM {table} WHERE record_id = ?", (record_id,))
            row = cur.fetchone()
            cols = [c[0] for c in cur.description]
        return None if row is None else {k: _clean(v) for k, v in zip(cols, row)}

    def records(self, source, offset=0, limit=20):
        table = "DBLP_RECORD" if source == "dblp" else "SCHOLAR_RECORD"
        df = self.sql(f"SELECT record_id, title, authors, venue, year FROM {table} ORDER BY rowid LIMIT ? OFFSET ?",
                      (int(limit), int(offset)))
        return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]

    def resolve_ids(self, dblp_id, scholar_id):
        """Accept ids even if a client URL-decoded the %3F / %26 / %3D inside some Scholar ids."""
        if scholar_id not in self.scholar_ids:
            alt = urllib.parse.quote(scholar_id, safe=":/")
            if alt in self.scholar_ids:
                scholar_id = alt
        return dblp_id, scholar_id

    # ------------------------------------------------------------------ pair scores
    def pair(self, dblp_id, scholar_id):
        """Records side by side + per-field similarities + overall score, band, LR probability.

        Stored pairs come from MATCH_RESULT. Other pairs are scored now with Student 1's
        record_similarity (same alpha / weights / measures as the experiments).
        """
        dblp_id, scholar_id = self.resolve_ids(dblp_id, scholar_id)
        a, b = self.record("dblp", dblp_id), self.record("scholar", scholar_id)
        if a is None or b is None:
            return None
        with self.lock:
            cur = self.con.execute("SELECT * FROM MATCH_RESULT WHERE dblp_id=? AND scholar_id=?", (dblp_id, scholar_id))
            row = cur.fetchone()
            cols = [c[0] for c in cur.description]
        if row is not None:
            m = dict(zip(cols, row))
            scores = {k: _clean(m[k]) for k in ("title_similarity", "author_similarity", "venue_similarity",
                                                 "year_similarity", "overall_similarity", "lr_probability")}
            scores.update(band=m["band"], prediction=int(m["prediction"]), source="stored (MATCH_RESULT)")
        else:
            scores = self._score_live(a, b)
        prev = self.review_of(dblp_id, scholar_id)
        return {"dblp": a, "scholar": b, **scores, "threshold": self.cfg.threshold,
                "candidate": row is not None, "review": prev}

    def _score_live(self, a, b):
        cfg = self.cfg
        weights = dict(zip(("title", "author", "venue", "year"), cfg.weights))
        if cfg.author_measure in ("author_best", "author_tok") and cfg.venue_measure == "venue_tri":
            overall, parts = S1.record_similarity(
                a["title"], a["authors"], a["venue"], a["year"], b["title"], b["authors"], b["venue"], b["year"],
                author_method="best_match" if cfg.author_measure == "author_best" else "jaccard",
                alpha=cfg.alpha, weights=weights)
            f = None
        else:                                      # a Student 3 extension measure was chosen
            f = self._features(a, b)
            parts = {"title": f["title_combined"], "author": f[cfg.author_measure],
                     "venue": f[cfg.venue_measure], "year": f["year"]}
            parts = {k: (None if v is None or np.isnan(v) else float(v)) for k, v in parts.items()}
            overall = S1.combine_scores(parts, weights)
        lr = None
        if self.model is not None:
            from ..ml import predict_proba
            f = f if f is not None else self._features(a, b)
            lr = float(predict_proba(self.model, pd.DataFrame([f]))[0])
        return {"title_similarity": _clean(parts["title"]), "author_similarity": _clean(parts["author"]),
                "venue_similarity": _clean(parts["venue"]), "year_similarity": _clean(parts["year"]),
                "overall_similarity": float(overall), "lr_probability": lr,
                "band": str(assign_band([overall], cfg.band_high, cfg.band_low)[0]),
                "prediction": int(overall >= cfg.threshold), "source": "computed now (Student 1 record_similarity)"}

    def _features(self, a, b):
        """All fast features for one pair (needed for the ML model and for extension measures)."""
        from ..matching.features import FeatureBuilder, add_normalized_columns
        cols = ["title", "authors", "venue", "year"]
        d = add_normalized_columns(pd.DataFrame([{c: a[c] for c in cols}]))
        s = add_normalized_columns(pd.DataFrame([{c: b[c] for c in cols}]))
        f = FeatureBuilder(d, s).compute([0], [0]).iloc[0].to_dict()
        f["title_combined"] = self.cfg.alpha * f["title_edit"] + (1 - self.cfg.alpha) * f["title_jac"]
        return f

    # ------------------------------------------------------------------ matching / matches
    def matches(self, band=None, limit=50, dblp_ids=None, min_score=None):
        q = ("SELECT m.dblp_id, m.scholar_id, d.title AS dblp_title, s.title AS scholar_title, "
             "m.title_similarity, m.author_similarity, m.venue_similarity, m.year_similarity, "
             "m.overall_similarity, m.lr_probability, m.band, m.split FROM MATCH_RESULT m "
             "JOIN DBLP_RECORD d ON d.record_id = m.dblp_id JOIN SCHOLAR_RECORD s ON s.record_id = m.scholar_id WHERE 1=1")
        params = []
        if band:
            q += " AND m.band = ?"; params.append(band)
        if min_score is not None:
            q += " AND m.overall_similarity >= ?"; params.append(float(min_score))
        if dblp_ids is not None:
            dblp_ids = list(dblp_ids)[:200]
            q += f" AND m.dblp_id IN ({','.join('?' * len(dblp_ids)) or 'NULL'})"; params += dblp_ids
        q += " ORDER BY m.overall_similarity DESC LIMIT ?"; params.append(int(limit))
        return self.sql(q, params)

    def run_matching(self, threshold):
        """Apply threshold T to the stored scores; P/R/F1 on the TEST split against all its gold pairs."""
        threshold = float(threshold)
        if threshold < self.min_score:
            raise ValueError(f"the app database stores pairs with score >= {self.min_score}; "
                             f"choose a threshold >= {self.min_score}")
        pred = self.sql("SELECT dblp_id, scholar_id, split FROM MATCH_RESULT WHERE overall_similarity >= ?", (threshold,))
        test_pred = pred[pred.split == "test"]
        test_gold = {p for p in self.gold if self.split.get(p[0]) == "test"}
        m = prf(zip(test_pred.dblp_id, test_pred.scholar_id), test_gold)
        bands = assign_band(np.array([threshold]), self.cfg.band_high, self.cfg.band_low)[0]
        return {"threshold": threshold, "predicted_matches": int(len(pred)),
                "predicted_matches_test": int(len(test_pred)), "threshold_band": str(bands), "test": m}

    # ------------------------------------------------------------------ blocking (on demand)
    def run_blocking(self, scholar_fraction=1.0, passes=None, seed=42):
        """Re-run the reference blocking on a reproducible sample of Scholar. Needs data/ (first call ~5 s)."""
        from ..matching.blocking_reference import PASS_NAMES, run_blocking
        import time
        if not 0 < scholar_fraction <= 1:
            raise ValueError("scholar_fraction must be in (0, 1]")
        passes = tuple(passes) if passes else PASS_NAMES
        dblp, scholar, gold_keys = self._load_blocking_data()
        n_s = len(scholar)
        s_rows = np.sort(np.random.default_rng(seed).permutation(n_s)[: int(round(scholar_fraction * n_s))])
        t0 = time.time()
        keys, per_pass, _ = run_blocking(dblp, scholar, s_rows, passes)
        seconds = time.time() - t0
        g = gold_keys[np.isin(gold_keys % n_s, s_rows)]
        cart = len(dblp) * len(s_rows)
        return {"scholar_fraction": scholar_fraction, "scholar_records": int(len(s_rows)), "passes": list(passes),
                "cartesian_pairs": int(cart), "candidates": int(len(keys)),
                "per_pass": {k: int(len(v)) for k, v in per_pass.items()},
                "reduction_ratio": 1 - len(keys) / cart, "blocking_recall": float(np.isin(g, keys).mean()) if len(g) else None,
                "seconds": round(seconds, 2)}

    def _load_blocking_data(self):
        with self.lock:
            if self._blocking_data is None:
                from ..data_loading import load_datasets
                from ..matching.features import add_normalized_columns
                dblp, scholar, gold = load_datasets()
                dblp, scholar = add_normalized_columns(dblp), add_normalized_columns(scholar)
                d_idx = {k: i for i, k in enumerate(dblp.id)}; s_idx = {k: i for i, k in enumerate(scholar.id)}
                gk = np.array(sorted(d_idx[d] * len(scholar) + s_idx[s] for d, s in zip(gold.dblp_id, gold.scholar_id)),
                              dtype=np.int64)
                self._blocking_data = (dblp, scholar, gk)
            return self._blocking_data

    # ------------------------------------------------------------------ reviews
    def _import_reviews(self):
        with self.lock:
            self.con.execute("DELETE FROM REVIEW")
            if os.path.exists(self.review_csv):
                r = pd.read_csv(self.review_csv, dtype=str, keep_default_na=False)
                r = r[r.decision.isin(DECISIONS)].drop_duplicates(["dblp_id", "scholar_id"], keep="last")
                self.con.executemany(f"INSERT INTO REVIEW ({', '.join(REVIEW_COLS)}) VALUES (?,?,?,?,?,?)",
                                     r[REVIEW_COLS].values.tolist())
            self.con.commit()

    def review_of(self, dblp_id, scholar_id):
        with self.lock:
            row = self.con.execute("SELECT decision, note, reviewer, reviewed_at FROM REVIEW WHERE dblp_id=? AND scholar_id=?",
                                   (dblp_id, scholar_id)).fetchone()
        return None if row is None else dict(zip(("decision", "note", "reviewer", "reviewed_at"), row))

    def save_review(self, dblp_id, scholar_id, decision, note="", reviewer="student3"):
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        dblp_id, scholar_id = self.resolve_ids(dblp_id, scholar_id)
        if dblp_id not in self.dblp_ids or scholar_id not in self.scholar_ids:
            raise KeyError("unknown dblp_id or scholar_id")
        row = (dblp_id, scholar_id, decision, note or "", reviewer or "student3", datetime.now().isoformat(timespec="seconds"))
        with self.lock:
            self.con.execute(f"INSERT OR REPLACE INTO REVIEW ({', '.join(REVIEW_COLS)}) VALUES (?,?,?,?,?,?)", row)
            self.con.commit()
            all_rows = pd.read_sql("SELECT * FROM REVIEW ORDER BY reviewed_at", self.con)
            tmp = self.review_csv + ".tmp"
            all_rows[REVIEW_COLS].to_csv(tmp, index=False)
            os.replace(tmp, self.review_csv)       # the CSV is the permanent record
        return dict(zip(REVIEW_COLS, row))

    def reviews(self):
        r = self.sql("SELECT * FROM REVIEW ORDER BY reviewed_at DESC")
        r["in_gold"] = [(d, s) in self.gold for d, s in zip(r.dblp_id, r.scholar_id)]
        dec = r[r.decision != "Needs Review"]
        agree = float(((dec.decision == "Confirm Match") == dec.in_gold).mean()) if len(dec) else None
        counts = {d: int((r.decision == d).sum()) for d in DECISIONS}
        return r, {"reviews": int(len(r)), **counts, "agreement_with_gold": agree}

    def review_queue(self, band, limit=1):
        """Unreviewed stored pairs of a band, the most uncertain (closest to T) first."""
        if band not in BAND_NAMES:
            raise ValueError(f"band must be one of {BAND_NAMES}")
        return self.sql(
            "SELECT m.dblp_id, m.scholar_id, m.overall_similarity FROM MATCH_RESULT m "
            "LEFT JOIN REVIEW r ON r.dblp_id = m.dblp_id AND r.scholar_id = m.scholar_id "
            "WHERE m.band = ? AND r.dblp_id IS NULL ORDER BY ABS(m.overall_similarity - ?) LIMIT ?",
            (band, self.cfg.threshold, int(limit)))

    def evaluation(self):
        _, rev = self.reviews()
        return {**self.summary, "human_review": rev,
                "app_database": {"stored_pairs": int(self.con.execute("SELECT COUNT(*) FROM MATCH_RESULT").fetchone()[0]),
                                 "min_stored_score": self.min_score, "candidate_pairs_total": self.n_candidates}}
