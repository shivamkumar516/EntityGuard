"""Optional ML extension (Sections 36-38): Logistic Regression and Decision Tree.

Rules followed (from the problem statement):
* scikit-learn on CPU only; no deep learning, embeddings or LLMs.
* Train ONLY on candidate pairs of the TRAIN split.
* Keep every positive; sample negatives reproducibly (random_state=42), 20 per positive by default.
* Choose hyper-parameters and the probability threshold on VALIDATION (full, unsampled).
* The test split is not touched here; it is scored once, afterwards, in the notebook.

Feature vector X = [S_title, S_author, S_venue, S_year, TitleLengthRatio,
                    author_missing, venue_missing, year_missing]
A missing similarity becomes 0 and its *_missing flag becomes 1, so the model can learn how much
a missing field should count, instead of us fixing that by hand.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..evaluation.metrics import best_threshold

SEED = 42
FEATURE_NAMES = ["S_title", "S_author", "S_venue", "S_year", "TitleLengthRatio",
                 "author_missing", "venue_missing", "year_missing"]
LR_C_GRID = (0.01, 0.1, 1, 10)
DT_DEPTH_GRID = (3, 4, 5, 6, 8, 10)
PROB_GRID = np.round(np.arange(0.05, 0.996, 0.01), 2)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "outputs", "lr_model.joblib")


def make_X(df, author_col, venue_col):
    """Feature matrix from a feature table (columns title_combined, <author_col>, <venue_col>, year, len_ratio)."""
    a, v, y = df[author_col].astype(float), df[venue_col].astype(float), df["year"].astype(float)
    return np.column_stack([df["title_combined"].astype(float), np.nan_to_num(a), np.nan_to_num(v),
                            np.nan_to_num(y), np.nan_to_num(df["len_ratio"].astype(float)),
                            a.isna(), v.isna(), y.isna()]).astype(np.float32)


def sample_training_pairs(train, neg_per_pos=20, seed=SEED):
    """All positives + a reproducible sample of negatives, shuffled."""
    pos = train[train["label"] == 1]
    neg = train[train["label"] == 0]
    neg_s = neg.sample(min(len(neg), neg_per_pos * len(pos)), random_state=seed)
    return pd.concat([pos, neg_s]).sample(frac=1, random_state=seed)


def train_models(train, val, n_gold_val, author_col, venue_col, neg_per_pos=20, seed=SEED):
    """Fit every LR / DT setting on sampled TRAIN pairs and pick the best of each on VALIDATION.

    n_gold_val = number of gold pairs in the validation split (including those blocking missed),
    so validation recall is honest.

    Returns (results_table, best) where best = {"LR": {...}, "DT": {...}} and each entry holds
    model, threshold, param and val_f1.
    """
    tr = sample_training_pairs(train, neg_per_pos, seed)
    X_tr, y_tr = make_X(tr, author_col, venue_col), tr["label"].to_numpy()
    X_va, y_va = make_X(val, author_col, venue_col), val["label"].to_numpy()

    candidates = [("LR", "Logistic Regression", f"C={C}",
                   make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed)))
                  for C in LR_C_GRID]
    candidates += [("DT", "Decision Tree", f"max_depth={d}",
                    DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=seed))
                   for d in DT_DEPTH_GRID]

    rows, best = [], {}
    for key, name, param, model in candidates:
        model.fit(X_tr, y_tr)
        T, f = best_threshold(model.predict_proba(X_va)[:, 1], y_va, n_gold_val, grid=PROB_GRID)
        rows.append({"model": name, "param": param, "val F1": f, "threshold": T})
        # ">=" keeps the LAST setting among ties, as the notebook's sort_values(...).iloc[-1] did
        if key not in best or f >= best[key]["val_f1"]:
            best[key] = {"model": model, "threshold": T, "param": param, "val_f1": f}

    info = {"train_pairs": len(tr), "positives": int((tr["label"] == 1).sum()),
            "negatives_sampled": int((tr["label"] == 0).sum()),
            "negatives_available": int((train["label"] == 0).sum())}
    return pd.DataFrame(rows), best, info


def save_model(entry, author_col, venue_col, path=DEFAULT_MODEL_PATH):
    """Save model + everything needed to use it (threshold, feature recipe) in one joblib file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {"model": entry["model"], "threshold": float(entry["threshold"]), "param": entry["param"],
              "val_f1": float(entry["val_f1"]), "feature_names": FEATURE_NAMES,
              "author_col": author_col, "venue_col": venue_col}
    joblib.dump(bundle, path)
    return path


def load_model(path=DEFAULT_MODEL_PATH):
    return joblib.load(path)


def predict_proba(bundle, df):
    """Match probability for each row of a feature table, using a loaded bundle."""
    return bundle["model"].predict_proba(make_X(df, bundle["author_col"], bundle["venue_col"]))[:, 1]
