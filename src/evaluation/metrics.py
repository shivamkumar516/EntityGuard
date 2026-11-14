
import numpy as np
import pandas as pd

def prf(pred_pairs, gold_pairs):

    pred_pairs, gold_pairs = set(pred_pairs), set(gold_pairs)
    tp = len(pred_pairs & gold_pairs); fp = len(pred_pairs) - tp; fn = len(gold_pairs) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "f1": f}

def blocking_metrics(cand_pairs, gold_pairs, total_pairs):
    cand_pairs = set(cand_pairs)
    hit = len(cand_pairs & set(gold_pairs))
    return {"candidates": len(cand_pairs), "reduction_ratio": 1 - len(cand_pairs) / total_pairs,
            "blocking_recall": hit / len(gold_pairs), "gold_in_candidates": hit}

def threshold_sweep(scores, labels, n_gold, thresholds):
    """P/R/F1 for each threshold. scores/labels are for candidate pairs; n_gold counts ALL gold pairs of the split."""
    scores = np.asarray(scores); labels = np.asarray(labels).astype(bool)
    rows = []
    for T in thresholds:
        pred = scores >= T
        tp = int((pred & labels).sum()); fp = int(pred.sum()) - tp; fn = n_gold - tp
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / n_gold if n_gold else 0.0
        rows.append({"threshold": T, "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return pd.DataFrame(rows)

def best_threshold(scores, labels, n_gold, grid=np.round(np.arange(0.30, 0.996, 0.005), 3)):
    sw = threshold_sweep(scores, labels, n_gold, grid)
    best = sw.loc[sw.f1.idxmax()]
    return float(best.threshold), float(best.f1)

def fast_best_f1(scores, labels, n_gold):
    """max F1 over every possible threshold (sort-based, used inside grid searches)"""
    order = np.argsort(-scores)
    tp = np.cumsum(labels[order]); k = np.arange(1, len(order) + 1)
    p = tp / k; r = tp / n_gold
    f = np.where(p + r > 0, 2 * p * r / (p + r), 0)
    return float(f.max())

