"""Legacy reference multi-pass blocking retained for regression comparisons.

The application now uses ``src.blocking`` as its canonical implementation. This module remains
available for comparison tests and historical benchmark reproduction. It accepts normalized
frames; raw frames are normalized on entry for backwards-compatible test and experiment use.

Each pass produces a candidate set C_i; the final set is C = C_1 ∪ ... ∪ C_k (duplicates removed).
Pairs are integer keys  dblp_row * n_scholar + scholar_row  (a fast stand-in for a set of
(dblp_id, scholar_id) tuples). Document frequencies come from the data only, never from gold.

Passes:
  title_prefix  first significant title token (Section 27)
  rare_tokens   the 3 rarest title tokens, skipping tokens in > 1000 Scholar titles (Section 28)
  char_prefix   first 3 characters of the first 2 tokens (Section 29)
  rare_char6    5 rarest shared character 6-grams of the title without spaces, so glued
                Scholar titles ("TheGlue-Naildeductive...") are still caught (Section 29)
Year blocking is NOT a pass: 54 % of Scholar records have no year.
"""
import math
import time
from collections import Counter, defaultdict

import numpy as np

from src.matching.features import add_normalized_columns

PASS_NAMES = ("title_prefix", "rare_tokens", "char_prefix", "rare_char6")


def _build_blocks(lists, key_fn):
    blocks = defaultdict(list)
    for i, toks in enumerate(lists):
        for k in set(key_fn(toks)):
            blocks[k].append(i)
    return blocks


def _block_pairs(d_lists, s_lists, s_map, n_s, key_fn, max_block=None):
    """candidate keys for one pass; s_map maps local scholar position -> global scholar row"""
    bd, bs = _build_blocks(d_lists, key_fn), _build_blocks(s_lists, key_fn)
    keys = []
    for k, d_rows in bd.items():
        s_rows = bs.get(k)
        if not s_rows or (max_block and len(s_rows) > max_block):
            continue
        s_glob = s_map[np.asarray(s_rows)]
        for d in d_rows:
            keys.append(d * n_s + s_glob)
    return np.unique(np.concatenate(keys)) if keys else np.array([], dtype=np.int64)


def run_blocking(dblp, scholar, s_rows=None, passes=PASS_NAMES):
    """Multi-pass blocking of all DBLP rows against Scholar rows s_rows (default: all).

    dblp / scholar need the columns t_tokens and t_squash.
    Returns (union_keys, {pass: keys}, {pass: seconds}).
    """
    if not {"t_tokens", "t_squash"}.issubset(dblp.columns):
        dblp = dblp.copy()
        for column in ("authors", "venue", "year"):
            if column not in dblp:
                dblp[column] = None
        add_normalized_columns(dblp)
    if not {"t_tokens", "t_squash"}.issubset(scholar.columns):
        scholar = scholar.copy()
        for column in ("authors", "venue", "year"):
            if column not in scholar:
                scholar[column] = None
        add_normalized_columns(scholar)

    n_s = len(scholar)
    s_rows = np.arange(n_s) if s_rows is None else np.asarray(s_rows)
    d_tok = dblp.t_tokens.to_list(); s_tok = scholar.t_tokens.iloc[s_rows].to_list()
    d_sq = dblp.t_squash.to_list(); s_sq = scholar.t_squash.iloc[s_rows].to_list()
    tok_df = Counter(t for toks in d_tok + s_tok for t in set(toks))
    n_docs = len(d_tok) + len(s_tok)
    rarity = lambda t: math.log(n_docs / (tok_df[t] + 1))

    def gram_set(s, n=6):
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    available = {
        "title_prefix": lambda: (d_tok, s_tok, lambda t: t[:1], None),
        "rare_tokens": lambda: (d_tok, s_tok, lambda t: sorted(set(t), key=lambda x: (-rarity(x), x))[:3], 1000),
        "char_prefix": lambda: (d_tok, s_tok, lambda t: ["".join(x[:3] for x in t[:2])] if t else [], None),
        "rare_char6": None,
    }
    if "rare_char6" in passes:
        d_gr = [gram_set(s) for s in d_sq]; s_gr = [gram_set(s) for s in s_sq]
        gram_df = Counter(g for gs in d_gr + s_gr for g in gs)
        available["rare_char6"] = lambda: (
            d_gr, s_gr, lambda gs: sorted([g for g in gs if gram_df[g] >= 2], key=lambda g: (gram_df[g], g))[:5], 500)

    result, timings = {}, {}
    for name in passes:
        if name not in available:
            raise ValueError(f"unknown blocking pass {name!r}; choose from {PASS_NAMES}")
        dl, sl, fn, mb = available[name]()
        t0 = time.time()
        result[name] = _block_pairs(dl, sl, s_rows, n_s, fn, mb)
        timings[name] = time.time() - t0
    t0 = time.time()
    union = np.unique(np.concatenate(list(result.values()))) if result else np.array([], dtype=np.int64)
    timings["union"] = time.time() - t0
    return union, result, timings


def keys_to_pairs(keys, n_s):
    return keys // n_s, keys % n_s


def candidate_pairs(dblp, scholar, s_rows=None, passes=PASS_NAMES):
    """The interface expected from Student 2: a set of (dblp_id, scholar_id) pairs."""
    keys, _, _ = run_blocking(dblp, scholar, s_rows, passes)
    di, si = keys_to_pairs(keys, len(scholar))
    return set(zip(dblp.id.to_numpy()[di], scholar.id.to_numpy()[si]))
