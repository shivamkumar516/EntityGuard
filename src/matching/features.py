"""Fast (vectorized) similarity features for many candidate pairs at once.

WHY: Student 1's pure-Python functions in src/similarity.py are the *reference* definitions
(Sections 15-24), but scoring ~460k candidate pairs with them takes many minutes, and all
168M pairs would take days. This module computes the SAME numbers much faster:

* edit similarity   -> rapidfuzz's C implementation of 1 - Levenshtein / max(len)
* Jaccard (tokens / char n-grams) -> binary sparse matrices: |A∩B| = row dot product,
                                     |A∪B| = |A| + |B| - |A∩B|

Every feature below is defined to be EQUAL to the matching Student 1 function, including when
it returns None (a missing field). tests/test_features.py checks this on tricky records.

    feature        Student 1 reference                         note
    title_exact    similarity.title_exact
    title_edit     similarity.title_edit_similarity
    title_jac      similarity.title_token_jaccard
    title_bi/tri/4g similarity.title_ngram_jaccard(n=2/3/4)
    author_tok     similarity.author_similarity_jaccard
    author_best    similarity.author_similarity_best_match
    venue_exact    similarity.venue_exact
    venue_tri      similarity.venue_similarity(n=3)
    year           similarity.year_similarity
    len_ratio      similarity.title_length_ratio
    author_last    -- Student 3 extension: Jaccard of last names
    venue_tok      -- Student 3 extension: venue token Jaccard without generic words
                      ("proceedings", "international", ...) plus an acronym token,
                      so "International Conference on Data Engineering" ~ "ICDE"

All text normalization comes from Student 1's src/normalize.py.
"""
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as RFLevenshtein
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from normalize import author_tokens, normalize_title, normalize_venue, parse_year, split_author_names  # noqa: E402
from similarity import char_ngrams  # noqa: E402

# ---------------------------------------------------------------- Student 3 extensions
# generic words that say nothing about WHICH venue it is
_GENERIC = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "by", "from", "at", "as", "is",
            "are", "its", "into", "via", "using", "based", "towards", "toward"}
VENUE_STOP = _GENERIC | {"proc", "proceedings", "conf", "conference", "international", "intl", "annual",
                         "symposium", "workshop", "acm", "ieee", "th", "st", "nd", "rd"}


def venue_tokens(venue_norm):
    """Venue tokens without generic words, plus an acronym of the remaining words.

    "international conference on data engineering" -> ["data", "engineering", "icde"]
    "proc icde"                                     -> ["icde"]
    """
    raw = [t for t in venue_norm.split() if not t.isdigit()]
    toks = [t for t in raw if t not in VENUE_STOP]
    acr_words = [t for t in raw if t not in _GENERIC and t not in {"proc", "proceedings", "th"}]
    acr = "".join(t[0] for t in acr_words) if len(acr_words) >= 2 else ""
    return toks + ([acr] if acr else [])


def last_name(name):
    """'j d ullman' -> 'ullman' (longest-surviving token at the end; initials are skipped)."""
    toks = name.split()
    long = [t for t in toks if len(t) > 1]
    return long[-1] if long else toks[-1]


# ---------------------------------------------------------------- normalization (Student 1)
def add_normalized_columns(df):
    """Add the normalized fields used by blocking and by the features. Returns df (modified in place)."""
    df["t_norm"] = df["title"].map(normalize_title)
    df["t_tokens"] = df["t_norm"].str.split()
    df["t_squash"] = df["t_tokens"].map("".join)      # for blocking: robust to glued words in Scholar titles
    df["a_names"] = df["authors"].map(split_author_names)
    df["a_tokens"] = df["authors"].map(lambda a: sorted(set(author_tokens(a))))
    df["a_last"] = df["a_names"].map(lambda ns: sorted({last_name(n) for n in ns}))
    df["v_norm"] = df["venue"].map(normalize_venue)
    df["v_tokens"] = df["v_norm"].map(venue_tokens)
    df["yr"] = df["year"].map(parse_year).astype(float)  # NaN = missing or junk year
    return df


# ---------------------------------------------------------------- vectorized Jaccard
class SetSpace:
    """Binary sparse matrices so Jaccard can be computed for many pairs at once."""

    def __init__(self, d_lists, s_lists):
        d_lists, s_lists = list(d_lists), list(s_lists)
        if any(len(x) for x in d_lists + s_lists):
            cv = CountVectorizer(analyzer=lambda x: x, binary=True, dtype=np.float32)
            cv.fit(d_lists + s_lists)
            self.D = cv.transform(d_lists).tocsr()
            self.S = cv.transform(s_lists).tocsr()
        else:                       # no tokens at all (e.g. one pair, both venues missing)
            self.D = sparse.csr_matrix((len(d_lists), 1), dtype=np.float32)
            self.S = sparse.csr_matrix((len(s_lists), 1), dtype=np.float32)
        self.d_size = np.asarray(self.D.sum(1)).ravel()
        self.s_size = np.asarray(self.S.sum(1)).ravel()

    def pairs(self, di, si, chunk=100_000):
        """Jaccard for the pairs (di[k], si[k]); NaN when both sets are empty."""
        out = np.empty(len(di), dtype=np.float32)
        for k in range(0, len(di), chunk):
            a, b = di[k:k + chunk], si[k:k + chunk]
            inter = np.asarray(self.D[a].multiply(self.S[b]).sum(1)).ravel()
            union = self.d_size[a] + self.s_size[b] - inter
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k:k + chunk] = np.where(union > 0, inter / union, np.nan)
        return out

    def matrix(self, d_rows, s_rows=None):
        """Jaccard of DBLP rows d_rows against all (or s_rows) Scholar rows, as a dense matrix."""
        S = self.S if s_rows is None else self.S[s_rows]
        ss = self.s_size if s_rows is None else self.s_size[s_rows]
        inter = (self.D[d_rows] @ S.T).toarray()
        union = self.d_size[d_rows][:, None] + ss[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(union > 0, inter / union, np.nan).astype(np.float32)


def ngram_lists(strings, n):
    return [char_ngrams(s, n) for s in strings]


def year_score_vec(y1, y2):
    """Student 1's year rule on arrays of parsed years: 1 same, 0.5 one apart, 0 otherwise, NaN missing."""
    d = np.abs(np.asarray(y1, dtype=float) - np.asarray(y2, dtype=float))
    return np.where(np.isnan(d), np.nan, np.where(d == 0, 1.0, np.where(d == 1, 0.5, 0.0))).astype(np.float32)


def _best_match(A, B):
    """Student 1's author_similarity_best_match on already split name lists (rapidfuzz for speed)."""
    if not A or not B:
        return np.nan
    m = [[RFLevenshtein.normalized_similarity(a, b) for b in B] for a in A]
    fwd = sum(max(r) for r in m) / len(A)
    bwd = sum(max(m[i][j] for i in range(len(A))) for j in range(len(B))) / len(B)
    return (fwd + bwd) / 2


def _nonempty(seq):
    return np.fromiter((len(x) > 0 for x in seq), dtype=bool, count=len(seq))


# ---------------------------------------------------------------- feature engine
class FeatureBuilder:
    """Precomputes per-record data once; compute(di, si) then scores any list of pairs.

    dblp / scholar must already have the columns from add_normalized_columns().
    Pairs are given as row positions (di = DBLP row, si = Scholar row).
    """

    def __init__(self, dblp, scholar):
        self.D_T, self.S_T = dblp.t_norm.to_numpy(object), scholar.t_norm.to_numpy(object)
        self.D_Y, self.S_Y = dblp.yr.to_numpy(float), scholar.yr.to_numpy(float)
        self.D_V, self.S_V = dblp.v_norm.to_numpy(object), scholar.v_norm.to_numpy(object)
        self.D_NAMES, self.S_NAMES = dblp.a_names.to_list(), scholar.a_names.to_list()
        self._t_ok = (_nonempty(self.D_T), _nonempty(self.S_T))
        self._v_ok = (_nonempty(self.D_V), _nonempty(self.S_V))
        self._n_ok = (_nonempty(self.D_NAMES), _nonempty(self.S_NAMES))
        self._a_ok = (_nonempty(dblp.a_tokens.to_list()), _nonempty(scholar.a_tokens.to_list()))
        self.spaces = {
            "title_tok": SetSpace(dblp.t_tokens, scholar.t_tokens),
            "title_bi": SetSpace(ngram_lists(dblp.t_norm, 2), ngram_lists(scholar.t_norm, 2)),
            "title_tri": SetSpace(ngram_lists(dblp.t_norm, 3), ngram_lists(scholar.t_norm, 3)),
            "title_4g": SetSpace(ngram_lists(dblp.t_norm, 4), ngram_lists(scholar.t_norm, 4)),
            "author_tok": SetSpace(dblp.a_tokens, scholar.a_tokens),
            "author_last": SetSpace(dblp.a_last, scholar.a_last),
            "venue_tok": SetSpace(dblp.v_tokens, scholar.v_tokens),
            "venue_tri": SetSpace(ngram_lists(dblp.v_norm, 3), ngram_lists(scholar.v_norm, 3)),
        }

    def compute(self, di, si):
        di = np.asarray(di, dtype=np.int64); si = np.asarray(si, dtype=np.int64)
        sp = self.spaces
        both = lambda ok: ok[0][di] & ok[1][si]
        miss = lambda ok, v: np.where(both(ok), v, np.nan).astype(np.float32)
        dt, st = self.D_T[di], self.S_T[si]
        dv, sv = self.D_V[di], self.S_V[si]

        F = pd.DataFrame({"di": di, "si": si})
        F["title_exact"] = miss(self._t_ok, dt == st)
        F["title_edit"] = miss(self._t_ok, np.fromiter(
            (RFLevenshtein.normalized_similarity(a, b) for a, b in zip(dt, st)), dtype=np.float32, count=len(di)))
        F["title_jac"] = miss(self._t_ok, sp["title_tok"].pairs(di, si))
        F["title_bi"] = miss(self._t_ok, sp["title_bi"].pairs(di, si))
        F["title_tri"] = miss(self._t_ok, sp["title_tri"].pairs(di, si))
        F["title_4g"] = miss(self._t_ok, sp["title_4g"].pairs(di, si))
        F["author_tok"] = miss(self._a_ok, sp["author_tok"].pairs(di, si))
        F["author_last"] = miss(self._n_ok, sp["author_last"].pairs(di, si))
        F["author_best"] = np.fromiter((_best_match(self.D_NAMES[a], self.S_NAMES[b]) for a, b in zip(di, si)),
                                       dtype=np.float32, count=len(di))
        F["venue_exact"] = miss(self._v_ok, dv == sv)
        F["venue_tok"] = miss(self._v_ok, sp["venue_tok"].pairs(di, si))
        F["venue_tri"] = miss(self._v_ok, sp["venue_tri"].pairs(di, si))
        F["year"] = year_score_vec(self.D_Y[di], self.S_Y[si])
        ld = np.fromiter((len(x) for x in dt), dtype=float, count=len(di))
        ls = np.fromiter((len(x) for x in st), dtype=float, count=len(di))
        with np.errstate(invalid="ignore", divide="ignore"):
            F["len_ratio"] = miss(self._t_ok, np.minimum(ld, ls) / np.maximum(ld, ls))
        return F
