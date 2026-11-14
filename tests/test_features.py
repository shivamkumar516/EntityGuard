"""The fast feature engine must give exactly Student 1's numbers (src/similarity.py)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src import similarity as S1
from src.matching import weighted_score
from src.matching.features import FeatureBuilder, add_normalized_columns, venue_tokens

# tricky on purpose: accents, mojibake, glued words, stop-word-only titles, missing fields,
# junk years, semicolon vs comma author lists, abbreviated venues
DBLP = pd.DataFrame({
    "id": ["d1", "d2", "d3", "d4", "d5"],
    "title": ["An Efficient Method for Record-Linkage!", "Dalí: A High Performance Storage Manager",
              "The", "Query Optimization in Databases", "Editorial"],
    "authors": ["Smith, J.; Kumar, A.", "M Rusinkiewicz, W Klas, J Wäsch", None, "J. D. Ullman", ""],
    "venue": ["Proc. ICDE", "International Conference on Data Engineering", "VLDB", None, "SIGMOD Record"],
    "year": [2001, 2002, 1999, None, 2003],
})
SCHOLAR = pd.DataFrame({
    "id": ["s1", "s2", "s3", "s4", "s5", "s6"],
    "title": ["efficient method record linkage", "Dali: a high performance storage manager",
              "QueryOptimization in databases", "Floridaâ??s Initial Time", "editorial", "of the"],
    "authors": ["J Smith, A Kumar", "W Klas; M Rusinkiewicz", "JD Ullman", None, "", "X Y"],
    "venue": ["ICDE", "icde", None, "vldb", "SIGMOD Rec.", ""],
    "year": [2001, 2003, 11, None, 2003, 2730],
})


def _nan(x):
    return np.nan if x is None else x


@pytest.fixture(scope="module")
def setup():
    d = add_normalized_columns(DBLP.copy()); s = add_normalized_columns(SCHOLAR.copy())
    pairs = list(itertools.product(range(len(d)), range(len(s))))
    di, si = map(np.array, zip(*pairs))
    return FeatureBuilder(d, s).compute(di, si), pairs


REFERENCE = {
    "title_exact": lambda a, b: S1.title_exact(a.title, b.title),
    "title_edit": lambda a, b: S1.title_edit_similarity(a.title, b.title),
    "title_jac": lambda a, b: S1.title_token_jaccard(a.title, b.title),
    "title_bi": lambda a, b: S1.title_ngram_jaccard(a.title, b.title, 2),
    "title_tri": lambda a, b: S1.title_ngram_jaccard(a.title, b.title, 3),
    "title_4g": lambda a, b: S1.title_ngram_jaccard(a.title, b.title, 4),
    "author_tok": lambda a, b: S1.author_similarity_jaccard(a.authors, b.authors),
    "author_best": lambda a, b: S1.author_similarity_best_match(a.authors, b.authors),
    "venue_exact": lambda a, b: S1.venue_exact(a.venue, b.venue),
    "venue_tri": lambda a, b: S1.venue_similarity(a.venue, b.venue, 3),
    "year": lambda a, b: S1.year_similarity(a.year, b.year),
    "len_ratio": lambda a, b: S1.title_length_ratio(a.title, b.title),
}


@pytest.mark.parametrize("feature", list(REFERENCE))
def test_fast_feature_equals_student1(setup, feature):
    F, pairs = setup
    ref = [_nan(REFERENCE[feature](DBLP.iloc[i], SCHOLAR.iloc[j])) for i, j in pairs]
    assert np.allclose(F[feature].to_numpy(float), np.array(ref, dtype=float), atol=1e-6, equal_nan=True)


def test_weighted_score_equals_student1_record_similarity(setup):
    """Whole pipeline check: our fast score == Student 1's record_similarity with the same settings."""
    F, pairs = setup
    alpha, w = 0.6, (0.45, 0.30, 0.05, 0.20)
    F = F.assign(title_combined=alpha * F.title_edit + (1 - alpha) * F.title_jac)
    fast = weighted_score(F, w, "author_best", "venue_tri")
    weights = dict(zip(["title", "author", "venue", "year"], w))
    for k, (i, j) in enumerate(pairs):
        a, b = DBLP.iloc[i], SCHOLAR.iloc[j]
        ref, parts = S1.record_similarity(a.title, a.authors, a.venue, a.year, b.title, b.authors, b.venue, b.year,
                                          author_method="best_match", alpha=alpha, weights=weights)
        if all(v is None for v in parts.values()):
            continue                       # Student 1 returns 0.0 here, we return NaN-safe 0 as well
        assert fast[k] == pytest.approx(ref, abs=1e-5), (i, j, parts)


def test_venue_token_extension():
    assert venue_tokens("international conference on data engineering") == ["data", "engineering", "icde"]
    assert venue_tokens("proc icde") == ["icde"]
