import pandas as pd

from src.matching.blocking_reference import candidate_pairs, run_blocking
from src.matching.features import add_normalized_columns


def _frames():
    d = pd.DataFrame({"id": ["d1", "d2"], "title": ["Efficient Query Processing", "Mining Association Rules"],
                      "authors": [None, None], "venue": [None, None], "year": [None, None]})
    s = pd.DataFrame({"id": ["s1", "s2", "s3"],
                      "title": ["efficient query processing", "MiningAssociation rules fast", "Unrelated Paper Title"],
                      "authors": [None] * 3, "venue": [None] * 3, "year": [None] * 3})
    return add_normalized_columns(d), add_normalized_columns(s)


def test_blocking_finds_true_pairs_and_glued_titles():
    d, s = _frames()
    pairs = candidate_pairs(d, s)
    assert ("d1", "s1") in pairs
    assert ("d2", "s2") in pairs            # glued words caught by the rare char 6-gram pass
    assert ("d1", "s3") not in pairs


def test_union_is_the_union_of_passes():
    d, s = _frames()
    union, per_pass, _ = run_blocking(d, s)
    all_keys = set().union(*[set(v) for v in per_pass.values()])
    assert set(union) == all_keys
