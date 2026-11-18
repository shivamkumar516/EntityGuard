import pandas as pd

from src.blocking import candidate_pairs, generate_candidate_details, run_blocking
from src.matching.blocking_reference import candidate_pairs as legacy_candidate_pairs
from src import pipeline as P


def _frames():
    dblp = pd.DataFrame({
        "id": ["d1", "d2", "d3"],
        "title": ["Efficient Query Processing", "Mining Association Rules", "A Short Title"],
        "authors": [None, None, None],
        "venue": [None, None, None],
        "year": [None, 2001, 2005],
    })
    scholar = pd.DataFrame({
        "id": ["s1", "s2", "s3"],
        "title": ["efficient query processing", "MiningAssociation rules fast", "Unrelated Publication"],
        "authors": [None, None, None],
        "venue": [None, None, None],
        "year": [None, 2001, 2005],
    })
    return dblp, scholar


def test_candidate_pairs_returns_union_and_finds_missing_year_match():
    dblp, scholar = _frames()
    pairs = candidate_pairs(dblp, scholar)
    assert ("d1", "s1") in pairs
    assert ("d2", "s2") in pairs
    assert ("d1", "s3") not in pairs


def test_details_preserve_all_rules_and_pairs_are_deduplicated():
    dblp, scholar = _frames()
    details = generate_candidate_details(dblp, scholar)
    pairs, per_pass, _timings, stats = run_blocking(dblp, scholar)

    assert set(details) == pairs
    assert pairs == set().union(*per_pass.values())
    assert all(isinstance(rules, set) and rules for rules in details.values())
    assert stats["candidate_pairs"] == len(pairs)
    assert stats["duplicate_rule_hits"] == sum(map(len, per_pass.values())) - len(pairs)


def test_legacy_application_interface_uses_same_candidate_generator():
    dblp, scholar = _frames()
    assert candidate_pairs(dblp, scholar) == legacy_candidate_pairs(dblp, scholar)



def test_legacy_adapter_preserves_original_scholar_row_positions():
    import numpy as np
    from src.matching.blocking_reference import run_blocking as legacy_run_blocking

    dblp, scholar = _frames()
    union, per_pass, _timings = legacy_run_blocking(dblp, scholar, s_rows=np.array([1, 2]))
    n_s = len(scholar)
    d_rows = union // n_s
    s_rows = union % n_s
    assert set(s_rows.tolist()).issubset({1, 2})
    assert all(len(values) >= 0 for values in per_pass.values())


def test_main_pipeline_uses_student2_blocking_module():
    """The application pipeline must use the canonical Student 2 implementation."""
    from src.blocking import BlockingIndex

    assert P.BlockingIndex is BlockingIndex
    dblp, scholar = _frames()
    index = P.BlockingIndex(dblp, scholar)
    keys, _per_pass, _timings = index.run_all()
    decoded = {
        (str(dblp.iloc[key // len(scholar)]["id"]),
         str(scholar.iloc[key % len(scholar)]["id"]))
        for key in keys
    }
    assert decoded == candidate_pairs(dblp, scholar)
