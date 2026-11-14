import numpy as np

from src.evaluation.split import split_labels, split_of_ids
from src.evaluation.student1_study import non_test_gold, run_study
from src.normalize import normalize_title


def test_split_sizes_and_reproducible():
    s = split_labels(2616)
    assert (s == "train").sum() == 1569 and (s == "val").sum() == 523 and (s == "test").sum() == 524
    assert (s == split_labels(2616)).all()


def test_split_matches_the_original_notebook_code():
    n, seed = 2616, 42
    perm = np.random.default_rng(seed).permutation(n)
    old = np.empty(n, dtype=object)
    old[perm[: int(0.6 * n)]] = "train"; old[perm[int(0.6 * n): int(0.8 * n)]] = "val"; old[perm[int(0.8 * n):]] = "test"
    assert (split_labels(n, seed) == old).all()


def _rec(title, authors, venue, year):
    return {"title": title, "authors": authors, "venue": venue, "year": year, "norm_title": normalize_title(title)}


def test_student1_study_runs_and_excludes_test_pairs():
    dblp = {f"d{i}": _rec(f"query processing method {i}", f"A Smith{i}, B Lee", "VLDB", 2000 + i) for i in range(10)}
    scholar = {f"s{i}": _rec(f"Query processing method {i}", f"A Smith{i}; B Lee", "vldb", 2000 + i) for i in range(10)}
    scholar.update({f"x{i}": _rec(f"query optimisation trick {i}", "C Wu", "SIGMOD", 1990) for i in range(10)})
    gold = [(f"d{i}", f"s{i}") for i in range(10)]

    kept = non_test_gold(dblp, gold)
    split = split_of_ids(dblp.keys())
    assert kept and all(split[d] != "test" for d, _ in kept)

    t = run_study(dblp, scholar, kept)
    assert t["sample"].gold_pairs[0] == len(kept)
    assert set(t) == {"sample", "title_measures", "alpha", "author_methods", "weight_trials", "threshold_curves"}
    exact = t["title_measures"].query("measure == 'exact' and nonmatches == 'same first word'").iloc[0]
    assert exact.recall == 1.0                       # identical titles after normalization
    assert len(t["threshold_curves"]) == 16          # 2 scores x 8 mandatory thresholds
