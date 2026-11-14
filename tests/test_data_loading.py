import pandas as pd

from src.data_loading import (
    baseline_error_samples,
    dataset_health,
    exact_title_baseline,
    find_data_file,
    load_datasets,
)


def _write_tiny_dataset(tmp_path, dblp_name="DBLP1.csv"):
    pd.DataFrame({"id": ["d1", "d2", "d3"],
                  "title": ["The Wäsch Method!", "Query Optimization", ""],
                  "authors": ["A. Wäsch", None, "C. Lee"],
                  "venue": ["VLDB", "SIGMOD", "ICDE"],
                  "year": [2001, 2002, 2003]}).to_csv(tmp_path / dblp_name, index=False, encoding="latin-1")
    pd.DataFrame({"id": ["s1", "s2", "s3"],
                  "title": ["Wäsch method", "query optimisation", "Another paper"],
                  "authors": ["A Wäsch", "B Smith", ""],
                  "venue": ["vldb", None, "icde"],
                  "year": [2001, None, None]}).to_csv(tmp_path / "Scholar.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"idDBLP": ["d1", "d2"], "idScholar": ["s1", "s2"]}).to_csv(
        tmp_path / "DBLP-Scholar_perfectMapping.csv", index=False)


def test_both_dblp_file_names_are_accepted(tmp_path):
    for name in ("DBLP.csv", "DBLP1.csv"):
        d = tmp_path / name.replace(".", "_")
        d.mkdir()
        _write_tiny_dataset(d, name)
        assert find_data_file(("DBLP.csv", "DBLP1.csv"), str(d)).endswith(name)


def test_encodings_and_cleaning(tmp_path):
    _write_tiny_dataset(tmp_path)
    dblp, scholar, gold = load_datasets(str(tmp_path))
    assert dblp.title[0] == "The Wäsch Method!"          # Latin-1 read correctly
    assert scholar.title[0] == "Wäsch method"            # UTF-8-sig read correctly
    assert list(scholar.columns)[0] == "id"              # BOM not glued to the header
    assert dblp.title[2] is None and scholar.authors[2] is None   # blanks -> None
    assert list(gold.columns) == ["dblp_id", "scholar_id"]
    h = dataset_health(dblp, scholar)
    assert h.loc["Scholar", "missing year"] == 2 and h.loc["DBLP", "missing title"] == 1


def test_exact_title_baseline_and_error_samples(tmp_path):
    _write_tiny_dataset(tmp_path)
    dblp, scholar, gold = load_datasets(str(tmp_path))
    pred = exact_title_baseline(dblp, scholar)
    assert pred == {("d1", "s1")}                        # "optimization" != "optimisation"
    gs = set(zip(gold.dblp_id, gold.scholar_id))
    err = baseline_error_samples(pred, gs, dblp, scholar, n=10)
    assert list(err.kind) == ["False negative"]
    assert err.dblp_title[0] == "Query Optimization"
