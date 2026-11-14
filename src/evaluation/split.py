"""Train / validation / test split by DBLP record (60 / 20 / 20, seed 42).

Every candidate pair inherits the split of its DBLP record, so all pairs of one DBLP paper
are in the same split and nothing leaks from test into tuning.
"""
import numpy as np

SPLITS = ("train", "val", "test")


def split_labels(n_dblp, seed=42, train=0.6, val=0.2):
    """Array of 'train' / 'val' / 'test', one per DBLP row (in file order)."""
    perm = np.random.default_rng(seed).permutation(n_dblp)
    labels = np.empty(n_dblp, dtype=object)
    a, b = int(train * n_dblp), int((train + val) * n_dblp)
    labels[perm[:a]] = "train"
    labels[perm[a:b]] = "val"
    labels[perm[b:]] = "test"
    return labels


def split_of_ids(dblp_ids, seed=42):
    """{dblp_id: split} for DBLP ids given in file order."""
    ids = list(dblp_ids)
    return dict(zip(ids, split_labels(len(ids), seed)))
