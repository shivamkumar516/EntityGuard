import numpy as np
from src.evaluation import (
    prf,
    blocking_metrics,
    threshold_sweep,
    best_threshold,
    fast_best_f1,
)


def test_evaluation_module():
    _g = {(1, 10), (2, 20), (3, 30), (4, 40)}
    _p = {(1, 10), (2, 20), (5, 50)}
    _m = prf(_p, _g)
    assert _m["TP"] == 2 and _m["FP"] == 1 and _m["FN"] == 2
    assert abs(_m["precision"] - 2 / 3) < 1e-9 and _m["recall"] == 0.5

    _b = blocking_metrics({(1, 10), (2, 20), (9, 90)}, _g, total_pairs=100)
    assert _b["blocking_recall"] == 0.5 and abs(_b["reduction_ratio"] - 0.97) < 1e-9

    _sw = threshold_sweep([0.9, 0.8, 0.3], [1, 0, 1], 3, [0.5])
    assert _sw.TP[0] == 1 and _sw.FP[0] == 1 and _sw.FN[0] == 2
    assert (
        abs(
            fast_best_f1(np.array([0.9, 0.8, 0.3]), np.array([1, 0, 1]), 3)
            - threshold_sweep([0.9, 0.8, 0.3], [1, 0, 1], 3, [0.9, 0.8, 0.3]).f1.max()
        )
        < 1e-9
    )


if __name__ == "__main__":
    test_evaluation_module()
    print("✅ evaluation module tests passed")