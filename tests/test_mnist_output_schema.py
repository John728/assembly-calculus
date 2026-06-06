from __future__ import annotations

import csv
import json
from pathlib import Path

from experiment_suite.aggregate import write_raw_results


def test_write_raw_results_json_serializes_mnist_vector_fields(tmp_path: Path) -> None:
    rows = [
        {
            "suite": "mnist-demo",
            "family": "MNIST_AC",
            "model_name": "local-ac",
            "list_type": "MNIST",
            "k_test": 1,
            "accuracy": 0.75,
            "overlaps": [0.1, 0.2, 0.3],
            "trajectory": [[1, 2], [3, 4]],
            "overlap_trajectory": [{"digit": 7, "overlap": 0.5}],
            "params": {"assembly_size": 16, "density": 0.2},
            "theta": {"min": 0.1, "max": 0.9},
        }
    ]

    path = write_raw_results(rows, tmp_path)

    with path.open(newline="", encoding="utf-8") as handle:
        written = next(csv.DictReader(handle))

    assert written["suite"] == "mnist-demo"
    assert written["k_test"] == "1"
    assert json.loads(written["overlaps"]) == [0.1, 0.2, 0.3]
    assert json.loads(written["trajectory"]) == [[1, 2], [3, 4]]
    assert json.loads(written["overlap_trajectory"]) == [{"digit": 7, "overlap": 0.5}]
    assert json.loads(written["params"]) == {"assembly_size": 16, "density": 0.2}
    assert json.loads(written["theta"]) == {"min": 0.1, "max": 0.9}
