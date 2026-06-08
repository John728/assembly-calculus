from __future__ import annotations

import csv


def test_write_summary_preserves_pointer_time_dimension(tmp_path):
    from experiment_suite.aggregate import write_summary

    rows = [
        {
            "family": "AC",
            "model_name": "Pointer-AC-Theory",
            "list_type": "Unseen",
            "k_test": 2,
            "experiment": "pointer_chasing",
            "L": 2,
            "t": 1,
            "accuracy": 0.0,
        },
        {
            "family": "AC",
            "model_name": "Pointer-AC-Theory",
            "list_type": "Unseen",
            "k_test": 2,
            "experiment": "pointer_chasing",
            "L": 2,
            "t": 2,
            "accuracy": 1.0,
        },
    ]

    summary_path = write_summary(rows, tmp_path)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(summary_rows) == 2
    assert {(row["k_test"], row["t"], row["mean_accuracy"]) for row in summary_rows} == {
        ("2", "1", "0.0"),
        ("2", "2", "1.0"),
    }
