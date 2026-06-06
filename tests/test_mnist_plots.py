from __future__ import annotations

import csv
from pathlib import Path


def test_generate_mnist_ac_plots_writes_all_required_pngs_from_raw_rows(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_mnist_ac_plots

    raw_results = tmp_path / "mnist_raw_results.csv"
    rows = [
        {"t": 1, "target": 0, "prediction": 0, "correct": True, "margin": 0.8,
         "correct_overlap": 0.9, "strongest_wrong_overlap": 0.1, "overlaps": "[0.9, 0.1]"},
        {"t": 1, "target": 1, "prediction": 0, "correct": False, "margin": -0.2,
         "correct_overlap": 0.6, "strongest_wrong_overlap": 0.8, "overlaps": "[0.6, 0.4]"},
        {"t": 2, "target": 0, "prediction": 0, "correct": True, "margin": 1.1,
         "correct_overlap": 0.95, "strongest_wrong_overlap": 0.05, "overlaps": "[0.95, 0.05]"},
        {"t": 2, "target": 1, "prediction": 1, "correct": True, "margin": 0.4,
         "correct_overlap": 0.2, "strongest_wrong_overlap": 0.2, "overlaps": "[0.2, 0.8]"},
        {"t": 3, "target": 0, "prediction": 0, "correct": True, "margin": 0.5,
         "correct_overlap": 0.7, "strongest_wrong_overlap": 0.2, "overlaps": "[0.7, 0.2]"},
        {"t": 3, "target": 1, "prediction": 1, "correct": True, "margin": 0.3,
         "correct_overlap": 0.5, "strongest_wrong_overlap": 0.2, "overlaps": "[0.5, 0.5]"},
    ]
    with raw_results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "t", "target", "prediction", "correct", "margin",
            "correct_overlap", "strongest_wrong_overlap", "overlaps",
        ])
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "plots"
    paths = generate_mnist_ac_plots(raw_results, out_dir)

    expected_names = {
        "mnist_accuracy_vs_t.png",
        "mnist_per_class_accuracy_vs_t.png",
        "mnist_margin_vs_t.png",
        "mnist_confusion_early.png",
        "mnist_confusion_best.png",
        "mnist_confusion_late.png",
    }
    actual_names = {path.name for path in paths}
    assert expected_names.issubset(actual_names), f"Missing: {expected_names - actual_names}"
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)
