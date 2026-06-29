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


def test_write_summary_preserves_pointer_c_dimension(tmp_path):
    from experiment_suite.aggregate import write_summary

    rows = []
    for c_value, accuracy in [(1, 1.0), (2, 0.0)]:
        rows.append(
            {
                "suite": "pointer-ac-theory",
                "family": "AC",
                "model_name": "Pointer-AC-Theory",
                "list_type": "Unseen",
                "k_test": 2,
                "experiment": "pointer_chasing",
                "pointer_variant": "proper_unseen",
                "L": 2,
                "t": 2,
                "c": c_value,
                "accuracy": accuracy,
                "path_accuracy": accuracy,
                "first_error_index": None if accuracy == 1.0 else 1,
            }
        )

    summary_path = write_summary(rows, tmp_path)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(summary_rows) == 2
    assert {(row["L"], row["t"], row["c"], row["mean_accuracy"]) for row in summary_rows} == {
        ("2", "2", "1", "1.0"),
        ("2", "2", "2", "0.0"),
    }
    assert all(row["pointer_variant"] == "proper_unseen" for row in summary_rows)


def test_write_summary_preserves_mnist_retention_dimensions_and_metrics(tmp_path):
    from experiment_suite.aggregate import write_summary

    rows = [
        {
            "suite": "mnist-ac-retention-phase",
            "family": "MNIST_AC",
            "model_name": "MNIST-Retention-Norm-Beta-0.5-R10",
            "list_type": "MNIST",
            "k_test": 2,
            "experiment": "mnist_retention",
            "stimulus_mode": "cue_then_off",
            "t": 6,
            "s": 4,
            "ell": 2,
            "cue_duration_s": 4,
            "retention_ell": 2,
            "presentation_rounds": 10,
            "settle_steps": 1,
            "normalization_on": True,
            "beta_train": 0.5,
            "T_train": 10,
            "plasticity_train_on": True,
            "plasticity_eval_on": False,
            "accuracy": 1.0,
            "correct_score": 0.8,
            "strongest_wrong_score": 0.2,
            "margin": 0.6,
            "retention_time": 20,
            "stayed_correct": True,
            "became_correct_later": False,
            "retained_full_horizon": True,
        },
        {
            "suite": "mnist-ac-retention-phase",
            "family": "MNIST_AC",
            "model_name": "MNIST-Retention-Norm-Beta-0.5-R10",
            "list_type": "MNIST",
            "k_test": 2,
            "experiment": "mnist_retention",
            "stimulus_mode": "cue_then_off",
            "t": 6,
            "s": 4,
            "ell": 2,
            "cue_duration_s": 4,
            "retention_ell": 2,
            "presentation_rounds": 10,
            "settle_steps": 1,
            "normalization_on": True,
            "beta_train": 0.5,
            "T_train": 10,
            "plasticity_train_on": True,
            "plasticity_eval_on": False,
            "accuracy": 0.0,
            "correct_score": 0.3,
            "strongest_wrong_score": 0.7,
            "margin": -0.4,
            "retention_time": 0,
            "stayed_correct": False,
            "became_correct_later": False,
            "retained_full_horizon": False,
        },
    ]

    summary_path = write_summary(rows, tmp_path)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row["stimulus_mode"] == "cue_then_off"
    assert row["s"] == "4"
    assert row["ell"] == "2"
    assert row["presentation_rounds"] == "10"
    assert row["normalization_on"] == "True"
    assert row["beta_train"] == "0.5"
    assert row["mean_accuracy"] == "0.5"
    assert row["mean_correct_score"] == "0.55"
    assert row["mean_strongest_wrong_score"] == "0.44999999999999996"
    assert row["mean_margin"] == "0.09999999999999998"
    assert row["mean_retention_time"] == "10.0"
    assert row["stayed_correct_rate"] == "0.5"
    assert row["retained_full_horizon_rate"] == "0.5"
