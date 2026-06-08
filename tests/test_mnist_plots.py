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


def test_generate_mnist_ac_plots_writes_sequence_pngs(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_mnist_ac_plots

    raw_results = tmp_path / "mnist_sequence_raw_results.csv"
    rows = [
        {
            "experiment": "mnist_sequence",
            "t": step,
            "sequence_step": step,
            "phase_digit": phase_digit,
            "step_in_phase": step % 3,
            "target": phase_digit,
            "prediction": prediction,
            "correct": prediction == phase_digit,
            "margin": margin,
            "correct_overlap": 0.7 if prediction == phase_digit else 0.2,
            "strongest_wrong_overlap": 0.2 if prediction == phase_digit else 0.7,
            "overlaps": overlaps,
        }
        for step, phase_digit, prediction, margin, overlaps in [
            (0, 0, 0, 0.5, "[0.7,0.2,0.1]"),
            (1, 0, 0, 0.6, "[0.8,0.1,0.1]"),
            (2, 0, 0, 0.6, "[0.8,0.1,0.1]"),
            (3, 1, 0, -0.3, "[0.6,0.3,0.1]"),
            (4, 1, 1, 0.4, "[0.2,0.7,0.1]"),
            (5, 1, 1, 0.5, "[0.1,0.8,0.1]"),
        ]
    ]
    with raw_results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "plots"
    paths = generate_mnist_ac_plots(raw_results, out_dir)

    expected_names = {
        "mnist_sequence_predictions_0_to_9.png",
        "mnist_sequence_overlaps_0_to_9.png",
        "mnist_sequence_margin_0_to_9.png",
    }
    actual_names = {path.name for path in paths}
    assert expected_names.issubset(actual_names), f"Missing: {expected_names - actual_names}"
    assert "mnist_accuracy_vs_t.png" not in actual_names
    assert "mnist_confusion_best.png" not in actual_names
    assert all((out_dir / name).stat().st_size > 0 for name in expected_names)


def test_generate_mnist_ac_plots_filters_sequence_rows_from_mixed_results(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_mnist_ac_plots

    raw_results = tmp_path / "mnist_mixed_raw_results.csv"
    rows = [
        {
            "experiment": "mnist",
            "t": 0,
            "sequence_step": "",
            "phase_digit": "",
            "step_in_phase": "",
            "target": 0,
            "prediction": 0,
            "correct": True,
            "margin": 0.8,
            "correct_overlap": 0.9,
            "strongest_wrong_overlap": 0.1,
            "overlaps": "[0.9,0.1]",
        },
        {
            "experiment": "mnist",
            "t": 1,
            "sequence_step": "",
            "phase_digit": "",
            "step_in_phase": "",
            "target": 0,
            "prediction": 0,
            "correct": True,
            "margin": 0.9,
            "correct_overlap": 0.95,
            "strongest_wrong_overlap": 0.05,
            "overlaps": "[0.95,0.05]",
        },
        {
            "experiment": "mnist_sequence",
            "t": 0,
            "sequence_step": 0,
            "phase_digit": 0,
            "step_in_phase": 0,
            "target": 0,
            "prediction": 0,
            "correct": True,
            "margin": 0.5,
            "correct_overlap": 0.7,
            "strongest_wrong_overlap": 0.2,
            "overlaps": "[0.7,0.2,0.1]",
        },
        {
            "experiment": "mnist_sequence",
            "t": 1,
            "sequence_step": 1,
            "phase_digit": 1,
            "step_in_phase": 0,
            "target": 1,
            "prediction": 0,
            "correct": False,
            "margin": -0.3,
            "correct_overlap": 0.2,
            "strongest_wrong_overlap": 0.7,
            "overlaps": "[0.6,0.3,0.1]",
        },
    ]
    with raw_results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "plots"
    paths = generate_mnist_ac_plots(raw_results, out_dir)

    actual_names = {path.name for path in paths}
    assert "mnist_sequence_predictions_0_to_9.png" in actual_names
    assert "mnist_accuracy_vs_t.png" in actual_names
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_generate_mnist_ac_plots_writes_hold_sweep_pngs(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_mnist_ac_plots

    raw_results = tmp_path / "mnist_sequence_hold_sweep_raw_results.csv"
    rows = []
    for hold_steps in [2, 4]:
        for phase_index, digit in enumerate([0, 1]):
            for step_in_phase in range(hold_steps):
                sequence_step = phase_index * hold_steps + step_in_phase
                prediction = digit if step_in_phase >= hold_steps // 2 else 0
                rows.append(
                    {
                        "experiment": "mnist_sequence",
                        "model_name": f"Tiny-Hold-{hold_steps}",
                        "hold_steps": hold_steps,
                        "steps_per_digit": hold_steps,
                        "t": sequence_step,
                        "sequence_step": sequence_step,
                        "phase_index": phase_index,
                        "phase_digit": digit,
                        "step_in_phase": step_in_phase,
                        "target": digit,
                        "prediction": prediction,
                        "correct": prediction == digit,
                        "margin": 0.4 if prediction == digit else -0.3,
                        "correct_overlap": 0.7 if prediction == digit else 0.2,
                        "strongest_wrong_overlap": 0.2 if prediction == digit else 0.7,
                        "overlaps": "[0.7,0.2,0.1]" if prediction == 0 else "[0.2,0.7,0.1]",
                    }
                )

    with raw_results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "plots"
    paths = generate_mnist_ac_plots(raw_results, out_dir)

    expected_names = {
        "mnist_sequence_hold_sweep_final_accuracy.png",
        "mnist_sequence_hold_sweep_switch_latency.png",
        "mnist_sequence_hold_sweep_predictions.png",
    }
    actual_names = {path.name for path in paths}
    assert expected_names.issubset(actual_names), f"Missing: {expected_names - actual_names}"
    assert "mnist_accuracy_vs_t.png" not in actual_names
    assert all((out_dir / name).stat().st_size > 0 for name in expected_names)


def test_hold_sweep_plots_aggregate_runs_and_align_prediction_progress(tmp_path: Path, monkeypatch) -> None:
    import experiment_suite.plots as plots

    captured_bar_data = []
    captured_latency_data = []
    captured_prediction_x = []
    original_barplot = plots.sns.barplot
    original_lineplot = plots.sns.lineplot

    def capture_barplot(*args, **kwargs):
        captured_bar_data.append(kwargs["data"].copy())
        return original_barplot(*args, **kwargs)

    def capture_lineplot(*args, **kwargs):
        if kwargs.get("y") == "latency":
            captured_latency_data.append(kwargs["data"].copy())
        if kwargs.get("y") == "prediction":
            captured_prediction_x.append(kwargs.get("x"))
        return original_lineplot(*args, **kwargs)

    monkeypatch.setattr(plots.sns, "barplot", capture_barplot)
    monkeypatch.setattr(plots.sns, "lineplot", capture_lineplot)

    raw_results = tmp_path / "mnist_sequence_hold_sweep_multi_seed.csv"
    rows = []
    for hold_steps in [2, 4]:
        for seed, final_correct in [(1, False), (2, True)]:
            for digit in [0, 1]:
                for step_in_phase in range(hold_steps):
                    sequence_step = digit * hold_steps + step_in_phase
                    correct = final_correct and step_in_phase in {0, hold_steps - 1}
                    prediction = digit if correct else 9
                    rows.append(
                        {
                            "experiment": "mnist_sequence",
                            "model_name": f"Hold-{hold_steps}",
                            "seed": seed,
                            "hold_steps": hold_steps,
                            "steps_per_digit": hold_steps,
                            "t": sequence_step,
                            "sequence_step": sequence_step,
                            "phase_index": digit,
                            "phase_digit": digit,
                            "step_in_phase": step_in_phase,
                            "target": digit,
                            "prediction": prediction,
                            "correct": correct,
                            "margin": 0.4 if correct else -0.3,
                            "correct_overlap": 0.7 if correct else 0.2,
                            "strongest_wrong_overlap": 0.2 if correct else 0.7,
                            "overlaps": "[0.7,0.2,0.1]" if prediction == 0 else "[0.1,0.2,0.7]",
                        }
                    )

    with raw_results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plots.generate_mnist_ac_plots(raw_results, tmp_path / "plots")

    final_acc = captured_bar_data[0].set_index("hold_steps")["final_accuracy"].to_dict()
    assert final_acc == {2: 0.5, 4: 0.5}

    latency = captured_latency_data[0]
    mean_latency = latency.groupby("hold_steps")["latency"].mean().to_dict()
    assert mean_latency == {2: 1.0, 4: 2.0}
    assert captured_prediction_x == ["phase_progress"]
