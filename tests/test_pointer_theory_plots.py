from __future__ import annotations

from pathlib import Path

import matplotlib.axes
import numpy as np
import pandas as pd


def test_generate_pointer_ac_plots_from_theory_rows(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_pointer_ac_plots

    rng = np.random.default_rng(42)
    rows = []
    for L in [1, 2, 3]:
        for t in [0, 1, 2, 4, 6]:
            c = t // L if L > 0 else 0
            for instance_id in range(5):
                start = int(rng.integers(0, 6))
                path_acc = 1.0 if t >= L else 0.0
                first_err = None if path_acc == 1.0 else 1
                rows.append({
                    "suite": "psuite",
                    "seed": 42,
                    "family": "AC",
                    "model_name": "P-Model",
                    "list_type": "Unseen",
                    "experiment": "pointer_chasing",
                    "N": 6,
                    "L": L,
                    "t": t,
                    "c": c,
                    "instance_id": f"p-{instance_id}",
                    "start_node": start,
                    "target": (start + L) % 6,
                    "prediction": (start + L) % 6 if t >= L else start,
                    "correct": t >= L,
                    "accuracy": 1.0 if t >= L else 0.0,
                    "true_trajectory": [0] * (L + 1),
                    "trajectory": [0] * (L + 1),
                    "path_accuracy": path_acc,
                    "first_error_index": first_err,
                })

    raw_csv = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)
    out_dir = tmp_path / "plots"

    paths = generate_pointer_ac_plots(raw_csv, out_dir)

    names = {p.name for p in paths}
    for expected in [
        "pointer_accuracy_heatmap_L_t_c1.png",
        "pointer_accuracy_heatmap_L_t.png",
        "pointer_accuracy_vs_t_by_L_c1.png",
        "pointer_accuracy_vs_t_by_L.png",
        "pointer_accuracy_vs_L_by_t_c1.png",
        "pointer_accuracy_vs_L_by_t.png",
        "pointer_path_accuracy_vs_L_c1.png",
        "pointer_path_accuracy_vs_L.png",
        "pointer_first_error_histogram.png",
    ]:
        assert expected in names, f"Missing plot: {expected}"
        out_path = out_dir / expected
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    assert "pointer_shortcut_ablation.png" not in names


def test_pointer_path_accuracy_plot_preserves_time_dimension(tmp_path: Path, monkeypatch) -> None:
    from experiment_suite.plots import generate_pointer_ac_plots

    def fail_if_collapsed(*args, **kwargs):
        raise AssertionError("path accuracy plot must not collapse all t values with one errorbar")

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", fail_if_collapsed)

    rows = []
    for L in [1, 2]:
        for t in [0, 1, 2]:
            for instance_id in range(2):
                rows.append({
                    "suite": "psuite",
                    "seed": 42,
                    "family": "AC",
                    "model_name": "P-Model",
                    "list_type": "Seen",
                    "experiment": "pointer_chasing",
                    "N": 4,
                    "L": L,
                    "t": t,
                    "c": 1,
                    "instance_id": f"p-{L}-{t}-{instance_id}",
                    "start_node": 0,
                    "target": L,
                    "prediction": L if t >= L else t,
                    "correct": t >= L,
                    "accuracy": 1.0 if t >= L else 0.0,
                    "true_trajectory": [0] * (L + 1),
                    "trajectory": [0] * (L + 1),
                    "path_accuracy": 1.0 if t >= L else (t + 1) / (L + 1),
                    "first_error_index": None if t >= L else t + 1,
                })

    raw_csv = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)

    paths = generate_pointer_ac_plots(raw_csv, tmp_path / "plots")

    assert (tmp_path / "plots" / "pointer_path_accuracy_vs_L.png") in paths


def test_pointer_plots_preserve_c_dimension(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_pointer_ac_plots

    rows = []
    for c in [1, 2]:
        for L in [1, 2]:
            for t in [1, 2, 4]:
                rows.append({
                    "suite": "psuite",
                    "seed": 42,
                    "family": "AC",
                    "model_name": "P-Model",
                    "list_type": "Unseen",
                    "experiment": "pointer_chasing",
                    "pointer_variant": "proper_unseen",
                    "N": 6,
                    "L": L,
                    "t": t,
                    "c": c,
                    "instance_id": f"p-{c}-{L}-{t}",
                    "start_node": 0,
                    "target": L,
                    "prediction": L if t >= c * L else 0,
                    "correct": t >= c * L,
                    "accuracy": 1.0 if t >= c * L else 0.0,
                    "true_trajectory": [0] * (L + 1),
                    "trajectory": [0] * (L + 1),
                    "path_accuracy": 1.0 if t >= c * L else 0.5,
                    "first_error_index": None if t >= c * L else 1,
                })

    raw_csv = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)

    paths = generate_pointer_ac_plots(raw_csv, tmp_path / "plots")
    names = {p.name for p in paths}

    assert "pointer_accuracy_heatmap_L_t_c1.png" in names
    assert "pointer_accuracy_heatmap_L_t_c2.png" in names
    assert "pointer_accuracy_vs_c.png" in names
    assert "pointer_first_error_histogram_by_c.png" in names


def test_pointer_shortcut_plot_requires_explicit_shortcut_labels(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_pointer_ac_plots

    base_row = {
        "suite": "psuite",
        "seed": 42,
        "family": "AC",
        "model_name": "P-Model",
        "list_type": "Seen",
        "experiment": "pointer_chasing",
        "N": 8,
        "L": 4,
        "t": 2,
        "c": 1,
        "instance_id": "p-1",
        "start_node": 0,
        "target": 4,
        "prediction": 4,
        "correct": True,
        "accuracy": 1.0,
        "true_trajectory": [0, 1, 2, 3, 4],
        "trajectory": [0, 1, 2, 3, 4],
        "path_accuracy": 1.0,
        "first_error_index": None,
    }

    no_shortcut_csv = tmp_path / "raw_no_shortcut.csv"
    pd.DataFrame([base_row]).to_csv(no_shortcut_csv, index=False)
    no_shortcut_paths = generate_pointer_ac_plots(no_shortcut_csv, tmp_path / "plots-no-shortcut")
    assert "pointer_shortcut_ablation.png" not in {p.name for p in no_shortcut_paths}

    shortcut_row = dict(base_row)
    shortcut_row["shortcut_operator"] = "M^2"
    shortcut_csv = tmp_path / "raw_shortcut.csv"
    pd.DataFrame([shortcut_row]).to_csv(shortcut_csv, index=False)
    shortcut_paths = generate_pointer_ac_plots(shortcut_csv, tmp_path / "plots-shortcut")
    assert "pointer_shortcut_ablation.png" in {p.name for p in shortcut_paths}
