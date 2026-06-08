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
        "pointer_accuracy_heatmap_L_t.png",
        "pointer_accuracy_vs_t_by_L.png",
        "pointer_accuracy_vs_L_by_t.png",
        "pointer_path_accuracy_vs_L.png",
        "pointer_first_error_histogram.png",
    ]:
        assert expected in names, f"Missing plot: {expected}"
        out_path = out_dir / expected
        assert out_path.exists()
        assert out_path.stat().st_size > 0


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
