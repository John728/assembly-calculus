from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    count, rows, cols = images.shape
    path.write_bytes(struct.pack(">IIII", 2051, count, rows, cols) + images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    path.write_bytes(struct.pack(">II", 2049, len(labels)) + labels.astype(np.uint8).tobytes())


def _create_synthetic_mnist(data_dir: Path, *, n_train: int = 24, n_test: int = 16) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    train_images = rng.integers(0, 256, size=(n_train, 28, 28), dtype=np.uint8)
    train_labels = rng.integers(0, 10, size=n_train, dtype=np.uint8)
    test_images = rng.integers(0, 256, size=(n_test, 28, 28), dtype=np.uint8)
    test_labels = rng.integers(0, 10, size=n_test, dtype=np.uint8)
    _write_idx_images(data_dir / "train-images-idx3-ubyte", train_images)
    _write_idx_labels(data_dir / "train-labels-idx1-ubyte", train_labels)
    _write_idx_images(data_dir / "t10k-images-idx3-ubyte", test_images)
    _write_idx_labels(data_dir / "t10k-labels-idx1-ubyte", test_labels)


def test_mnist_dev_yaml_parses_and_produces_jobs(tmp_path: Path) -> None:
    from experiment_suite.config import load_suite_config
    from experiment_suite.jobs import expand_jobs

    config_path = tmp_path / "mnist_ac_dev.yaml"
    config_path.write_text(
        "suite_name: mnist-ac-dev\n"
        "output_dir: " + str(tmp_path / "outputs") + "\n"
        "seeds: [1]\n"
        "conditions:\n"
        "  - list_type: MNIST\n"
        "    N: 10\n"
        "    num_train_lists: 60000\n"
        "    num_test_lists: 10000\n"
        "    k_train_min: 1\n"
        "    k_train_max: 1\n"
        "    k_test_min: 0\n"
        "    k_test_max: 2\n"
        "models:\n"
        "  MNIST_AC:\n"
        "    - model_name: MNIST-AC-Dev\n"
        "      data_dir: " + str(tmp_path / "mnist") + "\n"
        "      train_limit: 24\n"
        "      test_limit: 16\n"
        "      t_values: [0, 1, 2, 4, 8, 12, 16, 20]\n"
        "      n: 64\n"
        "      k: 8\n"
        "      p: 0.1\n"
        "      beta: 0.1\n"
        "      active_pixels: 16\n"
        "      pool_size: 4\n"
        "      presentation_rounds: 1\n"
        "      settle_steps: 1\n"
        "      stimulus_mode: held\n",
        encoding="utf-8",
    )

    config = load_suite_config(config_path)
    assert config.suite_name == "mnist-ac-dev"
    assert config.seeds == [1]
    assert len(config.conditions) == 1
    assert len(config.models) == 1
    assert "MNIST_AC" in config.models

    jobs = expand_jobs(config)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.family == "MNIST_AC"
    assert job.seed == 1
    assert job.model.model_name == "MNIST-AC-Dev"
    assert job.model.values["t_values"] == [0, 1, 2, 4, 8, 12, 16, 20]
    assert job.model.values["n"] == 64
    assert job.model.values["k"] == 8


def test_mnist_suite_fails_clearly_when_data_dir_is_missing(tmp_path: Path) -> None:
    import run_experiment_suite

    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        "suite_name: mnist-no-data\n"
        "output_dir: " + str(tmp_path / "outputs") + "\n"
        "seeds: [1]\n"
        "conditions:\n"
        "  - list_type: MNIST\n"
        "    N: 10\n"
        "    num_train_lists: 10\n"
        "    num_test_lists: 3\n"
        "    k_train_min: 1\n"
        "    k_train_max: 1\n"
        "    k_test_min: 0\n"
        "    k_test_max: 2\n"
        "models:\n"
        "  MNIST_AC:\n"
        "    - model_name: No-Data\n"
        "      data_dir: " + str(tmp_path / "nonexistent") + "\n"
        "      train_limit: 10\n"
        "      test_limit: 3\n"
        "      t_values: [0, 1]\n"
        "      n: 32\n"
        "      k: 4\n"
        "      p: 0.1\n"
        "      beta: 0.1\n"
        "      active_pixels: 8\n"
        "      pool_size: 2\n"
        "      presentation_rounds: 1\n"
        "      settle_steps: 1\n"
        "      stimulus_mode: held\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="MNIST"):
        run_experiment_suite.run_suite(config_path)


def test_mnist_full_pipeline_produces_outputs(tmp_path: Path, monkeypatch) -> None:
    _create_synthetic_mnist(tmp_path / "mnist", n_train=24, n_test=16)

    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        "suite_name: mnist-e2e\n"
        "output_dir: " + str(tmp_path / "outputs") + "\n"
        "seeds: [42]\n"
        "conditions:\n"
        "  - list_type: MNIST\n"
        "    N: 10\n"
        "    num_train_lists: 24\n"
        "    num_test_lists: 16\n"
        "    k_train_min: 1\n"
        "    k_train_max: 1\n"
        "    k_test_min: 0\n"
        "    k_test_max: 2\n"
        "models:\n"
        "  MNIST_AC:\n"
        "    - model_name: E2E-Test\n"
        "      data_dir: " + str(tmp_path / "mnist") + "\n"
        "      train_limit: 24\n"
        "      test_limit: 16\n"
        "      t_values: [0, 1, 2, 4, 8]\n"
        "      n: 100\n"
        "      k: 8\n"
        "      p: 0.1\n"
        "      beta: 0.1\n"
        "      active_pixels: 12\n"
        "      pool_size: 3\n"
        "      presentation_rounds: 1\n"
        "      settle_steps: 1\n"
        "      stimulus_mode: transient\n",
        encoding="utf-8",
    )

    import run_experiment_suite

    out_dir = run_experiment_suite.run_suite(config_path)

    # Raw results CSV
    raw_csv = out_dir / "raw_results.csv"
    assert raw_csv.exists()
    with raw_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
    assert len(raw_rows) > 0
    expected_row_count = 16 * 5  # n_test * len(t_values)
    assert len(raw_rows) == expected_row_count, f"Expected {expected_row_count} rows, got {len(raw_rows)}"

    required_fields = ["experiment", "t", "target", "prediction", "correct", "overlaps",
                       "correct_overlap", "strongest_wrong_overlap", "margin",
                       "trajectory", "overlap_trajectory", "plasticity_on",
                       "family", "suite", "seed", "model_name"]
    for field in required_fields:
        assert field in raw_rows[0], f"Missing field {field}"

    for row in raw_rows:
        assert row["experiment"] == "mnist"
        assert row["plasticity_on"] == "False"
        assert row["family"] == "MNIST_AC"
        assert row["model_name"] == "E2E-Test"
        t_val = int(row["t"])
        assert t_val in [0, 1, 2, 4, 8]
        target = int(row["target"])
        assert 0 <= target <= 9
        prediction = int(row["prediction"])
        assert 0 <= prediction <= 9
        assert row["correct"] in ("True", "False")

        overlaps = json.loads(row["overlaps"])
        assert isinstance(overlaps, list) and len(overlaps) == 10

        trajectory = json.loads(row["trajectory"])
        assert isinstance(trajectory, list) and len(trajectory) == t_val + 1

        overlap_trajectory = json.loads(row["overlap_trajectory"])
        assert isinstance(overlap_trajectory, list) and len(overlap_trajectory) == t_val + 1

        margin = float(row["margin"])
        assert isinstance(margin, float)

    # Summary CSV
    summary_csv = out_dir / "summary.csv"
    assert summary_csv.exists()
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        summary_rows = list(reader)
    assert len(summary_rows) > 0
    families = {row["family"] for row in summary_rows}
    assert families == {"MNIST_AC"}
    for row in summary_rows:
        assert row["list_type"] == "MNIST"
        assert float(row["mean_accuracy"]) >= 0.0

    # Config snapshot
    assert (out_dir / "config_snapshot.yaml").exists()

    # Plots (Theory Map Section 12 minimum figures)
    plots_dir = out_dir / "plots"
    assert plots_dir.is_dir()
    required_plots = [
            "mnist_accuracy_vs_t.png",
            "mnist_per_class_accuracy_vs_t.png",
            "mnist_margin_vs_t.png",
            "mnist_confusion_early.png",
            "mnist_confusion_best.png",
            "mnist_confusion_late.png",
        ]
    for plot_name in required_plots:
        assert (plots_dir / plot_name).exists(), f"Missing plot: {plot_name}"
