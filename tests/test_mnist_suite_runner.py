from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from experiment_suite.config import ExperimentCondition, ModelConfig
from experiment_suite.jobs import ExperimentJob


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    count, rows, cols = images.shape
    path.write_bytes(struct.pack(">IIII", 2051, count, rows, cols) + images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    path.write_bytes(struct.pack(">II", 2049, len(labels)) + labels.astype(np.uint8).tobytes())


def _write_mnist_idx_files(data_dir: Path) -> None:
    data_dir.mkdir(exist_ok=True)
    train_labels = np.arange(10, dtype=np.uint8)
    test_labels = np.array([0, 1, 2], dtype=np.uint8)
    train_images = np.arange(10 * 28 * 28, dtype=np.uint8).reshape(10, 28, 28)
    test_images = np.arange(3 * 28 * 28, dtype=np.uint8).reshape(3, 28, 28)
    _write_idx_images(data_dir / "train-images-idx3-ubyte", train_images)
    _write_idx_labels(data_dir / "train-labels-idx1-ubyte", train_labels)
    _write_idx_images(data_dir / "t10k-images-idx3-ubyte", test_images)
    _write_idx_labels(data_dir / "t10k-labels-idx1-ubyte", test_labels)


def _mnist_job(data_dir: Path) -> ExperimentJob:
    return ExperimentJob(
        suite_name="mnist-demo",
        output_dir="outputs/mnist-demo",
        family="MNIST_AC",
        model=ModelConfig(
            family="MNIST_AC",
            values={
                "model_name": "Tiny-MNIST-AC",
                "data_dir": str(data_dir),
                "train_limit": 10,
                "test_limit": 3,
                "t_values": [0, 2],
                "n": 32,
                "k": 4,
                "p": 0.2,
                "beta": 0.1,
                "active_pixels": 8,
                "pool_size": 2,
                "presentation_rounds": 1,
                "settle_steps": 1,
                "stimulus_mode": "transient",
            },
        ),
        seed=7,
        condition=ExperimentCondition(list_type="MNIST", N=10, num_train_lists=10, num_test_lists=3),
    )


def _mnist_sequence_job(data_dir: Path) -> ExperimentJob:
    job = _mnist_job(data_dir)
    values = dict(job.model.values)
    values.update(
        {
            "model_name": "Tiny-MNIST-AC-Sequence",
            "sequence_digits": [0, 1, 2],
            "steps_per_digit": 3,
        }
    )
    return ExperimentJob(
        suite_name=job.suite_name,
        output_dir=job.output_dir,
        family=job.family,
        model=ModelConfig(family="MNIST_AC", values=values),
        seed=job.seed,
        condition=job.condition,
    )


def _mnist_sequence_hold_sweep_job(data_dir: Path) -> ExperimentJob:
    job = _mnist_sequence_job(data_dir)
    values = dict(job.model.values)
    values.pop("steps_per_digit")
    values["steps_per_digit_values"] = [2, 4]
    return ExperimentJob(
        suite_name=job.suite_name,
        output_dir=job.output_dir,
        family=job.family,
        model=ModelConfig(family="MNIST_AC", values=values),
        seed=job.seed,
        condition=job.condition,
    )


def test_mnist_runner_trains_once_then_evaluates_all_t_values(tmp_path: Path, monkeypatch) -> None:
    _write_mnist_idx_files(tmp_path)
    calls: list[tuple[str, object]] = []

    def fake_build_mnist_network(n, k, p, beta, rng, *, encoder=None):
        calls.append(("build", (n, k, p, beta, encoder.active_pixels, encoder.neurons_per_pixel)))
        return object(), type("Task", (), {})()

    def fake_train_mnist_assemblies(
        network, task, images, labels, presentation_rounds, settle_steps, class_organized
    ):
        calls.append(("train", (len(images), labels.tolist(), presentation_rounds, settle_steps, class_organized)))

    def fake_evaluate_mnist_t_sweep(network, task, images, labels, t_values, instance_ids=None, stimulus_mode="held"):
        calls.append(("eval", (len(images), labels.tolist(), list(t_values), list(instance_ids), stimulus_mode)))
        return [
            {
                "experiment": "mnist",
                "seed": None,
                "n": 32,
                "k": 4,
                "p": 0.2,
                "beta": 0.1,
                "t": t,
                "instance_id": instance_id,
                "target": int(label),
                "prediction": int(label),
                "correct": True,
                "plasticity_on": False,
                "stimulus_mode": stimulus_mode,
            }
            for image, label, instance_id in zip(images, labels, instance_ids)
            for t in t_values
        ]

    import experiment_suite.runners.mnist_ac_runner as runner

    monkeypatch.setattr(runner, "build_mnist_network", fake_build_mnist_network)
    monkeypatch.setattr(runner, "train_mnist_assemblies", fake_train_mnist_assemblies)
    monkeypatch.setattr(runner, "evaluate_mnist_t_sweep", fake_evaluate_mnist_t_sweep)

    rows = runner.run_mnist_ac_job(_mnist_job(tmp_path))

    assert [name for name, _ in calls] == ["build", "train", "eval"]
    assert calls[0][1] == (32, 4, 0.2, 0.1, 8, 2)
    assert calls[1][1] == (10, list(range(10)), 1, 1, True)
    assert calls[2][1] == (3, [0, 1, 2], [0, 2], [0, 1, 2], "transient")
    assert len(rows) == 6
    assert {row["family"] for row in rows} == {"MNIST_AC"}
    assert {row["model_name"] for row in rows} == {"Tiny-MNIST-AC"}
    assert {row["k_test"] for row in rows} == {0, 2}
    assert {row["accuracy"] for row in rows} == {1.0}
    assert all(row["plasticity_on"] is False for row in rows)


def test_mnist_runner_dispatches_sequence_probe(tmp_path: Path, monkeypatch) -> None:
    _write_mnist_idx_files(tmp_path)
    calls: list[tuple[str, object]] = []

    def fake_build_mnist_network(n, k, p, beta, rng, *, encoder=None):
        calls.append(("build", (n, k, p, beta)))
        return object(), type("Task", (), {})()

    def fake_train_mnist_assemblies(
        network, task, images, labels, presentation_rounds, settle_steps, class_organized
    ):
        calls.append(("train", (len(images), labels.tolist(), presentation_rounds, settle_steps, class_organized)))

    def fake_evaluate_mnist_sequence(
        network, task, images, labels, sequence_digits, steps_per_digit, instance_ids=None
    ):
        calls.append(
            (
                "sequence",
                (len(images), labels.tolist(), list(sequence_digits), steps_per_digit, list(instance_ids)),
            )
        )
        return [
            {
                "experiment": "mnist_sequence",
                "seed": None,
                "n": 32,
                "k": 4,
                "p": 0.2,
                "beta": 0.1,
                "t": step,
                "sequence_step": step,
                "phase_digit": digit,
                "target": digit,
                "prediction": digit,
                "correct": True,
                "plasticity_on": False,
                "stimulus_mode": "sequence_held",
            }
            for step, digit in enumerate([0, 0, 0, 1, 1, 1, 2, 2, 2])
        ]

    import experiment_suite.runners.mnist_ac_runner as runner

    monkeypatch.setattr(runner, "build_mnist_network", fake_build_mnist_network)
    monkeypatch.setattr(runner, "train_mnist_assemblies", fake_train_mnist_assemblies)
    monkeypatch.setattr(runner, "evaluate_mnist_sequence", fake_evaluate_mnist_sequence)

    rows = runner.run_mnist_ac_job(_mnist_sequence_job(tmp_path))

    assert [name for name, _ in calls] == ["build", "train", "sequence"]
    assert calls[2][1] == (3, [0, 1, 2], [0, 1, 2], 3, [0, 1, 2])
    assert len(rows) == 9
    assert {row["family"] for row in rows} == {"MNIST_AC"}
    assert {row["model_name"] for row in rows} == {"Tiny-MNIST-AC-Sequence"}
    assert {row["k_test"] for row in rows} == set(range(9))
    assert {row["internal_steps"] for row in rows} == set(range(9))
    assert {row["accuracy"] for row in rows} == {1.0}


def test_mnist_runner_dispatches_sequence_hold_sweep_after_one_training_run(tmp_path: Path, monkeypatch) -> None:
    _write_mnist_idx_files(tmp_path)
    calls: list[tuple[str, object]] = []

    def fake_build_mnist_network(n, k, p, beta, rng, *, encoder=None):
        calls.append(("build", (n, k, p, beta)))
        return object(), type("Task", (), {})()

    def fake_train_mnist_assemblies(
        network, task, images, labels, presentation_rounds, settle_steps, class_organized
    ):
        calls.append(("train", (len(images), labels.tolist(), presentation_rounds, settle_steps, class_organized)))

    def fake_evaluate_mnist_sequence(
        network, task, images, labels, sequence_digits, steps_per_digit, instance_ids=None
    ):
        calls.append(
            (
                "sequence",
                (len(images), labels.tolist(), list(sequence_digits), steps_per_digit, list(instance_ids)),
            )
        )
        return [
            {
                "experiment": "mnist_sequence",
                "seed": None,
                "n": 32,
                "k": 4,
                "p": 0.2,
                "beta": 0.1,
                "t": step,
                "sequence_step": step,
                "phase_digit": digit,
                "steps_per_digit": steps_per_digit,
                "target": digit,
                "prediction": digit,
                "correct": True,
                "plasticity_on": False,
                "stimulus_mode": "sequence_held",
            }
            for step, digit in enumerate([0] * steps_per_digit + [1] * steps_per_digit + [2] * steps_per_digit)
        ]

    import experiment_suite.runners.mnist_ac_runner as runner

    monkeypatch.setattr(runner, "build_mnist_network", fake_build_mnist_network)
    monkeypatch.setattr(runner, "train_mnist_assemblies", fake_train_mnist_assemblies)
    monkeypatch.setattr(runner, "evaluate_mnist_sequence", fake_evaluate_mnist_sequence)

    rows = runner.run_mnist_ac_job(_mnist_sequence_hold_sweep_job(tmp_path))

    assert [name for name, _ in calls] == ["build", "train", "sequence", "sequence"]
    assert calls[2][1] == (3, [0, 1, 2], [0, 1, 2], 2, [0, 1, 2])
    assert calls[3][1] == (3, [0, 1, 2], [0, 1, 2], 4, [0, 1, 2])
    assert len(rows) == 18
    assert {row["model_name"] for row in rows} == {
        "Tiny-MNIST-AC-Sequence-Hold-2",
        "Tiny-MNIST-AC-Sequence-Hold-4",
    }
    assert {row["hold_steps"] for row in rows} == {2, 4}
    assert {row["accuracy"] for row in rows} == {1.0}


def test_mnist_runner_fails_clearly_when_data_dir_is_missing(tmp_path: Path) -> None:
    from experiment_suite.runners.mnist_ac_runner import run_mnist_ac_job

    with pytest.raises(FileNotFoundError, match="MNIST"):
        run_mnist_ac_job(_mnist_job(tmp_path / "missing-mnist"))


def test_experiment_suite_dispatches_mnist_ac_family(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text(
        "suite_name: mnist-demo\n"
        "output_dir: " + str(tmp_path / "outputs") + "\n"
        "seeds: [7]\n"
        "conditions:\n"
        "  - list_type: MNIST\n"
        "    N: 10\n"
        "    num_train_lists: 10\n"
        "    num_test_lists: 3\n"
        "models:\n"
        "  MNIST_AC:\n"
        "    - model_name: Tiny-MNIST-AC\n"
        "      data_dir: " + str(tmp_path / "mnist") + "\n"
        "      t_values: [0, 1]\n",
        encoding="utf-8",
    )

    import run_experiment_suite

    plot_calls = []

    monkeypatch.setattr(
        "experiment_suite.runners.mnist_ac_runner.run_mnist_ac_job",
        lambda job: [
            {
                "suite": job.suite_name,
                "seed": job.seed,
                "family": "MNIST_AC",
                "model_name": job.model.model_name,
                "list_type": "MNIST",
                "N": 10,
                "num_train_lists": 10,
                "num_test_lists": 3,
                "k_train_min": 1,
                "k_train_max": 1,
                "k_test": 0,
                "accuracy": 1.0,
                "internal_steps": 0,
                "params": None,
                "runtime_ms": None,
            }
        ],
    )
    monkeypatch.setattr(
        "experiment_suite.plots.generate_mnist_ac_plots",
        lambda raw_results_csv, plots_dir: plot_calls.append((raw_results_csv, plots_dir)),
    )

    out_dir = run_experiment_suite.run_suite(cfg_path)

    assert (out_dir / "raw_results.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert plot_calls == [(out_dir / "raw_results.csv", out_dir / "plots")]
