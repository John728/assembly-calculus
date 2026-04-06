from __future__ import annotations

from experiment_suite.config import ExperimentCondition, ModelConfig
from experiment_suite.jobs import ExperimentJob


def _accuracy(row: dict[str, object]) -> float:
    value = row["accuracy"]
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric accuracy, got {type(value).__name__}")


def _tiny_mlp_job() -> ExperimentJob:
    return ExperimentJob(
        suite_name="demo",
        output_dir="outputs/demo",
        family="MLP",
        model=ModelConfig(
            family="MLP",
            values={
                "model_name": "Tiny-MLP",
                "layers": 2,
                "hidden_dim": 32,
                "epochs": 1,
                "batch_size": 16,
                "lr": 1e-3,
                "samples_per_list_train": 4,
                "samples_per_list_eval": 4,
                "patience": 2,
            },
        ),
        seed=1,
        condition=ExperimentCondition(
            list_type="Seen",
            N=8,
            num_train_lists=2,
            num_test_lists=1,
            k_train_min=1,
            k_train_max=2,
            k_test_min=1,
            k_test_max=3,
        ),
    )


def _tiny_ac_job() -> ExperimentJob:
    return ExperimentJob(
        suite_name="demo",
        output_dir="outputs/demo",
        family="AC",
        model=ModelConfig(
            family="AC",
            values={
                "model_name": "Tiny-AC",
                "assembly_size": 8,
                "density": 0.2,
                "plasticity": 0.25,
                "presentation_rounds": 2,
                "transition_rounds": 4,
                "association_steps": 2,
                "samples_per_list_eval": 4,
            },
        ),
        seed=1,
        condition=ExperimentCondition(
            list_type="Seen",
            N=8,
            num_train_lists=2,
            num_test_lists=0,
            k_train_min=1,
            k_train_max=2,
            k_test_min=1,
            k_test_max=3,
        ),
    )


def _tiny_unseen_ac_job() -> ExperimentJob:
    return ExperimentJob(
        suite_name="demo",
        output_dir="outputs/demo",
        family="AC",
        model=ModelConfig(
            family="AC",
            values={
                "model_name": "Tiny-Unseen-AC",
                "assembly_size": 10,
                "density": 0.5,
                "plasticity": 0.25,
                "samples_per_list_eval": 4,
                "time_budgets": [2, 4],
            },
        ),
        seed=1,
        condition=ExperimentCondition(
            list_type="Unseen",
            N=6,
            num_train_lists=4,
            num_test_lists=2,
            k_train_min=1,
            k_train_max=2,
            k_test_min=1,
            k_test_max=3,
        ),
    )


def _tiny_proper_unseen_ac_job() -> ExperimentJob:
    return ExperimentJob(
        suite_name="demo",
        output_dir="outputs/demo",
        family="AC",
        model=ModelConfig(
            family="AC",
            values={
                "model_name": "Tiny-Proper-Unseen-AC",
                "protocol_variant": "proper_unseen",
                "assembly_size": 8,
                "density": 0.35,
                "plasticity": 0.2,
                "train_episodes": 2,
                "samples_per_list_eval": 4,
                "t_equals_k": True,
            },
        ),
        seed=1,
        condition=ExperimentCondition(
            list_type="Unseen",
            N=6,
            num_train_lists=4,
            num_test_lists=2,
            k_train_min=1,
            k_train_max=3,
            k_test_min=1,
            k_test_max=3,
        ),
    )


def test_mlp_runner_returns_standardized_rows() -> None:
    from experiment_suite.runners.mlp_runner import run_mlp_job

    rows = run_mlp_job(_tiny_mlp_job())

    assert rows
    assert all(row["family"] == "MLP" for row in rows)
    assert all(0.0 <= _accuracy(row) <= 1.0 for row in rows)
    assert {row["list_type"] for row in rows} == {"Seen"}
    assert all(row["epochs"] <= 1 for row in rows)


def test_ac_runner_returns_standardized_rows() -> None:
    from experiment_suite.runners.ac_runner import run_ac_job

    rows = run_ac_job(_tiny_ac_job())

    assert rows
    assert all(row["family"] == "AC" for row in rows)
    assert all(0.0 <= _accuracy(row) <= 1.0 for row in rows)
    assert {row["list_type"] for row in rows} == {"Seen"}


def test_ac_runner_supports_unseen_jobs() -> None:
    from experiment_suite.runners.ac_runner import run_ac_job

    rows = run_ac_job(_tiny_unseen_ac_job())

    assert rows
    assert all(row["family"] == "AC" for row in rows)
    assert all(0.0 <= _accuracy(row) <= 1.0 for row in rows)
    assert {row["list_type"] for row in rows} == {"Unseen"}
    assert {row["internal_steps"] for row in rows} == {2, 4}


def test_ac_runner_supports_proper_unseen_jobs() -> None:
    from experiment_suite.runners.ac_runner import run_ac_job_with_artifacts

    rows, artifacts = run_ac_job_with_artifacts(_tiny_proper_unseen_ac_job())

    assert rows
    assert all(row["family"] == "AC" for row in rows)
    assert all(0.0 <= _accuracy(row) <= 1.0 for row in rows)
    assert {row["list_type"] for row in rows} == {"Unseen"}
    assert all(row["model_name"] == "Tiny-Proper-Unseen-AC" for row in rows)
    assert all(int(row["internal_steps"]) == int(row["k_test"]) for row in rows)
    assert artifacts["task"].__class__.__name__ == "ProperUnseenPointerTask"
    assert "training_history" in artifacts
    assert "mechanism_trace" in artifacts
