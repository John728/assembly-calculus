from __future__ import annotations

from experiment_suite.config import ExperimentCondition, ModelConfig
from experiment_suite.jobs import ExperimentJob


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


def test_mlp_runner_returns_standardized_rows() -> None:
    from experiment_suite.runners.mlp_runner import run_mlp_job

    rows = run_mlp_job(_tiny_mlp_job())

    assert rows
    assert all(row["family"] == "MLP" for row in rows)
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in rows)
    assert {row["list_type"] for row in rows} == {"Seen"}


def test_ac_runner_returns_standardized_rows() -> None:
    from experiment_suite.runners.ac_runner import run_ac_job

    rows = run_ac_job(_tiny_ac_job())

    assert rows
    assert all(row["family"] == "AC" for row in rows)
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in rows)
    assert {row["list_type"] for row in rows} == {"Seen"}
