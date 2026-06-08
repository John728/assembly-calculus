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
    assert {row["internal_steps"] for row in rows} == {1, 2, 3}


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


def _tiny_theory_pointer_job() -> ExperimentJob:
    return ExperimentJob(
        suite_name="demo",
        output_dir="outputs/demo",
        family="AC",
        model=ModelConfig(
            family="AC",
            values={
                "model_name": "Tiny-Theory-Pointer",
                "assembly_size": 8,
                "density": 0.5,
                "plasticity": 0.25,
                "train_episodes": 1,
                "samples_per_list_eval": 2,
                "time_budgets": [2, 4],
                "theory_pointer": True,
            },
        ),
        seed=1,
        condition=ExperimentCondition(
            list_type="Unseen",
            N=4,
            num_train_lists=2,
            num_test_lists=1,
            k_train_min=1,
            k_train_max=2,
            k_test_min=1,
            k_test_max=2,
        ),
    )


def _tiny_seen_theory_pointer_job() -> ExperimentJob:
    return ExperimentJob(
        suite_name="demo",
        output_dir="outputs/demo",
        family="AC",
        model=ModelConfig(
            family="AC",
            values={
                "model_name": "Tiny-Seen-Theory-Pointer",
                "assembly_size": 8,
                "density": 0.5,
                "plasticity": 0.25,
                "presentation_rounds": 1,
                "settle_steps": 1,
                "transition_rounds": 1,
                "association_steps": 1,
                "samples_per_list_eval": 2,
                "time_budgets": [1, 2],
                "theory_pointer": True,
            },
        ),
        seed=1,
        condition=ExperimentCondition(
            list_type="Seen",
            N=4,
            num_train_lists=2,
            num_test_lists=0,
            k_train_min=1,
            k_train_max=2,
            k_test_min=1,
            k_test_max=2,
        ),
    )


def test_runner_theory_pointer_dispatches_per_instance_evaluation() -> None:
    from experiment_suite.runners.ac_runner import run_ac_job_with_artifacts

    job = _tiny_theory_pointer_job()

    call_args: list[dict[str, object]] = []

    def fake_evaluate(*args, hops, time_budget, samples_per_list, rng, **kwargs):
        call_args.append(
            dict(hops=hops, time_budget=time_budget,
                 samples_per_list=samples_per_list))
        return [
            {
                "list_idx": 0,
                "start_node": i,
                "target": i,
                "prediction": i,
                "correct": True,
                "true_trajectory": [i],
                "trajectory": [i],
                "path_accuracy": 1.0,
                "first_error_index": None,
                "experiment": "pointer_chasing",
                "N": 4,
                "L": hops,
                "t": time_budget,
                "c": 1,
                "instance_id": f"ptr-{i}",
                "plasticity_on": False,
            }
            for i in range(2)
        ]

    from unittest import mock
    with mock.patch(
            "pyac.tasks.pointer.build_proper_unseen_pointer_network",
        ) as fake_build, mock.patch(
            "pyac.tasks.pointer.train_proper_unseen_controller",
        ) as fake_train, mock.patch(
            "pyac.tasks.pointer.evaluate_proper_unseen_per_instance",
            side_effect=fake_evaluate,
        ) as fake_eval, mock.patch(
            "pyac.tasks.pointer.rollout_proper_unseen_pointer",
            return_value={"final_prediction": 0, "current_state_nodes": [0]},
        ):
            fake_network = mock.Mock()
            fake_task = mock.Mock()
            fake_task.list_length = 4
            fake_build.return_value = (fake_network, fake_task)
            fake_train.return_value = []

            rows, _artifacts = run_ac_job_with_artifacts(job)

    calls = list(call_args)
    assert len(calls) == 4  # (hops=1, t=2), (1,4), (2,2), (2,4)
    hop_t_pairs = {(c["hops"], c["time_budget"]) for c in calls}
    assert hop_t_pairs == {(1, 2), (1, 4), (2, 2), (2, 4)}
    assert all(c["samples_per_list"] == 2 for c in calls)

    assert len(rows) == 8  # 4 combinations * 2 samples
    assert all(row["model_name"] == "Tiny-Theory-Pointer" for row in rows)
    assert all(row["family"] == "AC" for row in rows)
    assert all(row["list_type"] == "Unseen" for row in rows)
    assert all(row["experiment"] == "pointer_chasing" for row in rows)
    for row in rows:
        assert "L" in row
        assert "t" in row
        assert "c" in row
        assert "path_accuracy" in row
        assert "first_error_index" in row
        assert "true_trajectory" in row
        assert "trajectory" in row


def test_runner_seen_theory_pointer_dispatches_per_instance_evaluation() -> None:
    from experiment_suite.runners.ac_runner import run_ac_job_with_artifacts

    job = _tiny_seen_theory_pointer_job()
    call_args: list[dict[str, object]] = []

    def fake_evaluate(*args, hops, time_budget, samples_per_list, rng, **kwargs):
        call_args.append(
            dict(hops=hops, time_budget=time_budget,
                 samples_per_list=samples_per_list)
        )
        return [
            {
                "list_idx": 0,
                "start_node": i,
                "target": i,
                "prediction": i,
                "correct": True,
                "true_trajectory": [i],
                "trajectory": [i],
                "path_accuracy": 1.0,
                "first_error_index": None,
                "experiment": "pointer_chasing",
                "pointer_variant": "seen",
                "N": 4,
                "L": hops,
                "t": time_budget,
                "c": 1,
                "completed_hops": min(hops, time_budget),
                "readout_step": min(hops, time_budget),
                "instance_id": f"seen-{i}",
                "plasticity_on": False,
            }
            for i in range(2)
        ]

    from unittest import mock
    with mock.patch(
        "pyac.tasks.pointer.build_pointer_network",
    ) as fake_build, mock.patch(
        "pyac.tasks.pointer.train_node_assemblies",
    ), mock.patch(
        "pyac.tasks.pointer.train_seen_transitions",
    ), mock.patch(
        "pyac.tasks.pointer.evaluate_seen_per_instance",
        side_effect=fake_evaluate,
    ):
        fake_build.return_value = (object(), object())
        rows, _artifacts = run_ac_job_with_artifacts(job)

    assert [(c["hops"], c["time_budget"]) for c in call_args] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    assert len(rows) == 8
    assert {row["family"] for row in rows} == {"AC"}
    assert {row["model_name"] for row in rows} == {"Tiny-Seen-Theory-Pointer"}
    assert {row["list_type"] for row in rows} == {"Seen"}
    assert {row["pointer_variant"] for row in rows} == {"seen"}
    assert {row["k_test"] for row in rows} == {1, 2}
    assert {row["internal_steps"] for row in rows} == {1, 2}
