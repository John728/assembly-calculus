from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from experiment_suite.jobs import ExperimentJob
from experiment_suite.schema import standardize_ac_row


ROOT = Path(__file__).resolve().parents[2]
PYAC_SRC = ROOT / "pyac" / "src"
if str(PYAC_SRC) not in sys.path:
    sys.path.insert(0, str(PYAC_SRC))


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid integer hyperparameters")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Unsupported integer value type: {type(value).__name__}")


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid float hyperparameters")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"Unsupported float value type: {type(value).__name__}")


def run_ac_job_with_artifacts(job: ExperimentJob) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng_module = import_module("pyac.core.rng")
    pointer_module = import_module("pyac.tasks.pointer")

    make_rng = rng_module.make_rng
    spawn_rngs = rng_module.spawn_rngs
    accuracy_vs_hop = pointer_module.accuracy_vs_hop
    build_pointer_network = pointer_module.build_pointer_network
    build_unseen_pointer_network = pointer_module.build_unseen_pointer_network
    evaluate_unseen_rollout = pointer_module.evaluate_unseen_rollout
    generate_unique_lists = pointer_module.generate_unique_lists
    train_node_assemblies = pointer_module.train_node_assemblies
    train_seen_transitions = pointer_module.train_seen_transitions

    model_values = job.model.values
    root_rng = make_rng(job.seed)
    list_rng, net_rng, eval_rng = spawn_rngs(root_rng, 3)

    if job.condition.list_type == "Unseen":
        lists = generate_unique_lists(job.condition.num_test_lists, job.condition.N, list_rng)
        network, task = build_unseen_pointer_network(
            list_length=job.condition.N,
            assembly_size=_as_int(model_values.get("assembly_size"), 16),
            density=_as_float(model_values.get("density"), 0.35),
            plasticity=_as_float(model_values.get("plasticity"), 0.2),
            rng=net_rng,
        )
        rows: list[dict[str, object]] = []
        time_budgets_raw = model_values.get("time_budgets")
        if isinstance(time_budgets_raw, list) and time_budgets_raw:
            time_budgets = [_as_int(value, job.condition.k_test_max) for value in time_budgets_raw]
        else:
            time_budgets = [job.condition.k_test_max]
        for internal_steps in time_budgets:
            for hop in range(job.condition.k_test_min, job.condition.k_test_max + 1):
                accuracy = evaluate_unseen_rollout(
                    network,
                    task,
                    lists,
                    hops=hop,
                    internal_steps=internal_steps,
                    samples_per_list=_as_int(model_values.get("samples_per_list_eval"), 64),
                    rng=eval_rng,
                )
                raw_row = {
                    "List Type": "Unseen",
                    "Model": str(model_values.get("model_name", job.model.model_name)),
                    "N": job.condition.N,
                    "Num Lists": job.condition.num_test_lists,
                    "k": hop,
                    "Accuracy": accuracy,
                    "Internal Steps": internal_steps,
                    "Assembly Size": _as_int(model_values.get("assembly_size"), 16),
                    "Density": _as_float(model_values.get("density"), 0.35),
                    "Plasticity": _as_float(model_values.get("plasticity"), 0.2),
                    "Transition Rounds": None,
                    "Association Steps": None,
                }
                rows.append(
                    standardize_ac_row(
                        raw_row,
                        suite=job.suite_name,
                        seed=job.seed,
                        N=job.condition.N,
                        num_train_lists=job.condition.num_train_lists,
                        num_test_lists=job.condition.num_test_lists,
                        k_train_min=job.condition.k_train_min,
                        k_train_max=job.condition.k_train_max,
                    )
                )
        return rows, {"network": network, "task": task, "lists": lists}

    lists = generate_unique_lists(job.condition.num_train_lists, job.condition.N, list_rng)
    network, task = build_pointer_network(
        num_lists=job.condition.num_train_lists,
        list_length=job.condition.N,
        assembly_size=_as_int(model_values.get("assembly_size"), 16),
        density=_as_float(model_values.get("density"), 0.15),
        plasticity=_as_float(model_values.get("plasticity"), 0.25),
        rng=net_rng,
    )

    train_node_assemblies(
        network,
        task,
        presentation_rounds=_as_int(model_values.get("presentation_rounds"), 4),
        settle_steps=_as_int(model_values.get("settle_steps"), 2),
    )
    train_seen_transitions(
        network,
        task,
        lists,
        transition_rounds=_as_int(model_values.get("transition_rounds"), 12),
        association_steps=_as_int(model_values.get("association_steps"), 2),
        teacher_strength=_as_float(model_values.get("teacher_strength"), 12.0),
    )

    raw_rows = accuracy_vs_hop(
        network,
        task,
        lists,
        k_values=list(range(job.condition.k_test_min, job.condition.k_test_max + 1)),
        samples_per_list=_as_int(model_values.get("samples_per_list_eval"), 64),
        rng=eval_rng,
        model_name=str(model_values.get("model_name", job.model.model_name)),
        settle_steps=1,
    )

    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        enriched_row = dict(raw_row)
        enriched_row["Assembly Size"] = _as_int(model_values.get("assembly_size"), 16)
        enriched_row["Density"] = _as_float(model_values.get("density"), 0.15)
        enriched_row["Plasticity"] = _as_float(model_values.get("plasticity"), 0.25)
        enriched_row["Transition Rounds"] = _as_int(model_values.get("transition_rounds"), 12)
        enriched_row["Association Steps"] = _as_int(model_values.get("association_steps"), 2)
        rows.append(
            standardize_ac_row(
                enriched_row,
                suite=job.suite_name,
                seed=job.seed,
                N=job.condition.N,
                num_train_lists=job.condition.num_train_lists,
                num_test_lists=job.condition.num_test_lists,
                k_train_min=job.condition.k_train_min,
                k_train_max=job.condition.k_train_max,
            )
        )
    artifacts = {
        "network": network,
        "task": task,
        "lists": lists,
    }
    return rows, artifacts


def run_ac_job(job: ExperimentJob) -> list[dict[str, object]]:
    rows, _ = run_ac_job_with_artifacts(job)
    return rows
