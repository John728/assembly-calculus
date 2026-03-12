from __future__ import annotations

import sys
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
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import accuracy_vs_hop, build_pointer_network, generate_unique_lists, train_node_assemblies, train_seen_transitions

    if job.condition.list_type != "Seen":
        raise NotImplementedError("AC runner currently supports seen-list jobs only")

    model_values = job.model.values
    root_rng = make_rng(job.seed)
    list_rng, net_rng, eval_rng = spawn_rngs(root_rng, 3)

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
