from __future__ import annotations

from typing import Any


CORE_RESULT_FIELDS = [
    "suite",
    "seed",
    "family",
    "model_name",
    "list_type",
    "N",
    "num_train_lists",
    "num_test_lists",
    "k_train_min",
    "k_train_max",
    "k_test",
    "accuracy",
    "internal_steps",
    "params",
    "runtime_ms",
]


def _base_row(
    *,
    suite: str,
    seed: int,
    family: str,
    model_name: str,
    list_type: str,
    N: int,
    num_train_lists: int,
    num_test_lists: int,
    k_train_min: int,
    k_train_max: int,
    k_test: int,
    accuracy: float,
    internal_steps: int | None,
    params: int | None,
    runtime_ms: float | None = None,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "seed": seed,
        "family": family,
        "model_name": model_name,
        "list_type": list_type,
        "N": N,
        "num_train_lists": num_train_lists,
        "num_test_lists": num_test_lists,
        "k_train_min": k_train_min,
        "k_train_max": k_train_max,
        "k_test": k_test,
        "accuracy": accuracy,
        "internal_steps": internal_steps,
        "params": params,
        "runtime_ms": runtime_ms,
    }


def standardize_mlp_row(
    row: dict[str, Any],
    *,
    suite: str,
    seed: int,
    N: int,
    num_train_lists: int,
    num_test_lists: int,
    k_train_min: int,
    k_train_max: int,
) -> dict[str, Any]:
    standardized = _base_row(
        suite=suite,
        seed=seed,
        family="MLP",
        model_name=str(row["Model"]),
        list_type=str(row["List Type"]),
        N=int(N),
        num_train_lists=int(num_train_lists),
        num_test_lists=int(num_test_lists),
        k_train_min=int(k_train_min),
        k_train_max=int(k_train_max),
        k_test=int(row["k"]),
        accuracy=float(row["Accuracy"]),
        internal_steps=None,
        params=int(row["Params"]) if "Params" in row and row["Params"] is not None else None,
        runtime_ms=float(row["Runtime ms"]) if "Runtime ms" in row and row["Runtime ms"] is not None else None,
    )
    standardized.update(
        {
            "layers": int(row["Layers"]) if "Layers" in row and row["Layers"] is not None else None,
            "hidden_dim": int(row["Dim"]) if "Dim" in row and row["Dim"] is not None else None,
            "lr": float(row["LR"]) if "LR" in row and row["LR"] is not None else None,
            "epochs": int(row["Epochs"]) if "Epochs" in row and row["Epochs"] is not None else None,
            "assembly_size": None,
            "density": None,
            "plasticity": None,
            "transition_rounds": None,
            "association_steps": None,
        }
    )
    return standardized


def standardize_ac_row(
    row: dict[str, Any],
    *,
    suite: str,
    seed: int,
    N: int | None = None,
    num_train_lists: int,
    num_test_lists: int,
    k_train_min: int,
    k_train_max: int,
) -> dict[str, Any]:
    if ("N" not in row or row["N"] is None) and N is None:
        raise ValueError("AC row requires N either in the row or as an explicit argument")

    standardized = _base_row(
        suite=suite,
        seed=seed,
        family="AC",
        model_name=str(row["Model"]),
        list_type=str(row["List Type"]),
        N=int(row["N"]) if "N" in row and row["N"] is not None else int(N),
        num_train_lists=int(num_train_lists),
        num_test_lists=int(num_test_lists),
        k_train_min=int(k_train_min),
        k_train_max=int(k_train_max),
        k_test=int(row["k"]),
        accuracy=float(row["Accuracy"]),
        internal_steps=int(row["Internal Steps"]) if "Internal Steps" in row and row["Internal Steps"] is not None else None,
        params=int(row["Params"]) if "Params" in row and row["Params"] is not None else None,
        runtime_ms=float(row["Runtime ms"]) if "Runtime ms" in row and row["Runtime ms"] is not None else None,
    )
    standardized.update(
        {
            "layers": None,
            "hidden_dim": None,
            "lr": None,
            "epochs": None,
            "assembly_size": int(row["Assembly Size"]) if "Assembly Size" in row and row["Assembly Size"] is not None else None,
            "density": float(row["Density"]) if "Density" in row and row["Density"] is not None else None,
            "plasticity": float(row["Plasticity"]) if "Plasticity" in row and row["Plasticity"] is not None else None,
            "transition_rounds": int(row["Transition Rounds"]) if "Transition Rounds" in row and row["Transition Rounds"] is not None else None,
            "association_steps": int(row["Association Steps"]) if "Association Steps" in row and row["Association Steps"] is not None else None,
        }
    )
    return standardized
