from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from experiment_suite.jobs import ExperimentJob


ROOT = Path(__file__).resolve().parents[2]
PYAC_SRC = ROOT / "pyac" / "src"
if str(PYAC_SRC) not in sys.path:
    sys.path.insert(0, str(PYAC_SRC))

from pyac.tasks.mnist import (  # noqa: E402
    PixelAssemblyEncoder,
    RawPixelEncoder,
    build_mnist_network,
    evaluate_mnist_t_sweep,
    load_mnist_split,
    train_mnist_assemblies,
)


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


def _as_int_list(value: object, default: list[int]) -> list[int]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError("t_values must be a list of integers")
    return [_as_int(item, 0) for item in value]


def _limited(array: np.ndarray, limit: int) -> np.ndarray:
    if limit < 0:
        raise ValueError("MNIST limits must be >= 0")
    return array[:limit]


def run_mnist_ac_job(job: ExperimentJob) -> list[dict[str, object]]:
    model_values = job.model.values
    data_dir = model_values.get("data_dir")
    if data_dir is None:
        raise ValueError("MNIST_AC requires data_dir pointing to MNIST IDX files")

    train_split = load_mnist_split(str(data_dir), "train")
    test_split = load_mnist_split(str(data_dir), "test")
    train_limit = _as_int(model_values.get("train_limit"), len(train_split.images))
    test_limit = _as_int(model_values.get("test_limit"), len(test_split.images))
    train_images = _limited(train_split.images, train_limit)
    train_labels = _limited(train_split.labels, train_limit)
    test_images = _limited(test_split.images, test_limit)
    test_labels = _limited(test_split.labels, test_limit)

    rng = np.random.default_rng(job.seed)
    encoder_type = str(model_values.get("encoder_type", "pool"))
    if encoder_type == "raw":
        encoder = RawPixelEncoder(
            k=_as_int(model_values.get("raw_k"), 200),
            area_name="X",
        )
    else:
        encoder = PixelAssemblyEncoder(
            active_pixels=_as_int(model_values.get("active_pixels"), 64),
            neurons_per_pixel=_as_int(model_values.get("pool_size"), 8),
            rng=rng,
            area_name="X",
        )
    network, task = build_mnist_network(
        n=_as_int(model_values.get("n"), 1000),
        k=_as_int(model_values.get("k"), 100),
        p=_as_float(model_values.get("p"), 0.1),
        beta=_as_float(model_values.get("beta"), 0.1),
        rng=rng,
        encoder=encoder,
    )
    task.seed = job.seed

    presentation_rounds = _as_int(model_values.get("presentation_rounds"), 1)
    settle_steps = _as_int(model_values.get("settle_steps"), 1)
    class_organized = str(model_values.get("class_organized", "true")).lower() in ("true", "1", "yes")
    train_mnist_assemblies(
        network,
        task,
        train_images,
        train_labels,
        presentation_rounds=presentation_rounds,
        settle_steps=settle_steps,
        class_organized=class_organized,
    )

    t_values = _as_int_list(model_values.get("t_values"), [0])
    raw_rows = evaluate_mnist_t_sweep(
        network,
        task,
        test_images,
        test_labels,
        t_values=t_values,
        instance_ids=list(range(len(test_images))),
        stimulus_mode=str(model_values.get("stimulus_mode", "held")),
    )

    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        row = dict(raw_row)
        t_value = _as_int(row.get("t"), 0)
        row.update(
            {
                "suite": job.suite_name,
                "seed": job.seed,
                "family": "MNIST_AC",
                "model_name": str(model_values.get("model_name", job.model.model_name)),
                "list_type": job.condition.list_type,
                "N": job.condition.N,
                "num_train_lists": len(train_images),
                "num_test_lists": len(test_images),
                "k_train_min": job.condition.k_train_min,
                "k_train_max": job.condition.k_train_max,
                "k_test": t_value,
                "accuracy": 1.0 if bool(row.get("correct")) else 0.0,
                "internal_steps": t_value,
                "params": None,
                "runtime_ms": None,
                "train_limit": train_limit,
                "test_limit": test_limit,
                "presentation_rounds": presentation_rounds,
                "settle_steps": settle_steps,
            }
        )
        rows.append(row)
    return rows
