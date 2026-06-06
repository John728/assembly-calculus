# MNIST-First PYAC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a real MNIST Assembly Calculus experiment that measures how internal update time `t` changes accuracy, overlap margins, saturation, and drift under fixed trained weights.

**Architecture:** Keep reusable AC mechanics in `pyac` and add MNIST as a task module, while `experiment_suite` only loads configs, dispatches runs, writes raw rows, and generates plots. The implementation must use real MNIST files, sparse assemblies, k-cap dynamics, Hebbian plasticity, frozen evaluation, and raw measured overlaps rather than synthetic or classifier-derived metrics.

**Tech Stack:** Python 3.11+, NumPy, SciPy sparse matrices, Matplotlib/Pandas for suite plots, Pytest. Real MNIST IDX files are the required data source for thesis runs.

---

## Guardrails

- Do not use sklearn digits, generated images, MLP logits, or synthetic labels as MNIST evidence.
- Do not silently fall back when real MNIST files are missing.
- Do not retrain per `t`; train once per seed/model instance, freeze weights, then sweep `t`.
- Do not leave plasticity enabled during evaluation.
- Do not claim MNIST measures execution depth. It only measures static-time settling, completion, saturation, or drift.
- Commit steps below are checkpoints only. Do not run `git commit` unless the user explicitly requests commits.

## Task 1: Real MNIST IDX Loader

**Files:**
- Create: `pyac/src/pyac/tasks/mnist/data.py`
- Create: `pyac/src/pyac/tasks/mnist/__init__.py`
- Test: `tests/test_mnist_data.py`

**Step 1: Write failing tests**

Add tests that create tiny valid IDX image/label files in `tmp_path` and verify loading.

```python
from __future__ import annotations

import gzip
import struct

import numpy as np
import pytest


def _write_idx_images(path, images: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, images.shape[0], images.shape[1], images.shape[2]))
        handle.write(images.astype(np.uint8).tobytes())


def _write_idx_labels(path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, labels.shape[0]))
        handle.write(labels.astype(np.uint8).tobytes())


def test_load_mnist_idx_reads_real_file_format(tmp_path):
    from pyac.tasks.mnist.data import load_mnist_split

    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    labels = np.array([3, 7], dtype=np.uint8)
    _write_idx_images(tmp_path / "train-images-idx3-ubyte.gz", images)
    _write_idx_labels(tmp_path / "train-labels-idx1-ubyte.gz", labels)

    split = load_mnist_split(tmp_path, split="train")

    assert split.images.shape == (2, 28, 28)
    assert split.images.dtype == np.float64
    assert split.labels.tolist() == [3, 7]
    assert split.images.max() <= 1.0


def test_load_mnist_split_fails_when_real_files_missing(tmp_path):
    from pyac.tasks.mnist.data import load_mnist_split

    with pytest.raises(FileNotFoundError, match="MNIST"):
        load_mnist_split(tmp_path, split="train")
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_data.py -q`

Expected: FAIL because `pyac.tasks.mnist.data` does not exist.

**Step 3: Implement minimal loader**

Implement:

```python
from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MnistSplit:
    images: np.ndarray
    labels: np.ndarray


_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _open_idx(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"MNIST file not found: {path}")


def _read_images(path: Path) -> np.ndarray:
    _require_file(path)
    with _open_idx(path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic in {path}: {magic}")
        raw = np.frombuffer(handle.read(), dtype=np.uint8)
    expected = count * rows * cols
    if raw.size != expected:
        raise ValueError(f"Invalid MNIST image payload in {path}: expected {expected}, got {raw.size}")
    return raw.reshape(count, rows, cols).astype(np.float64) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    _require_file(path)
    with _open_idx(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic in {path}: {magic}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Invalid MNIST label payload in {path}: expected {count}, got {labels.size}")
    return labels.astype(np.int64)


def load_mnist_split(data_dir: str | Path, *, split: str) -> MnistSplit:
    if split not in _FILES:
        raise ValueError(f"Unknown MNIST split: {split}")
    image_name, label_name = _FILES[split]
    root = Path(data_dir)
    images = _read_images(root / image_name)
    labels = _read_labels(root / label_name)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("MNIST images and labels have different lengths")
    return MnistSplit(images=images, labels=labels)
```

Export from `pyac/src/pyac/tasks/mnist/__init__.py`:

```python
from pyac.tasks.mnist.data import MnistSplit, load_mnist_split

__all__ = ["MnistSplit", "load_mnist_split"]
```

**Step 4: Run tests**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_data.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- pyac/src/pyac/tasks/mnist tests/test_mnist_data.py`

Expected: diff only contains MNIST loader and tests.

## Task 2: MNIST Image-To-Assembly Encoder

**Files:**
- Create: `pyac/src/pyac/tasks/mnist/encoding.py`
- Modify: `pyac/src/pyac/tasks/mnist/__init__.py`
- Test: `tests/test_mnist_encoding.py`

**Step 1: Write failing tests**

```python
from __future__ import annotations

import numpy as np


def test_pixel_encoder_is_deterministic_and_sparse():
    from pyac.tasks.mnist.encoding import PixelAssemblyEncoder

    rng = np.random.default_rng(1)
    encoder = PixelAssemblyEncoder(num_pixels=4, neurons_per_pixel=3, active_pixels=2, rng=rng)
    image = np.array([[0.1, 0.9], [0.7, 0.2]], dtype=np.float64)

    first = encoder.encode(image)
    second = encoder.encode(image)

    assert first.indices.tolist() == second.indices.tolist()
    assert first.area_name == "X"
    assert first.indices.size == 6


def test_pixel_encoder_uses_intensity_rank_not_label_information():
    from pyac.tasks.mnist.encoding import PixelAssemblyEncoder

    encoder = PixelAssemblyEncoder(num_pixels=4, neurons_per_pixel=2, active_pixels=1, rng=np.random.default_rng(2))
    image = np.array([[0.0, 0.1], [0.2, 1.0]], dtype=np.float64)

    assembly = encoder.encode(image)
    expected_pool = set(encoder.pixel_indices[3].tolist())

    assert set(assembly.indices.tolist()) == expected_pool
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_encoding.py -q`

Expected: FAIL because encoder does not exist.

**Step 3: Implement encoder**

Implement an auditable pixel encoder:

- Construct fixed disjoint sensory pools per pixel.
- Select top-intensity pixels from an image.
- Return the union of their sensory neurons as an `Assembly(area_name="X", indices=...)`.
- This is not a classifier: it uses image intensity only.

**Step 4: Run tests**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_encoding.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- pyac/src/pyac/tasks/mnist/encoding.py tests/test_mnist_encoding.py`

Expected: encoder only.

## Task 3: Overlap And Margin Measures

**Files:**
- Modify: `pyac/src/pyac/measures/overlap.py`
- Test: `tests/test_mnist_measures.py`

**Step 1: Write failing tests**

```python
from __future__ import annotations

import numpy as np


def test_class_overlap_vector_and_margin_are_measured_from_assemblies():
    from pyac.core.types import Assembly
    from pyac.measures.overlap import class_overlap_vector, correct_class_margin

    active = Assembly("Y", np.array([0, 1, 2, 10]))
    prototypes = {
        0: Assembly("Y", np.array([0, 1, 2, 3])),
        1: Assembly("Y", np.array([10, 11, 12, 13])),
    }

    overlaps = class_overlap_vector(active, prototypes, num_classes=2)
    margin = correct_class_margin(overlaps, target=0)

    assert overlaps.tolist() == [0.75, 0.25]
    assert margin.correct_overlap == 0.75
    assert margin.strongest_wrong_overlap == 0.25
    assert margin.margin == 0.5
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_measures.py -q`

Expected: FAIL because functions do not exist.

**Step 3: Implement minimal functions**

Add:

- `ClassMargin` dataclass.
- `class_overlap_vector(active, prototypes, num_classes)` using intersection divided by prototype size.
- `correct_class_margin(overlaps, target)`.

**Step 4: Run tests**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_measures.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- pyac/src/pyac/measures/overlap.py tests/test_mnist_measures.py`

Expected: overlap helpers only.

## Task 4: MNIST AC Model Construction And Training

**Files:**
- Create: `pyac/src/pyac/tasks/mnist/protocol.py`
- Modify: `pyac/src/pyac/tasks/mnist/__init__.py`
- Test: `tests/test_mnist_protocol.py`

**Step 1: Write failing tests**

```python
from __future__ import annotations

import numpy as np


def test_build_mnist_network_uses_sensory_and_coding_areas():
    from pyac.tasks.mnist.protocol import build_mnist_network

    network, task = build_mnist_network(n=200, k=10, p=0.2, beta=0.1, rng=np.random.default_rng(3))

    assert task.area_map == {"sensory": "X", "coding": "Y"}
    assert network.areas_by_name["X"].dynamics_type == "feedforward"
    assert network.areas_by_name["Y"].dynamics_type == "recurrent"
    assert ("X", "Y") in network.weights
    assert ("Y", "Y") in network.weights


def test_training_creates_one_class_assembly_per_digit():
    from pyac.tasks.mnist.protocol import build_mnist_network, train_mnist_assemblies

    network, task = build_mnist_network(n=200, k=10, p=0.2, beta=0.1, rng=np.random.default_rng(4))
    images = np.zeros((20, 28, 28), dtype=np.float64)
    labels = np.array([digit for digit in range(10) for _ in range(2)], dtype=np.int64)
    for idx, label in enumerate(labels):
        images[idx, label:label + 1, :] = 1.0

    train_mnist_assemblies(network, task, images, labels, presentation_rounds=1, settle_steps=1)

    assert set(task.class_assemblies.keys()) == set(range(10))
    assert all(assembly.area_name == "Y" for assembly in task.class_assemblies.values())
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_protocol.py -q`

Expected: FAIL because protocol does not exist.

**Step 3: Implement minimal AC protocol**

Add:

- `MnistTask` dataclass with encoder, area map, class assemblies, hyperparameters.
- `build_mnist_network(...)` with `X` feedforward and `Y` recurrent areas plus `X -> Y` fibre.
- `train_mnist_assemblies(...)` that presents image stimuli, turns plasticity on, and records one prototype assembly per class from observed `Y` activations.

Keep implementation simple. Do not add classifier weights, dense neural layers, or label lookup at evaluation time.

**Step 4: Run tests**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_protocol.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- pyac/src/pyac/tasks/mnist/protocol.py tests/test_mnist_protocol.py`

Expected: model construction and training only.

## Task 5: Frozen MNIST Evaluation And `t` Sweep

**Files:**
- Modify: `pyac/src/pyac/tasks/mnist/protocol.py`
- Test: `tests/test_mnist_evaluation.py`

**Step 1: Write failing tests**

```python
from __future__ import annotations

import numpy as np


def _trained_tiny_model():
    from pyac.tasks.mnist.protocol import build_mnist_network, train_mnist_assemblies
    network, task = build_mnist_network(n=240, k=12, p=0.25, beta=0.1, rng=np.random.default_rng(5))
    images = np.zeros((20, 28, 28), dtype=np.float64)
    labels = np.array([digit for digit in range(10) for _ in range(2)], dtype=np.int64)
    for idx, label in enumerate(labels):
        images[idx, label:label + 1, :] = 1.0
    train_mnist_assemblies(network, task, images, labels, presentation_rounds=1, settle_steps=1)
    return network, task, images, labels


def test_evaluate_mnist_example_records_overlap_margin_and_trajectory():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example
    network, task, images, labels = _trained_tiny_model()

    row = evaluate_mnist_example(network, task, images[0], int(labels[0]), instance_id="ex0", t=2)

    assert row["experiment"] == "mnist"
    assert row["target"] == int(labels[0])
    assert len(row["overlaps"]) == 10
    assert row["margin"] == row["correct_overlap"] - row["strongest_wrong_overlap"]
    assert row["plasticity_on"] is False
    assert len(row["trajectory"]) == 3


def test_evaluation_does_not_change_weights():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example
    network, task, images, labels = _trained_tiny_model()
    before = {fiber: weights.copy() for fiber, weights in network.weights.items()}

    evaluate_mnist_example(network, task, images[0], int(labels[0]), instance_id="ex0", t=3)

    for fiber, weights in network.weights.items():
        diff = (weights != before[fiber]).nnz
        assert diff == 0
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_evaluation.py -q`

Expected: FAIL because evaluation functions do not exist.

**Step 3: Implement evaluation**

Add:

- network activity reset helper local to MNIST protocol or reusable if already available.
- `decode_mnist_class(...)` from measured overlap vector.
- `evaluate_mnist_example(...)` with `plasticity_on=False`, trajectory capture for steps `0..t`, and exact row fields.
- `evaluate_mnist_t_sweep(...)` that trains once and evaluates multiple `t` values without rebuilding or retraining.

**Step 4: Run tests**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_evaluation.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- pyac/src/pyac/tasks/mnist/protocol.py tests/test_mnist_evaluation.py`

Expected: evaluation only.

## Task 6: Experiment Suite MNIST Runner

**Files:**
- Modify: `experiment_suite/config.py`
- Modify: `experiment_suite/jobs.py`
- Modify: `run_experiment_suite.py`
- Create: `experiment_suite/runners/mnist_ac_runner.py`
- Create: `experiments/mnist_ac_dev.yaml`
- Test: `tests/test_mnist_suite_runner.py`

**Step 1: Write failing tests**

Use monkeypatching and tiny IDX files to avoid requiring full MNIST in unit tests.

```python
from __future__ import annotations

import gzip
import struct

import numpy as np


def _write_idx_pair(root, split, images, labels):
    image_name = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    label_name = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    with gzip.open(root / image_name, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, images.shape[0], 28, 28))
        handle.write(images.astype(np.uint8).tobytes())
    with gzip.open(root / label_name, "wb") as handle:
        handle.write(struct.pack(">II", 2049, labels.shape[0]))
        handle.write(labels.astype(np.uint8).tobytes())


def test_mnist_ac_runner_returns_theory_rows(tmp_path):
    from experiment_suite.config import ExperimentCondition, ModelConfig
    from experiment_suite.jobs import ExperimentJob
    from experiment_suite.runners.mnist_ac_runner import run_mnist_ac_job

    data_dir = tmp_path / "mnist"
    data_dir.mkdir()
    train_images = np.zeros((20, 28, 28), dtype=np.uint8)
    train_labels = np.array([digit for digit in range(10) for _ in range(2)], dtype=np.uint8)
    test_images = np.zeros((2, 28, 28), dtype=np.uint8)
    test_labels = np.array([0, 1], dtype=np.uint8)
    _write_idx_pair(data_dir, "train", train_images, train_labels)
    _write_idx_pair(data_dir, "test", test_images, test_labels)

    job = ExperimentJob(
        suite_name="mnist-dev",
        seed=1,
        condition=ExperimentCondition(list_type="MNIST"),
        model=ModelConfig(family="MNIST_AC", values={
            "data_dir": str(data_dir), "n": 240, "k": 12, "p": 0.25, "beta": 0.1,
            "train_limit": 20, "test_limit": 2, "t_values": [0, 1],
        }),
    )

    rows = run_mnist_ac_job(job)

    assert {row["experiment"] for row in rows} == {"mnist"}
    assert {row["t"] for row in rows} == {0, 1}
    assert all(len(row["overlaps"]) == 10 for row in rows)
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_suite_runner.py -q`

Expected: FAIL because runner does not exist or dispatcher does not support it.

**Step 3: Implement runner**

Add `run_mnist_ac_job(job)` that:

- reads `data_dir`, `train_limit`, `test_limit`, `t_values`, `n`, `k`, `p`, `beta`, `presentation_rounds`, `settle_steps`;
- loads real MNIST train/test splits;
- builds one model per job seed;
- trains once;
- evaluates all requested `t` values;
- returns raw rows.

Update dispatcher so `family == "MNIST_AC"` works without disturbing existing `AC` pointer jobs.

**Step 4: Run tests**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_suite_runner.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- experiment_suite run_experiment_suite.py experiments/mnist_ac_dev.yaml tests/test_mnist_suite_runner.py`

Expected: MNIST runner support only.

## Task 7: MNIST Raw Output Serialization

**Files:**
- Modify: `experiment_suite/aggregate.py`
- Test: `tests/test_mnist_output_schema.py`

**Step 1: Write failing tests**

```python
from __future__ import annotations


def test_raw_results_can_write_vector_fields(tmp_path):
    from experiment_suite.aggregate import write_raw_results

    rows = [{
        "experiment": "mnist",
        "seed": 1,
        "t": 0,
        "instance_id": "0",
        "target": 1,
        "prediction": 1,
        "correct": True,
        "overlaps": [0.1] * 10,
        "trajectory": [1, 1],
        "margin": 0.2,
    }]

    path = write_raw_results(rows, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "overlaps" in text
    assert "[0.1" in text
```

**Step 2: Run tests**

Run: `pytest tests/test_mnist_output_schema.py -q`

Expected: It may already pass. If it fails because CSV cannot serialize lists consistently, add explicit JSON serialization for list/dict values.

**Step 3: Implement only if needed**

If needed, add a small `_serialize_cell(value)` helper in `aggregate.py` that converts `list` and `dict` values to JSON strings before CSV writing.

**Step 4: Run tests**

Run: `pytest tests/test_mnist_output_schema.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- experiment_suite/aggregate.py tests/test_mnist_output_schema.py`

Expected: output serialization only.

## Task 8: MNIST Plots

**Files:**
- Modify: `experiment_suite/plots.py`
- Test: `tests/test_mnist_plots.py`

**Step 1: Write failing tests**

```python
from __future__ import annotations

import csv


def test_generate_mnist_plots_creates_minimum_figures(tmp_path):
    from experiment_suite.plots import generate_mnist_ac_plots

    raw = tmp_path / "raw_results.csv"
    rows = []
    for t in [0, 1, 2]:
        for target in range(10):
            rows.append({
                "experiment": "mnist", "seed": 1, "t": t, "target": target,
                "prediction": target, "correct": "True", "margin": 0.1 + t,
                "correct_overlap": 0.5, "strongest_wrong_overlap": 0.2,
                "overlaps": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
            })
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out = tmp_path / "plots"
    generate_mnist_ac_plots(raw, out)

    assert (out / "mnist_accuracy_vs_t.png").exists()
    assert (out / "mnist_margin_vs_t.png").exists()
    assert (out / "mnist_per_class_accuracy_vs_t.png").exists()
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_mnist_plots.py -q`

Expected: FAIL because `generate_mnist_ac_plots` does not exist.

**Step 3: Implement minimal plots**

Add `generate_mnist_ac_plots(raw_results_csv, plots_dir)` that creates:

- `mnist_accuracy_vs_t.png`
- `mnist_margin_vs_t.png`
- `mnist_per_class_accuracy_vs_t.png`
- optionally confusion matrices if enough rows exist.

Use existing plotting style where possible. Keep plotting derived only from raw rows.

**Step 4: Run tests**

Run: `pytest tests/test_mnist_plots.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- experiment_suite/plots.py tests/test_mnist_plots.py`

Expected: MNIST plot support only.

## Task 9: Dispatcher Plot Integration

**Files:**
- Modify: `run_experiment_suite.py`
- Test: `tests/test_run_experiment_suite.py`

**Step 1: Add/adjust tests**

Add a test that when all rows have `experiment == "mnist"`, `_generate_plots` dispatches to `generate_mnist_ac_plots`.

**Step 2: Run test to verify failure**

Run: `pytest tests/test_run_experiment_suite.py -q`

Expected: FAIL until dispatcher recognizes MNIST rows.

**Step 3: Implement minimal dispatcher change**

In `_generate_plots`, detect `experiment == "mnist"` before pointer-specific `list_type` logic and call `suite_plots.generate_mnist_ac_plots`.

**Step 4: Run tests**

Run: `pytest tests/test_run_experiment_suite.py -q`

Expected: PASS.

**Step 5: Checkpoint**

Run: `git diff -- run_experiment_suite.py tests/test_run_experiment_suite.py`

Expected: plot dispatch only.

## Task 10: End-To-End Dev Run Contract

**Files:**
- Modify: `experiments/mnist_ac_dev.yaml`
- Test: `tests/test_mnist_end_to_end.py`

**Step 1: Write test for missing data failure**

```python
from __future__ import annotations

import pytest


def test_mnist_suite_fails_clearly_without_real_mnist(tmp_path):
    from run_experiment_suite import run_suite

    config = tmp_path / "mnist_missing.yaml"
    config.write_text(
        """
suite_name: mnist-missing
output_dir: {output_dir}
seeds: [1]
conditions:
  - list_type: MNIST
models:
  MNIST_AC:
    - model_name: MNIST-AC-dev
      data_dir: {data_dir}
      n: 240
      k: 12
      p: 0.25
      beta: 0.1
      train_limit: 20
      test_limit: 2
      t_values: [0, 1]
""".format(output_dir=tmp_path / "out", data_dir=tmp_path / "missing"),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="MNIST"):
        run_suite(config)
```

**Step 2: Run test**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_end_to_end.py -q`

Expected: PASS once runner and loader are wired.

**Step 3: Ensure dev config is explicit**

Use `experiments/mnist_ac_dev.yaml` with a placeholder real-data path such as `data/mnist/raw`, small limits, and `t_values: [0, 1, 2, 4, 8, 12, 16, 20]`.

**Step 4: Run focused MNIST tests**

Run: `PYTHONPATH=pyac/src pytest tests/test_mnist_data.py tests/test_mnist_encoding.py tests/test_mnist_measures.py tests/test_mnist_protocol.py tests/test_mnist_evaluation.py tests/test_mnist_suite_runner.py tests/test_mnist_output_schema.py tests/test_mnist_plots.py tests/test_mnist_end_to_end.py -q`

Expected: PASS.

**Step 5: Run broader suite**

Run: `PYTHONPATH=pyac/src pytest -q`

Expected: PASS or only documented unrelated failures. Investigate any failure caused by the MNIST changes.

## Final Verification

Run these before claiming completion:

- `PYTHONPATH=pyac/src pytest -q`
- If real MNIST files exist at the configured path: `PYTHONPATH=pyac/src python run_experiment_suite.py --config experiments/mnist_ac_dev.yaml`
- Inspect output directory for `raw_results.csv`, summary, and MNIST plots.

## Completion Criteria

- Real MNIST IDX files are required for real runs.
- MNIST AC model uses sparse assemblies, k-cap dynamics, recurrent coding area, Hebbian training, and frozen evaluation.
- Raw rows contain `experiment`, `seed`, `theta_id`, `n`, `k`, `p`, `beta`, `t`, `instance_id`, `target`, `prediction`, `correct`, `overlaps`, `correct_overlap`, `strongest_wrong_overlap`, `margin`, `trajectory`, and `plasticity_on`.
- Accuracy and margin plots derive from raw rows.
- Existing pointer tests still pass.
- No non-AC shortcut or fake-result path is introduced.
