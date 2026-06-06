from __future__ import annotations

import numpy as np
import pytest

from pyac.core.types import Assembly


def _trained_tiny_model():
    from pyac.tasks.mnist.protocol import build_mnist_network, train_mnist_assemblies

    network, task = build_mnist_network(
        n=240,
        k=12,
        p=0.25,
        beta=0.1,
        rng=np.random.default_rng(5),
    )
    images = np.zeros((20, 28, 28), dtype=np.float64)
    labels = np.array([digit for digit in range(10) for _ in range(2)], dtype=np.int64)
    for idx, label in enumerate(labels):
        images[idx, label : label + 1, :] = 1.0

    train_mnist_assemblies(
        network,
        task,
        images,
        labels,
        presentation_rounds=1,
        settle_steps=1,
    )
    return network, task, images, labels


def test_evaluate_mnist_example_records_overlap_margin_and_trajectory():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, labels = _trained_tiny_model()

    row = evaluate_mnist_example(
        network,
        task,
        images[0],
        int(labels[0]),
        instance_id="ex0",
        t=2,
    )

    assert row["experiment"] == "mnist"
    assert row["target"] == int(labels[0])
    assert len(row["overlaps"]) == 10
    assert row["margin"] == row["correct_overlap"] - row["strongest_wrong_overlap"]
    assert row["plasticity_on"] is False
    assert len(row["trajectory"]) == 3
    assert len(row["overlap_trajectory"]) == 3
    assert all(len(overlaps) == 10 for overlaps in row["overlap_trajectory"])


def test_evaluate_mnist_example_records_t_zero_initial_state():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, labels = _trained_tiny_model()

    row = evaluate_mnist_example(
        network,
        task,
        images[0],
        int(labels[0]),
        instance_id="ex0",
        t=0,
    )

    assert row["t"] == 0
    assert len(row["trajectory"]) == 1
    assert len(row["overlap_trajectory"]) == 1


def test_evaluate_mnist_example_includes_task_seed_when_available():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, labels = _trained_tiny_model()
    task.task_seed = 123

    row = evaluate_mnist_example(
        network,
        task,
        images[0],
        int(labels[0]),
        instance_id="ex0",
        t=1,
    )

    assert row["seed"] is None
    assert row["task_seed"] == 123


def test_evaluate_mnist_example_rejects_incomplete_class_assemblies():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, labels = _trained_tiny_model()
    del task.class_assemblies[9]

    with pytest.raises(ValueError, match="MNIST.*class assemblies"):
        evaluate_mnist_example(
            network,
            task,
            images[0],
            int(labels[0]),
            instance_id="ex0",
            t=1,
        )


def test_evaluate_mnist_example_rejects_non_mnist_target():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, _labels = _trained_tiny_model()

    with pytest.raises(ValueError, match="target must be an MNIST digit"):
        evaluate_mnist_example(
            network,
            task,
            images[0],
            10,
            instance_id="ex0",
            t=1,
        )


def test_evaluate_mnist_example_rejects_bool_target():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, _labels = _trained_tiny_model()

    with pytest.raises(ValueError, match="target must be an MNIST digit"):
        evaluate_mnist_example(
            network,
            task,
            images[0],
            True,
            instance_id="ex0",
            t=1,
        )


def test_evaluate_mnist_example_rejects_empty_class_assembly():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, labels = _trained_tiny_model()
    task.class_assemblies[0] = Assembly("Y", np.array([], dtype=np.int64))

    with pytest.raises(ValueError, match="MNIST.*class assemblies"):
        evaluate_mnist_example(
            network,
            task,
            images[0],
            int(labels[0]),
            instance_id="ex0",
            t=1,
        )


def test_evaluate_mnist_example_rejects_wrong_area_class_assembly():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, labels = _trained_tiny_model()
    task.class_assemblies[0] = Assembly("Z", task.class_assemblies[0].indices)

    with pytest.raises(ValueError, match="MNIST.*class assemblies"):
        evaluate_mnist_example(
            network,
            task,
            images[0],
            int(labels[0]),
            instance_id="ex0",
            t=1,
        )


def test_evaluation_does_not_change_weights():
    from pyac.tasks.mnist.protocol import evaluate_mnist_example

    network, task, images, labels = _trained_tiny_model()
    before = {fiber: weights.copy() for fiber, weights in network.weights.items()}

    evaluate_mnist_example(
        network,
        task,
        images[0],
        int(labels[0]),
        instance_id="ex0",
        t=3,
    )

    for fiber, weights in network.weights.items():
        diff = (weights != before[fiber]).nnz
        assert diff == 0


def test_evaluate_mnist_t_sweep_reuses_model_and_preserves_weights():
    from pyac.tasks.mnist.protocol import evaluate_mnist_t_sweep

    network, task, images, labels = _trained_tiny_model()
    before = {fiber: weights.copy() for fiber, weights in network.weights.items()}
    t_values = [0, 2, 4]

    rows = evaluate_mnist_t_sweep(
        network,
        task,
        images[:3],
        labels[:3],
        t_values,
        instance_ids=["a", "b", "c"],
    )

    assert len(rows) == 3 * len(t_values)
    assert {row["t"] for row in rows} == set(t_values)
    assert {row["instance_id"] for row in rows} == {"a", "b", "c"}
    for row in rows:
        assert len(row["trajectory"]) == row["t"] + 1
        assert row["plasticity_on"] is False

    for fiber, weights in network.weights.items():
        diff = (weights != before[fiber]).nnz
        assert diff == 0


def test_evaluate_mnist_t_sweep_passes_stimulus_mode_through():
    from pyac.tasks.mnist.protocol import evaluate_mnist_t_sweep

    network, task, images, labels = _trained_tiny_model()

    rows = evaluate_mnist_t_sweep(
        network,
        task,
        images[:2],
        labels[:2],
        [0, 1],
        stimulus_mode="transient",
    )

    assert {row["stimulus_mode"] for row in rows} == {"transient"}


def test_evaluate_mnist_t_sweep_rejects_bool_labels():
    from pyac.tasks.mnist.protocol import evaluate_mnist_t_sweep

    network, task, images, _labels = _trained_tiny_model()

    with pytest.raises(ValueError, match="target must be an MNIST digit"):
        evaluate_mnist_t_sweep(
            network,
            task,
            images[:1],
            np.array([True], dtype=np.bool_),
            [1],
        )


def test_mnist_package_exports_evaluation_helpers():
    from pyac.tasks.mnist import (
        decode_mnist_class,
        evaluate_mnist_example,
        evaluate_mnist_t_sweep,
    )

    assert decode_mnist_class is not None
    assert evaluate_mnist_example is not None
    assert evaluate_mnist_t_sweep is not None
