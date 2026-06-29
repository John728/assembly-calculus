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


def test_evaluate_mnist_retention_sweep_records_cue_and_post_removal_metrics():
    from pyac.tasks.mnist.protocol import evaluate_mnist_retention_sweep

    network, task, images, labels = _trained_tiny_model()
    before = {fiber: weights.copy() for fiber, weights in network.weights.items()}

    rows = evaluate_mnist_retention_sweep(
        network,
        task,
        images[:2],
        labels[:2],
        cue_duration_values=[1, 3],
        retention_ell_values=[0, 2],
        instance_ids=["a", "b"],
    )

    assert len(rows) == 2 * 2 * 2
    assert {row["experiment"] for row in rows} == {"mnist_retention"}
    assert {row["stimulus_mode"] for row in rows} == {"cue_then_off"}
    assert {row["s"] for row in rows} == {1, 3}
    assert {row["cue_duration_s"] for row in rows} == {1, 3}
    assert {row["ell"] for row in rows} == {0, 2}
    assert {row["retention_ell"] for row in rows} == {0, 2}
    assert {row["instance_id"] for row in rows} == {"a", "b"}
    for row in rows:
        assert row["t"] == row["s"] + row["ell"]
        assert row["correct_score"] == row["correct_overlap"]
        assert row["strongest_wrong_score"] == row["strongest_wrong_overlap"]
        assert row["margin"] == row["correct_score"] - row["strongest_wrong_score"]
        assert isinstance(row["correct_at_t1"], bool)
        assert isinstance(row["stayed_correct"], bool)
        assert isinstance(row["became_correct_later"], bool)
        assert row["retention_time"] >= 0
        assert row["plasticity_on"] is False

    for fiber, weights in network.weights.items():
        diff = (weights != before[fiber]).nnz
        assert diff == 0


def test_evaluate_mnist_retention_sweep_rejects_invalid_horizons():
    from pyac.tasks.mnist.protocol import evaluate_mnist_retention_sweep

    network, task, images, labels = _trained_tiny_model()

    with pytest.raises(ValueError, match="cue_duration_values"):
        evaluate_mnist_retention_sweep(network, task, images[:1], labels[:1], [0], [0])
    with pytest.raises(ValueError, match="retention_ell_values"):
        evaluate_mnist_retention_sweep(network, task, images[:1], labels[:1], [1], [-1])


def test_evaluate_mnist_retention_sweep_rejects_bool_labels():
    from pyac.tasks.mnist.protocol import evaluate_mnist_retention_sweep

    network, task, images, _labels = _trained_tiny_model()

    with pytest.raises(ValueError, match="target must be an MNIST digit"):
        evaluate_mnist_retention_sweep(
            network,
            task,
            images[:1],
            np.array([True], dtype=np.bool_),
            cue_duration_values=[1],
            retention_ell_values=[0],
        )


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


def test_evaluate_mnist_sequence_rejects_bool_labels():
    from pyac.tasks.mnist.protocol import evaluate_mnist_sequence

    network, task, images, _labels = _trained_tiny_model()

    with pytest.raises(ValueError, match="labels must be MNIST digits"):
        evaluate_mnist_sequence(
            network,
            task,
            images[:1],
            np.array([True], dtype=np.bool_),
            sequence_digits=[1],
            steps_per_digit=1,
        )


def test_mnist_sequence_keeps_state_across_digit_changes():
    from types import SimpleNamespace

    from pyac.tasks.mnist.protocol import evaluate_mnist_sequence

    class MarkerEncoder:
        area_name = "X"

        def encode(self, image):
            return Assembly("X", np.array([int(image[0, 0])], dtype=np.int64))

    class RecordingNetwork:
        def __init__(self):
            self.areas_by_name = {
                "X": SimpleNamespace(n=10, k=1),
                "Y": SimpleNamespace(n=10, k=1),
            }
            self.activations = {
                "X": np.array([9], dtype=np.int64),
                "Y": np.array([9], dtype=np.int64),
            }
            self.step_count = 99
            self.had_previous_y: list[bool] = []
            self.plasticity_calls: list[bool] = []

        def step(self, external_stimuli=None, plasticity_on=True, biases=None):
            self.had_previous_y.append(self.activations["Y"].size > 0)
            self.plasticity_calls.append(bool(plasticity_on))
            stimulus = external_stimuli["X"]
            active = int(np.flatnonzero(stimulus)[0])
            self.activations["X"] = np.array([active], dtype=np.int64)
            self.activations["Y"] = np.array([active], dtype=np.int64)
            self.step_count += 1

        def get_assembly(self, area_name):
            return Assembly(area_name, self.activations[area_name])

    network = RecordingNetwork()
    task = SimpleNamespace(
        encoder=MarkerEncoder(),
        area_map={"sensory": "X", "coding": "Y"},
        class_assemblies={digit: Assembly("Y", np.array([digit], dtype=np.int64)) for digit in range(10)},
        n=10,
        k=1,
        p=0.1,
        beta=1.0,
        coding_bias=np.zeros(10, dtype=np.float64),
        seed=7,
    )
    images = np.zeros((2, 28, 28), dtype=np.float64)
    images[0, 0, 0] = 0
    images[1, 0, 0] = 1
    labels = np.array([0, 1], dtype=np.int64)

    rows = evaluate_mnist_sequence(
        network,
        task,
        images,
        labels,
        sequence_digits=[0, 1],
        steps_per_digit=2,
        instance_ids=["zero", "one"],
    )

    assert len(rows) == 4
    assert [row["phase_digit"] for row in rows] == [0, 0, 1, 1]
    assert [row["step_in_phase"] for row in rows] == [0, 1, 0, 1]
    assert [row["sequence_step"] for row in rows] == [0, 1, 2, 3]
    assert [row["instance_id"] for row in rows] == ["zero", "zero", "one", "one"]
    assert network.had_previous_y == [False, True, True, True]
    assert network.plasticity_calls == [False, False, False, False]
    assert all(row["plasticity_on"] is False for row in rows)
    assert all(len(row["overlaps"]) == 10 for row in rows)
    assert all(len(row["trajectory"]) == row["sequence_step"] + 1 for row in rows)


def test_mnist_package_exports_evaluation_helpers():
    from pyac.tasks.mnist import (
        decode_mnist_class,
        evaluate_mnist_example,
        evaluate_mnist_retention_sweep,
        evaluate_mnist_sequence,
        evaluate_mnist_t_sweep,
    )

    assert decode_mnist_class is not None
    assert evaluate_mnist_example is not None
    assert evaluate_mnist_retention_sweep is not None
    assert evaluate_mnist_sequence is not None
    assert evaluate_mnist_t_sweep is not None
