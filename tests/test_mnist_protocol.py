from __future__ import annotations

import numpy as np
import pytest

from pyac.core.types import Assembly


def test_build_mnist_network_uses_sensory_and_coding_areas():
    from pyac.tasks.mnist.protocol import build_mnist_network

    network, task = build_mnist_network(
        n=200,
        k=10,
        p=0.2,
        beta=0.1,
        rng=np.random.default_rng(3),
    )

    assert task.area_map == {"sensory": "X", "coding": "Y"}
    assert network.areas_by_name["X"].dynamics_type == "feedforward"
    assert network.areas_by_name["Y"].dynamics_type == "recurrent"
    assert ("X", "Y") in network.weights
    assert ("Y", "Y") in network.weights


def test_training_creates_one_class_assembly_per_digit():
    from pyac.tasks.mnist.protocol import build_mnist_network, train_mnist_assemblies

    network, task = build_mnist_network(
        n=200,
        k=10,
        p=0.2,
        beta=0.1,
        rng=np.random.default_rng(4),
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

    assert set(task.class_assemblies.keys()) == set(range(10))
    assert all(
        assembly.area_name == "Y" for assembly in task.class_assemblies.values()
    )
    assert all(assembly.indices.size == 10 for assembly in task.class_assemblies.values())


def test_training_updates_hebbian_weights():
    from pyac.tasks.mnist.protocol import build_mnist_network, train_mnist_assemblies

    network, task = build_mnist_network(
        n=200,
        k=10,
        p=0.2,
        beta=0.5,
        rng=np.random.default_rng(5),
    )
    images = np.zeros((2, 28, 28), dtype=np.float64)
    images[:, :4, :] = 1.0
    labels = np.array([0, 0], dtype=np.int64)
    before = network.weights[("X", "Y")].copy()

    train_mnist_assemblies(network, task, images, labels)

    assert (network.weights[("X", "Y")] != before).nnz > 0


def test_training_aggregates_repeated_label_observations_and_clears_examples():
    from pyac.tasks.mnist.protocol import MnistTask, train_mnist_assemblies

    class Area:
        def __init__(self, n: int, k: int):
            self.n = n
            self.k = k

    class Encoder:
        area_name = "X"

        def encode(self, image):
            return Assembly(area_name="X", indices=np.array([0], dtype=np.int64))

    class RecordingNetwork:
        def __init__(self):
            self.areas_by_name = {"X": Area(n=1, k=1), "Y": Area(n=4, k=2)}
            self.activations = {
                "X": np.array([0], dtype=np.int64),
                "Y": np.array([3], dtype=np.int64),
            }
            self.outputs = [
                np.array([0, 1], dtype=np.int64),
                np.array([0, 2], dtype=np.int64),
                np.array([2, 3], dtype=np.int64),
            ]
            self.step_calls = 0

        def step(self, external_stimuli, plasticity_on):
            assert plasticity_on is True
            assert self.activations["X"].size == 0
            assert self.activations["Y"].size == 0
            self.activations["Y"] = self.outputs[self.step_calls]
            self.step_calls += 1

        def get_assembly(self, area_name):
            return Assembly(area_name=area_name, indices=self.activations[area_name])

    task = MnistTask(
        encoder=Encoder(),
        area_map={"sensory": "X", "coding": "Y"},
        k=2,
    )

    train_mnist_assemblies(
        RecordingNetwork(),
        task,
        np.zeros((3, 1), dtype=np.float64),
        np.array([1, 1, 1], dtype=np.int64),
    )

    assert task.class_assemblies[1].indices.tolist() == [0, 2]


def test_training_rejects_non_mnist_labels():
    from pyac.tasks.mnist.protocol import build_mnist_network, train_mnist_assemblies

    network, task = build_mnist_network(
        n=200,
        k=10,
        p=0.2,
        beta=0.1,
        rng=np.random.default_rng(6),
    )

    with pytest.raises(ValueError, match="labels must be MNIST digits"):
        train_mnist_assemblies(
            network,
            task,
            np.zeros((1, 28, 28), dtype=np.float64),
            np.array([10], dtype=np.int64),
        )


def test_build_mnist_network_rejects_encoder_area_mismatch():
    from pyac.tasks.mnist.encoding import PixelAssemblyEncoder
    from pyac.tasks.mnist.protocol import build_mnist_network

    encoder = PixelAssemblyEncoder(area_name="Z")

    with pytest.raises(ValueError, match="encoder area_name must be 'X'"):
        build_mnist_network(
            n=200,
            k=10,
            p=0.2,
            beta=0.1,
            rng=np.random.default_rng(7),
            encoder=encoder,
        )


def test_mnist_protocol_exports_task_helpers():
    from pyac.tasks.mnist import MnistTask, build_mnist_network, train_mnist_assemblies

    assert MnistTask is not None
    assert build_mnist_network is not None
    assert train_mnist_assemblies is not None
