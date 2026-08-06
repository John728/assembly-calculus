from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "thesis_c"
    / "static"
    / "generate_static_training_dynamics.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("static_training_dynamics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_balanced_subset_uses_equal_nested_class_prefixes():
    module = load_module()
    labels = np.repeat(np.arange(10), 7)
    images = np.arange(len(labels))[:, None]

    selected_images, selected_labels, selected_ids = module.balanced_subset(
        images, labels, per_class=3
    )

    assert selected_images.shape == (30, 1)
    assert np.array_equal(np.bincount(selected_labels), np.full(10, 3))
    for digit in range(10):
        expected = np.flatnonzero(labels == digit)[:3]
        actual = selected_ids[selected_labels == digit]
        assert np.array_equal(actual, expected)


def test_training_schedules_fix_updates_and_distinguish_order():
    module = load_module()
    labels = np.repeat(np.arange(10), module.TRAIN_PER_CLASS)
    images = np.arange(len(labels))[:, None]

    repeated, repeated_labels = module.build_training_schedule(
        images, labels, "repeat_1x50"
    )
    blocked, blocked_labels = module.build_training_schedule(
        images, labels, "blocked_10x5"
    )
    interleaved, interleaved_labels = module.build_training_schedule(
        images, labels, "interleaved_10x5"
    )
    distinct, distinct_labels = module.build_training_schedule(
        images, labels, "distinct_50"
    )

    for scheduled_labels in (
        repeated_labels,
        blocked_labels,
        interleaved_labels,
        distinct_labels,
    ):
        assert np.array_equal(
            np.bincount(scheduled_labels), np.full(10, module.TRAIN_PER_CLASS)
        )

    assert np.unique(repeated[repeated_labels == 0]).size == 1
    assert np.unique(blocked[blocked_labels == 0]).size == 10
    assert np.unique(interleaved[interleaved_labels == 0]).size == 10
    assert np.unique(distinct[distinct_labels == 0]).size == 50
    assert np.array_equal(blocked[blocked_labels == 0][:5], np.zeros((5, 1)))
    assert np.array_equal(
        interleaved[interleaved_labels == 0][:10, 0], np.arange(10)
    )


def test_recurrent_gain_intervention_preserves_sensory_weights_and_topology():
    module = load_module()
    coding_n = 20
    sensory = csr_matrix(np.full((3, coding_n), 0.25))
    recurrent_dense = np.zeros((coding_n, coding_n), dtype=float)
    assemblies = {}
    for digit in range(10):
        indices = np.array([2 * digit, 2 * digit + 1])
        assemblies[digit] = SimpleNamespace(indices=indices)
        recurrent_dense[indices[0], indices[1]] = 0.2 + 0.05 * digit
        recurrent_dense[indices[1], indices[0]] = 0.2 + 0.05 * digit
    recurrent = csr_matrix(recurrent_dense)
    network = SimpleNamespace(
        weights={("X", "Y"): sensory.copy(), ("Y", "Y"): recurrent}
    )
    task = SimpleNamespace(
        area_map={"sensory": "X", "coding": "Y"},
        class_assemblies=assemblies,
    )
    original_sensory = network.weights[("X", "Y")].copy()
    original_pattern = network.weights[("Y", "Y")].copy()
    original_pattern.data[:] = 1.0
    module.K = 2

    targets = {digit: 0.4 for digit in range(10)}
    module.assign_recurrent_gains(network, task, targets)
    measured = module.class_within_gains(network, task)

    assert all(np.isclose(measured[digit], 0.4) for digit in range(10))
    assert (network.weights[("X", "Y")] != original_sensory).nnz == 0
    new_pattern = network.weights[("Y", "Y")].copy()
    new_pattern.data[:] = 1.0
    assert (new_pattern != original_pattern).nnz == 0


def test_settling_readout_is_the_last_observed_change():
    module = load_module()

    assert module.last_change_readout([2, 2, 2]) == 1
    assert module.last_change_readout([2, 3, 3, 3]) == 2
    assert module.last_change_readout([2, 3, 2, 2]) == 3
