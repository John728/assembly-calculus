from __future__ import annotations

import numpy as np


def test_class_organized_training_records_negative_bias_for_final_class_assemblies():
    from pyac.tasks.mnist.protocol import build_mnist_network, train_mnist_assemblies

    network, task = build_mnist_network(
        n=240,
        k=12,
        p=0.25,
        beta=0.1,
        rng=np.random.default_rng(7),
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
        presentation_rounds=2,
        settle_steps=1,
        class_organized=True,
    )

    coding_bias = task.coding_bias
    assert coding_bias.shape == (task.n,)
    assert np.count_nonzero(coding_bias < 0.0) > 0
    for assembly in task.class_assemblies.values():
        assert np.any(coding_bias[assembly.indices] < 0.0)
