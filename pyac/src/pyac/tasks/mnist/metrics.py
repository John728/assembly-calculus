from __future__ import annotations

import importlib
import numpy as np
from numpy.random import Generator

from pyac.core.network import Network
from pyac.core.rng import spawn_rngs
from pyac.core.types import NetworkSpec
from pyac.tasks.mnist.encoders import MNISTEncoder


def accuracy_vs_t(
    network_spec: NetworkSpec,
    encoder: MNISTEncoder,
    t_values: list[int],
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rng: Generator,
) -> dict[int, float]:
    protocol = importlib.import_module("pyac.tasks.mnist.protocol")

    x_train, y_train, x_test, y_test = data
    results: dict[int, float] = {}

    if "class" not in [area.name for area in network_spec.areas]:
        raise ValueError("network_spec must include area 'class'")

    for t_per_image in t_values:
        if t_per_image <= 0:
            raise ValueError("t_values must contain only positive integers")

        net_rng, train_rng, feat_train_rng, feat_test_rng = spawn_rngs(rng, 4)
        net = Network(network_spec, net_rng)

        assemblies = protocol.train_assemblies(
            network=net,
            area_name="class",
            images=x_train,
            labels=y_train,
            encoder=encoder,
            t_per_image=t_per_image,
            rng=train_rng,
        )
        feat_train = protocol.extract_features(
            network=net,
            area_name="class",
            images=x_train,
            encoder=encoder,
            assemblies=assemblies,
            rng=feat_train_rng,
        )
        feat_test = protocol.extract_features(
            network=net,
            area_name="class",
            images=x_test,
            encoder=encoder,
            assemblies=assemblies,
            rng=feat_test_rng,
        )
        result = protocol.classify(feat_train, y_train, feat_test, y_test)
        results[t_per_image] = float(result["test_accuracy"])

    return results
