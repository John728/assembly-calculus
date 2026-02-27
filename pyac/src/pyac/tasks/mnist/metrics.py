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


def evaluate_softmax(
    outputs: np.ndarray,
    n_train_per_class: int,
    n_test_per_class: int,
    rng: Generator | None = None
) -> dict[str, float]:
    """
    Evaluate feature rollouts using legacy SGD softmax optimization.
    """
    from pyac.core.rng import make_rng
    effective_rng = make_rng(0) if rng is None else rng
    n_neurons = outputs.shape[-1]
    
    def softmax(x):
        x_shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    v = 0.1 * effective_rng.standard_normal((10, n_neurons))
    targets = np.zeros((100, 10))
    for i in range(10):
        targets[i*10:(i+1)*10, i] = 1
    update = np.zeros_like(v)
    
    for _ in range(100):
        permutation = effective_rng.permutation(n_train_per_class)
        for j in range(n_train_per_class // 10):
            batch = outputs[:, 1, permutation[j*10:(j+1)*10]].reshape(10 * 10, n_neurons)
            scores = softmax((batch[:, :, np.newaxis] * v.T[np.newaxis, :, :]).sum(axis=1))
            update = 0.5 * update + 1e-3 * (batch[:, np.newaxis, :] * (scores - targets)[:, :, np.newaxis]).sum(axis=0)
            v -= update
            
    train_correct = ((outputs[:, 1, :n_train_per_class] @ v.T).argmax(axis=-1) == np.arange(10)[:, np.newaxis]).sum()
    test_correct = ((outputs[:, 1, n_train_per_class:n_train_per_class + n_test_per_class] @ v.T).argmax(axis=-1) == np.arange(10)[:, np.newaxis]).sum()
    
    return {
        "train_accuracy": float(train_correct) / (10 * n_train_per_class),
        "test_accuracy": float(test_correct) / (10 * n_test_per_class)
    }


def evaluate_voting(
    outputs: np.ndarray,
    cap_size: int,
    n_train_per_class: int,
) -> dict[str, float]:
    """
    Evaluate feature rollouts by computing class-wise prototypes (c) 
    and returning the most active intersection count.
    """
    n_neurons = outputs.shape[-1]
    n_test_per_class = outputs.shape[2] - n_train_per_class
    c = np.zeros((10, n_neurons))
    for i in range(10):
        train_outputs = outputs[i, 1, :n_train_per_class]
        c[i, train_outputs.sum(axis=0).argsort()[-cap_size:]] = 1
        
    predictions_train = (outputs[:, 1, :n_train_per_class] @ c.T).argmax(axis=-1)
    train_acc = (predictions_train == np.arange(10)[:, np.newaxis]).sum() / (10 * n_train_per_class)
    
    if n_test_per_class > 0:
        predictions_test = (outputs[:, 1, n_train_per_class:] @ c.T).argmax(axis=-1)
        test_acc = (predictions_test == np.arange(10)[:, np.newaxis]).sum() / (10 * n_test_per_class)
    else:
        test_acc = 0.0
    
    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc)
    }

