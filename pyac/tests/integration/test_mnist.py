from __future__ import annotations

import importlib
import numpy as np
import pytest
from sklearn.datasets import fetch_openml, load_digits

from pyac.core.network import Network
from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, Assembly, FiberSpec, NetworkSpec
from pyac.tasks.mnist.encoders import COLT2022Encoder

_protocol = importlib.import_module("pyac.tasks.mnist.protocol")
classify = _protocol.classify
extract_features = _protocol.extract_features
train_assemblies = _protocol.train_assemblies


def _build_spec() -> NetworkSpec:
    return NetworkSpec(
        areas=[AreaSpec("input", n=200, k=20), AreaSpec("class", n=200, k=20)],
        fibers=[FiberSpec("input", "class", 0.1)],
        beta=0.5,
    )


def _build_balanced_tiny_subset(
    x: np.ndarray,
    y: np.ndarray,
    train_per_class: int = 10,
    test_per_class: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in range(10):
        label_indices = np.flatnonzero(y == label)
        needed = train_per_class + test_per_class
        if label_indices.size < needed:
            raise ValueError(f"not enough samples for class {label}")
        train_indices.extend(label_indices[:train_per_class].tolist())
        test_indices.extend(label_indices[train_per_class:needed].tolist())

    x_train = x[np.array(train_indices, dtype=np.int64)]
    y_train = y[np.array(train_indices, dtype=np.int64)]
    x_test = x[np.array(test_indices, dtype=np.int64)]
    y_test = y[np.array(test_indices, dtype=np.int64)]
    return x_train, y_train, x_test, y_test


def _load_mnist_like() -> tuple[np.ndarray, np.ndarray]:
    try:
        x_raw, y_raw = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="auto",
            return_X_y=True,
        )
        return np.asarray(x_raw, dtype=np.float64) / 255.0, np.asarray(y_raw, dtype=np.int64)
    except Exception:
        pass

    try:
        x_raw, y_raw = fetch_openml(
            "mnist_784",
            as_frame=False,
            parser="auto",
            return_X_y=True,
        )
        return np.asarray(x_raw, dtype=np.float64) / 255.0, np.asarray(y_raw, dtype=np.int64)
    except Exception:
        x_digits, y_digits = load_digits(return_X_y=True)
        x_small = np.asarray(x_digits, dtype=np.float64) / 16.0
        y_small = np.asarray(y_digits, dtype=np.int64)
        x_small = x_small.reshape(-1, 8, 8)
        x_padded = np.pad(x_small, ((0, 0), (10, 10), (10, 10)), mode="constant")
        return x_padded.reshape(-1, 28 * 28), y_small


@pytest.fixture(scope="module")
def tiny_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_all, y_all = _load_mnist_like()
    return _build_balanced_tiny_subset(x_all, y_all, train_per_class=10, test_per_class=2)


def _make_pipeline_state(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[dict[int, Assembly], np.ndarray, np.ndarray]:
    net = Network(_build_spec(), make_rng(42))
    enc = COLT2022Encoder(n_neurons=200, k_stimulus=20, rng=make_rng(99))

    assemblies = train_assemblies(
        net,
        "class",
        x_train.reshape(-1, 28, 28),
        y_train,
        enc,
        t_per_image=2,
        rng=make_rng(77),
    )

    features_train = extract_features(
        net,
        "class",
        x_train.reshape(-1, 28, 28),
        enc,
        assemblies,
        make_rng(55),
    )
    features_test = extract_features(
        net,
        "class",
        x_test.reshape(-1, 28, 28),
        enc,
        assemblies,
        make_rng(66),
    )

    return assemblies, features_train, features_test


def test_train_assemblies_returns_dict(
    tiny_mnist_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x_train, y_train, _, _ = tiny_mnist_data
    net = Network(_build_spec(), make_rng(12))
    enc = COLT2022Encoder(n_neurons=200, k_stimulus=20, rng=make_rng(13))

    assemblies = train_assemblies(
        net,
        "class",
        x_train.reshape(-1, 28, 28),
        y_train,
        enc,
        t_per_image=1,
        rng=make_rng(14),
    )

    assert isinstance(assemblies, dict)
    assert set(assemblies.keys()) == set(range(10))
    for label in range(10):
        assert assemblies[label].area_name == "class"
        assert assemblies[label].indices.shape == (20,)


def test_train_assemblies_deterministic(
    tiny_mnist_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x_train, y_train, _, _ = tiny_mnist_data

    net_a = Network(_build_spec(), make_rng(123))
    net_b = Network(_build_spec(), make_rng(123))
    enc_a = COLT2022Encoder(n_neurons=200, k_stimulus=20, rng=make_rng(999))
    enc_b = COLT2022Encoder(n_neurons=200, k_stimulus=20, rng=make_rng(999))

    asm_a = train_assemblies(
        net_a,
        "class",
        x_train.reshape(-1, 28, 28),
        y_train,
        enc_a,
        t_per_image=2,
        rng=make_rng(222),
    )
    asm_b = train_assemblies(
        net_b,
        "class",
        x_train.reshape(-1, 28, 28),
        y_train,
        enc_b,
        t_per_image=2,
        rng=make_rng(222),
    )

    for label in range(10):
        assert np.array_equal(asm_a[label].indices, asm_b[label].indices)


def test_extract_features_shape(
    tiny_mnist_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x_train, y_train, x_test, _ = tiny_mnist_data
    assemblies, _, features_test = _make_pipeline_state(x_train, y_train, x_test)

    assert set(assemblies.keys()) == set(range(10))
    assert features_test.shape == (x_test.shape[0], 10)
    assert features_test.dtype == np.float64


def test_classify_above_chance(
    tiny_mnist_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x_train, y_train, x_test, y_test = tiny_mnist_data
    _, features_train, features_test = _make_pipeline_state(x_train, y_train, x_test)

    result = classify(features_train, y_train, features_test, y_test)

    assert result["n_train"] == int(y_train.shape[0])
    assert result["n_test"] == int(y_test.shape[0])
    assert result["test_accuracy"] > 0.15


def test_full_pipeline_deterministic(
    tiny_mnist_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x_train, y_train, x_test, y_test = tiny_mnist_data

    _, features_train_a, features_test_a = _make_pipeline_state(x_train, y_train, x_test)
    _, features_train_b, features_test_b = _make_pipeline_state(x_train, y_train, x_test)
    result_a = classify(features_train_a, y_train, features_test_a, y_test)
    result_b = classify(features_train_b, y_train, features_test_b, y_test)

    assert np.allclose(features_train_a, features_train_b)
    assert np.allclose(features_test_a, features_test_b)
    assert result_a["n_train"] == result_b["n_train"]
    assert result_a["n_test"] == result_b["n_test"]
    assert np.isclose(result_a["train_accuracy"], result_b["train_accuracy"])
    assert np.isclose(result_a["test_accuracy"], result_b["test_accuracy"])
