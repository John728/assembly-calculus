from __future__ import annotations

import numpy as np
import pytest

from pyac.core.rng import make_rng
from pyac.tasks.mnist.extractors import (
    FeatureExtractor,
    LargeAreaExtractor,
    LinearExtractor,
    NonlinearExtractor,
    RandomAssemblyExtractor,
    SplitAssemblyExtractor,
)


def _sample_images_and_labels(n: int, seed: int = 123) -> tuple[list[np.ndarray], np.ndarray]:
    rng = make_rng(seed)
    images = [rng.random((28, 28), dtype=np.float64) for _ in range(n)]
    labels = np.array([i % 10 for i in range(n)], dtype=np.int64)
    return images, labels


def test_feature_extractor_is_abstract() -> None:
    assert hasattr(FeatureExtractor, "__abstractmethods__")
    assert {"fit", "transform", "n_features"}.issubset(FeatureExtractor.__abstractmethods__)


@pytest.mark.parametrize(
    ("extractor", "n_features"),
    [
        (LinearExtractor(m=500), 500),
        (NonlinearExtractor(m=200), 200),
        (
            LargeAreaExtractor(m=120, beta=0.5, t_internal=2, n_examples_per_class=1),
            120,
        ),
        (
            RandomAssemblyExtractor(m=300, beta=0.5, t_internal=2, n_examples_per_class=1),
            300,
        ),
        (
            SplitAssemblyExtractor(m=200, beta=0.5, t_internal=2, n_examples_per_class=1),
            200,
        ),
    ],
)
def test_transform_requires_fit(extractor: FeatureExtractor, n_features: int) -> None:
    images, _ = _sample_images_and_labels(2)
    with pytest.raises(RuntimeError, match="fit"):
        _ = extractor.transform(images, make_rng(42))
    assert extractor.n_features == n_features


def test_linear_extractor_shape() -> None:
    rng = make_rng(42)
    ext = LinearExtractor(m=500)
    images, labels = _sample_images_and_labels(10)

    ext.fit(images, labels, rng)
    features = ext.transform(images, rng)

    assert features.shape == (10, 500)
    assert ext.n_features == 500


def test_nonlinear_extractor_binary() -> None:
    rng = make_rng(42)
    ext = NonlinearExtractor(m=200)
    images, labels = _sample_images_and_labels(10)

    ext.fit(images, labels, rng)
    features = ext.transform(images, rng)

    assert features.shape == (10, 200)
    unique_vals = np.unique(features)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


def test_large_area_extractor_shape_and_binary() -> None:
    rng = make_rng(42)
    ext = LargeAreaExtractor(m=100, beta=0.5, t_internal=3, n_examples_per_class=1)
    images, labels = _sample_images_and_labels(20)

    ext.fit(images, labels, rng)
    features = ext.transform(images, rng)

    assert features.shape == (20, 100)
    unique_vals = np.unique(features)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


def test_random_assembly_extractor_shape_and_binary() -> None:
    rng = make_rng(42)
    ext = RandomAssemblyExtractor(m=300, beta=0.5, t_internal=3, n_examples_per_class=1)
    images, labels = _sample_images_and_labels(20)

    ext.fit(images, labels, rng)
    features = ext.transform(images, rng)

    assert features.shape == (20, 300)
    unique_vals = np.unique(features)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


def test_split_assembly_extractor_shape_and_binary() -> None:
    rng = make_rng(42)
    m = 1000
    ext = SplitAssemblyExtractor(m=m, beta=0.5, t_internal=3, n_examples_per_class=2)
    images, labels = _sample_images_and_labels(20)

    ext.fit(images, labels, rng)
    features = ext.transform(images, rng)

    assert features.shape == (20, m)
    unique_vals = np.unique(features)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


def test_linear_and_nonlinear_deterministic_with_same_seed() -> None:
    images, labels = _sample_images_and_labels(5, seed=99)

    linear_a = LinearExtractor(m=100)
    linear_a.fit(images, labels, make_rng(42))
    f_linear_a = linear_a.transform(images, make_rng(42))

    linear_b = LinearExtractor(m=100)
    linear_b.fit(images, labels, make_rng(42))
    f_linear_b = linear_b.transform(images, make_rng(42))

    assert np.array_equal(f_linear_a, f_linear_b)

    nonlinear_a = NonlinearExtractor(m=100)
    nonlinear_a.fit(images, labels, make_rng(42))
    f_nonlinear_a = nonlinear_a.transform(images, make_rng(42))

    nonlinear_b = NonlinearExtractor(m=100)
    nonlinear_b.fit(images, labels, make_rng(42))
    f_nonlinear_b = nonlinear_b.transform(images, make_rng(42))

    assert np.array_equal(f_nonlinear_a, f_nonlinear_b)


def test_assembly_extractors_deterministic_with_same_seed() -> None:
    images, labels = _sample_images_and_labels(20, seed=777)

    large_a = LargeAreaExtractor(m=100, beta=0.5, t_internal=2, n_examples_per_class=1)
    large_a.fit(images, labels, make_rng(42))
    f_large_a = large_a.transform(images, make_rng(42))

    large_b = LargeAreaExtractor(m=100, beta=0.5, t_internal=2, n_examples_per_class=1)
    large_b.fit(images, labels, make_rng(42))
    f_large_b = large_b.transform(images, make_rng(42))

    assert np.array_equal(f_large_a, f_large_b)

    random_a = RandomAssemblyExtractor(m=300, beta=0.5, t_internal=2, n_examples_per_class=1)
    random_a.fit(images, labels, make_rng(42))
    f_random_a = random_a.transform(images, make_rng(42))

    random_b = RandomAssemblyExtractor(m=300, beta=0.5, t_internal=2, n_examples_per_class=1)
    random_b.fit(images, labels, make_rng(42))
    f_random_b = random_b.transform(images, make_rng(42))

    assert np.array_equal(f_random_a, f_random_b)

    split_a = SplitAssemblyExtractor(m=200, beta=0.5, t_internal=2, n_examples_per_class=1)
    split_a.fit(images, labels, make_rng(42))
    f_split_a = split_a.transform(images, make_rng(42))

    split_b = SplitAssemblyExtractor(m=200, beta=0.5, t_internal=2, n_examples_per_class=1)
    split_b.fit(images, labels, make_rng(42))
    f_split_b = split_b.transform(images, make_rng(42))

    assert np.array_equal(f_split_a, f_split_b)


def test_random_assembly_requires_feature_count_multiple_of_100() -> None:
    with pytest.raises(ValueError, match="multiple of 100"):
        RandomAssemblyExtractor(m=250)


def test_split_assembly_requires_feature_count_multiple_of_10() -> None:
    with pytest.raises(ValueError, match="multiple of 10"):
        SplitAssemblyExtractor(m=205)
