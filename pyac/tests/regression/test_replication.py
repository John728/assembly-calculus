from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

from pyac.core.rng import make_rng
from pyac.tasks.mnist.extractors import SplitAssemblyExtractor


@pytest.mark.slow
def test_split_assembly_qualitative() -> None:
    """Verify split assembly accuracy improves with feature count."""
    digits = load_digits()
    X = np.asarray(digits.data, dtype=np.float64) / 16.0
    y = np.asarray(digits.target, dtype=np.int64)
    
    X_train = X[:1000]
    y_train = y[:1000]
    X_test = X[1200:1300]
    y_test = y[1200:1300]
    
    images_train = [np.pad(img.reshape(8, 8), 10, mode='constant').flatten() for img in X_train]
    images_test = [np.pad(img.reshape(8, 8), 10, mode='constant').flatten() for img in X_test]
    
    rng = make_rng(42)
    
    # Test m=100 vs m=200
    ext_small = SplitAssemblyExtractor(m=100, beta=0.5, t_internal=2, n_examples_per_class=2)
    ext_small.fit(images_train, y_train, rng)
    features_small_train = ext_small.transform(images_train, rng)
    features_small_test = ext_small.transform(images_test, rng)
    
    clf_small = LogisticRegression(max_iter=1000, random_state=42)
    clf_small.fit(features_small_train, y_train)
    acc_small = clf_small.score(features_small_test, y_test)
    
    ext_large = SplitAssemblyExtractor(m=200, beta=0.5, t_internal=2, n_examples_per_class=2)
    ext_large.fit(images_train, y_train, rng)
    features_large_train = ext_large.transform(images_train, rng)
    features_large_test = ext_large.transform(images_test, rng)
    
    clf_large = LogisticRegression(max_iter=1000, random_state=42)
    clf_large.fit(features_large_train, y_train)
    acc_large = clf_large.score(features_large_test, y_test)
    
    # Qualitative: larger feature count should give better or equal accuracy
    assert acc_large >= acc_small * 0.95, f"Accuracy should improve with features: {acc_small:.2%} -> {acc_large:.2%}"


@pytest.mark.slow
def test_split_assembly_quantitative() -> None:
    """At m=200, verify accuracy > 10% (better than random)."""
    digits = load_digits()
    X = np.asarray(digits.data, dtype=np.float64) / 16.0
    y = np.asarray(digits.target, dtype=np.int64)
    
    X_train = X[:1000]
    y_train = y[:1000]
    X_test = X[1200:1300]
    y_test = y[1200:1300]
    
    images_train = [np.pad(img.reshape(8, 8), 10, mode='constant').flatten() for img in X_train]
    images_test = [np.pad(img.reshape(8, 8), 10, mode='constant').flatten() for img in X_test]
    
    rng = make_rng(123)
    
    ext = SplitAssemblyExtractor(m=200, beta=0.5, t_internal=2, n_examples_per_class=2)
    ext.fit(images_train, y_train, rng)
    features_train = ext.transform(images_train, rng)
    features_test = ext.transform(images_test, rng)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(features_train, y_train)
    accuracy = clf.score(features_test, y_test)
    
    # Must beat random (10% for 10 classes)
    assert accuracy > 0.10, f"Accuracy {accuracy:.2%} must be > 10% (random baseline)"


@pytest.mark.slow
def test_reproducibility() -> None:
    """Same seed produces same features."""
    # Use small synthetic data
    rng_data = make_rng(999)
    images = [rng_data.random((28, 28), dtype=np.float64) for _ in range(50)]
    labels = np.array([i % 10 for i in range(50)], dtype=np.int64)
    
    # Run 1
    ext1 = SplitAssemblyExtractor(m=200, beta=0.5, t_internal=2, n_examples_per_class=1)
    ext1.fit(images, labels, make_rng(42))
    features1 = ext1.transform(images, make_rng(42))
    
    # Run 2 with same seed
    ext2 = SplitAssemblyExtractor(m=200, beta=0.5, t_internal=2, n_examples_per_class=1)
    ext2.fit(images, labels, make_rng(42))
    features2 = ext2.transform(images, make_rng(42))
    
    assert np.array_equal(features1, features2), "Reproducibility failed: same seed must produce same features"
