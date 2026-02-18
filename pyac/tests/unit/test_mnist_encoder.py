"""
Test suite for MNIST encoder module (TDD approach).

Tests cover:
- MNISTEncoder ABC structure and abstract methods
- COLT2022Encoder initialization and projection matrix creation
- encode() method output validation (shape, dtype, uniqueness, range)
- Determinism with same RNG seed
- Normalization behavior
- Edge cases (all zeros, single pixel, etc.)
"""

import pytest
import numpy as np
from pyac.core.rng import make_rng
from pyac.tasks.mnist.encoders import MNISTEncoder, COLT2022Encoder


class TestMNISTEncoderABC:
    """Test MNISTEncoder abstract base class."""

    def test_abc_cannot_instantiate(self):
        """MNISTEncoder is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            MNISTEncoder()

    def test_abc_has_encode_method(self):
        """MNISTEncoder has abstract encode method."""
        assert hasattr(MNISTEncoder, "encode")


class TestCOLT2022EncoderInit:
    """Test COLT2022Encoder initialization."""

    def test_init_stores_params(self):
        """Encoder stores n_neurons and k_stimulus parameters."""
        rng = make_rng(42)
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=rng)
        assert enc.n_neurons == 100
        assert enc.k_stimulus == 10

    def test_init_creates_projection_matrix(self):
        """Encoder creates projection matrix on initialization."""
        rng = make_rng(42)
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=rng)
        assert hasattr(enc, "projection_matrix")
        assert enc.projection_matrix.shape == (100, 784)
        assert enc.projection_matrix.dtype == np.float64

    def test_init_projection_deterministic(self):
        """Same seed produces identical projection matrix."""
        enc1 = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        enc2 = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        assert np.allclose(enc1.projection_matrix, enc2.projection_matrix)

    def test_init_validates_k_le_n(self):
        """Raises ValueError if k_stimulus > n_neurons."""
        with pytest.raises(ValueError, match="k_stimulus.*n_neurons"):
            COLT2022Encoder(n_neurons=10, k_stimulus=20, rng=make_rng(42))

    def test_init_validates_positive_n(self):
        """Raises ValueError if n_neurons <= 0."""
        with pytest.raises(ValueError, match="n_neurons"):
            COLT2022Encoder(n_neurons=0, k_stimulus=5, rng=make_rng(42))

    def test_init_validates_positive_k(self):
        """Raises ValueError if k_stimulus <= 0."""
        with pytest.raises(ValueError, match="k_stimulus"):
            COLT2022Encoder(n_neurons=100, k_stimulus=0, rng=make_rng(42))


class TestCOLT2022EncoderEncode:
    """Test COLT2022Encoder encode method."""

    def test_encode_returns_ndarray(self):
        """encode() returns numpy ndarray."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert isinstance(result, np.ndarray)

    def test_encode_returns_k_indices(self):
        """encode() returns exactly k_stimulus indices."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert len(result) == 10

    def test_encode_returns_int64(self):
        """encode() returns int64 array."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert result.dtype == np.int64

    def test_encode_indices_in_range(self):
        """encode() returns indices in [0, n_neurons)."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert np.all(result >= 0)
        assert np.all(result < 100)

    def test_encode_indices_unique(self):
        """encode() returns unique indices."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert len(set(result)) == 10

    def test_encode_indices_sorted(self):
        """encode() returns sorted indices (Assembly convention)."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert np.all(result[:-1] <= result[1:])

    def test_encode_deterministic(self):
        """Same image + same seed produces identical indices."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result1 = enc.encode(image, make_rng(7))
        result2 = enc.encode(image, make_rng(7))
        assert np.array_equal(result1, result2)

    def test_encode_different_images_different_indices(self):
        """Different images produce different indices (probabilistically)."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image1 = np.random.default_rng(99).random((28, 28))
        image2 = np.random.default_rng(77).random((28, 28))
        result1 = enc.encode(image1, make_rng(7))
        result2 = enc.encode(image2, make_rng(7))
        # At least some indices should differ (probabilistic, but very likely)
        assert not np.array_equal(result1, result2)

    def test_encode_validates_image_shape(self):
        """Raises ValueError if image is not (28, 28)."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        invalid_image = np.random.default_rng(99).random((20, 20))
        with pytest.raises(ValueError, match="28.*28"):
            enc.encode(invalid_image, make_rng(7))

    def test_encode_handles_zero_image(self):
        """encode() handles all-zero image without error."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        zero_image = np.zeros((28, 28), dtype=np.float64)
        result = enc.encode(zero_image, make_rng(7))
        # Should still return k_stimulus indices
        assert len(result) == 10
        assert result.dtype == np.int64

    def test_encode_handles_single_pixel(self):
        """encode() handles single bright pixel."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        single_pixel = np.zeros((28, 28), dtype=np.float64)
        single_pixel[14, 14] = 1.0
        result = enc.encode(single_pixel, make_rng(7))
        assert len(result) == 10
        assert result.dtype == np.int64

    def test_encode_normalization_invariant(self):
        """Scaled images produce same indices (normalized projection)."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=make_rng(42))
        image1 = np.random.default_rng(99).random((28, 28))
        image2 = image1 * 2.0  # Scale by 2
        result1 = enc.encode(image1, make_rng(7))
        result2 = enc.encode(image2, make_rng(7))
        # Normalized projection should give same top-k indices
        assert np.array_equal(result1, result2)


class TestCOLT2022EncoderEdgeCases:
    """Test edge cases for COLT2022Encoder."""

    def test_k_equals_n(self):
        """Encoder works when k_stimulus == n_neurons (all neurons active)."""
        enc = COLT2022Encoder(n_neurons=20, k_stimulus=20, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert len(result) == 20
        assert set(result) == set(range(20))  # All neurons present

    def test_k_equals_1(self):
        """Encoder works with k_stimulus=1 (single neuron)."""
        enc = COLT2022Encoder(n_neurons=100, k_stimulus=1, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert len(result) == 1
        assert 0 <= result[0] < 100

    def test_small_network(self):
        """Encoder works with small n_neurons."""
        enc = COLT2022Encoder(n_neurons=10, k_stimulus=3, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert len(result) == 3
        assert np.all(result >= 0)
        assert np.all(result < 10)

    def test_large_k(self):
        """Encoder works with large k_stimulus (high sparsity)."""
        enc = COLT2022Encoder(n_neurons=1000, k_stimulus=500, rng=make_rng(42))
        image = np.random.default_rng(99).random((28, 28))
        result = enc.encode(image, make_rng(7))
        assert len(result) == 500
        assert len(set(result)) == 500  # All unique


class TestCOLT2022EncoderProjection:
    """Test projection matrix properties."""

    def test_projection_matrix_not_zero(self):
        """Projection matrix is not all zeros."""
        enc = COLT2022Encoder(n_neurons=50, k_stimulus=5, rng=make_rng(42))
        assert not np.allclose(enc.projection_matrix, 0.0)

    def test_projection_matrix_seeded(self):
        """Projection matrix uses init-time RNG, not encode-time RNG."""
        # Same init seed, different encode seeds → same matrix
        enc1 = COLT2022Encoder(n_neurons=50, k_stimulus=5, rng=make_rng(123))
        enc2 = COLT2022Encoder(n_neurons=50, k_stimulus=5, rng=make_rng(123))
        assert np.allclose(enc1.projection_matrix, enc2.projection_matrix)

        # Different init seed → different matrix
        enc3 = COLT2022Encoder(n_neurons=50, k_stimulus=5, rng=make_rng(999))
        assert not np.allclose(enc1.projection_matrix, enc3.projection_matrix)

    def test_projection_independence_from_encode_rng(self):
        """encode() RNG does not affect projection matrix."""
        enc = COLT2022Encoder(n_neurons=50, k_stimulus=5, rng=make_rng(42))
        original_matrix = enc.projection_matrix.copy()

        # Encode multiple times with different RNGs
        image = np.random.default_rng(99).random((28, 28))
        enc.encode(image, make_rng(1))
        enc.encode(image, make_rng(2))
        enc.encode(image, make_rng(3))

        # Matrix should not change
        assert np.allclose(enc.projection_matrix, original_matrix)
