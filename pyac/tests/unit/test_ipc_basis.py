"""
TDD tests for IPC basis functions (Legendre polynomials).

Tests legendre polynomial basis functions, input signal generation,
and target function generation for IPC measurement.
"""

import numpy as np
import pytest
from scipy.special import eval_legendre as scipy_eval_legendre

from pyac.measures.ipc.basis import (
    legendre_basis,
    generate_input_signal,
    generate_targets,
)
from pyac.core.rng import make_rng


class TestLegendBasis:
    """Test legendre_basis(degree, x) function."""

    def test_legendre_basis_scalar_input(self):
        """Test evaluation at single point."""
        result = legendre_basis(0, np.array([0.0]))
        assert result.shape == (1,)
        assert np.isclose(result[0], 1.0)

    def test_legendre_basis_array_input(self):
        """Test evaluation at multiple points."""
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = legendre_basis(1, x)
        assert result.shape == (5,)
        # P_1(x) = x
        expected = x
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_legendre_basis_degree_zero(self):
        """Test P_0(x) = 1 for all x."""
        x = np.linspace(-1, 1, 11)
        result = legendre_basis(0, x)
        np.testing.assert_allclose(result, np.ones_like(x), atol=1e-14)

    def test_legendre_basis_degree_two(self):
        """Test P_2(x) = (3x^2 - 1) / 2."""
        x = np.array([-1.0, 0.0, 1.0])
        result = legendre_basis(2, x)
        # P_2(-1) = (3 - 1) / 2 = 1
        # P_2(0) = (0 - 1) / 2 = -0.5
        # P_2(1) = (3 - 1) / 2 = 1
        expected = np.array([1.0, -0.5, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_legendre_basis_matches_scipy(self):
        """Test that result matches scipy.special.eval_legendre."""
        x = np.linspace(-1, 1, 21)
        for degree in range(5):
            result = legendre_basis(degree, x)
            expected = scipy_eval_legendre(degree, x)
            np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_legendre_basis_float64_output(self):
        """Test output is float64."""
        result = legendre_basis(1, np.array([0.5]))
        assert result.dtype == np.float64

    def test_legendre_basis_domain_minus_one_to_one(self):
        """Test evaluation within [-1, 1] domain."""
        x = np.linspace(-1, 1, 51)
        for degree in range(3):
            result = legendre_basis(degree, x)
            # Should be finite
            assert np.all(np.isfinite(result))

    def test_legendre_basis_orthogonality_property(self):
        """Test orthogonality: integral P_m(x) P_n(x) dx ≈ 0 for m != n."""
        x = np.linspace(-1, 1, 1000)
        p1 = legendre_basis(1, x)
        p2 = legendre_basis(2, x)
        # Numerical integration via Riemann sum
        dx = x[1] - x[0]
        integral = np.sum(p1 * p2) * dx
        assert abs(integral) < 0.01  # Should be near zero


class TestGenerateInputSignal:
    """Test generate_input_signal(length, rng) function."""

    def test_generate_input_signal_deterministic(self):
        """Test determinism with fixed seed."""
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        signal1 = generate_input_signal(100, rng1)
        signal2 = generate_input_signal(100, rng2)
        np.testing.assert_array_equal(signal1, signal2)

    def test_generate_input_signal_length(self):
        """Test signal has correct length."""
        rng = make_rng(42)
        for length in [10, 50, 100, 1000]:
            signal = generate_input_signal(length, rng)
            assert signal.shape == (length,)

    def test_generate_input_signal_range(self):
        """Test signal values in [-1, 1]."""
        rng = make_rng(42)
        signal = generate_input_signal(1000, rng)
        assert np.all(signal >= -1.0)
        assert np.all(signal <= 1.0)

    def test_generate_input_signal_float64_output(self):
        """Test output is float64."""
        rng = make_rng(42)
        signal = generate_input_signal(10, rng)
        assert signal.dtype == np.float64

    def test_generate_input_signal_different_rng_different_result(self):
        """Test different RNG objects produce different signals."""
        rng1 = make_rng(42)
        rng2 = make_rng(43)
        signal1 = generate_input_signal(100, rng1)
        signal2 = generate_input_signal(100, rng2)
        assert not np.allclose(signal1, signal2)

    def test_generate_input_signal_independent_calls(self):
        """Test consecutive calls with same RNG produce different signals."""
        rng = make_rng(42)
        signal1 = generate_input_signal(10, rng)
        signal2 = generate_input_signal(10, rng)
        assert not np.allclose(signal1, signal2)

    def test_generate_input_signal_uses_uniform_distribution(self):
        """Test distribution is approximately uniform [-1, 1]."""
        rng = make_rng(42)
        signal = generate_input_signal(10000, rng)
        # Check mean is near 0 (center of [-1, 1])
        assert abs(np.mean(signal)) < 0.1
        # Check min and max are near boundaries
        assert np.min(signal) < -0.9
        assert np.max(signal) > 0.9


class TestGenerateTargets:
    """Test generate_targets(input_signal, max_degree, max_delay) function."""

    def test_generate_targets_returns_list(self):
        """Test return type is list."""
        rng = make_rng(42)
        signal = generate_input_signal(50, rng)
        targets = generate_targets(signal, max_degree=1, max_delay=1)
        assert isinstance(targets, list)

    def test_generate_targets_tuple_structure(self):
        """Test each target is (degrees_tuple, delays_tuple, target_array)."""
        rng = make_rng(42)
        signal = generate_input_signal(50, rng)
        targets = generate_targets(signal, max_degree=1, max_delay=1)
        assert len(targets) > 0
        for item in targets:
            assert isinstance(item, tuple)
            assert len(item) == 3
            degrees, delays, target_array = item
            assert isinstance(degrees, tuple)
            assert isinstance(delays, tuple)
            assert isinstance(target_array, np.ndarray)

    def test_generate_targets_non_empty(self):
        """Test at least one target is generated."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        targets = generate_targets(signal, max_degree=2, max_delay=2)
        assert len(targets) > 0

    def test_generate_targets_array_shape(self):
        """Test target array shape matches signal length (accounting for delays)."""
        rng = make_rng(42)
        signal = generate_input_signal(50, rng)
        max_degree = 2
        max_delay = 2
        targets = generate_targets(signal, max_degree=max_degree, max_delay=max_delay)
        for degrees, delays, target_array in targets:
            max_d = max(delays) if delays else 0
            # Target should be signal length minus max delay
            expected_len = len(signal) - max_d
            assert len(target_array) == expected_len

    def test_generate_targets_float64_output(self):
        """Test target arrays are float64."""
        rng = make_rng(42)
        signal = generate_input_signal(50, rng)
        targets = generate_targets(signal, max_degree=1, max_delay=1)
        for _, _, target_array in targets:
            assert target_array.dtype == np.float64

    def test_generate_targets_deterministic(self):
        """Test determinism with same signal."""
        rng = make_rng(42)
        signal = generate_input_signal(50, rng)
        targets1 = generate_targets(signal, max_degree=2, max_delay=2)
        targets2 = generate_targets(signal, max_degree=2, max_delay=2)
        assert len(targets1) == len(targets2)
        for (d1, t1, a1), (d2, t2, a2) in zip(targets1, targets2):
            assert d1 == d2
            assert t1 == t2
            np.testing.assert_array_equal(a1, a2)

    def test_generate_targets_includes_single_degree(self):
        """Test single-degree targets P_d(u_{t-τ})."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        targets = generate_targets(signal, max_degree=1, max_delay=1)
        # Should have targets for degree 1 with different delays
        single_degree_targets = [
            t for t in targets if len(t[0]) == 1
        ]
        assert len(single_degree_targets) > 0

    def test_generate_targets_includes_product_targets(self):
        """Test product targets with multiple degrees."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        targets = generate_targets(signal, max_degree=2, max_delay=2)
        # Should have some targets with multiple degrees (products)
        product_targets = [
            t for t in targets if len(t[0]) > 1
        ]
        # With max_degree=2, max_delay=2, should have products
        assert len(product_targets) > 0

    def test_generate_targets_degrees_within_range(self):
        """Test all degrees <= max_degree."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        max_degree = 3
        targets = generate_targets(signal, max_degree=max_degree, max_delay=2)
        for degrees, _, _ in targets:
            for degree in degrees:
                assert degree <= max_degree

    def test_generate_targets_delays_within_range(self):
        """Test all delays <= max_delay and > 0."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        max_delay = 3
        targets = generate_targets(signal, max_degree=2, max_delay=max_delay)
        for _, delays, _ in targets:
            for delay in delays:
                assert 0 < delay <= max_delay

    def test_generate_targets_product_evaluation(self):
        """Test product target is actual product of Legendre evaluations."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        targets = generate_targets(signal, max_degree=2, max_delay=2)
        # Find a product target with 2 degrees
        product_targets = [
            t for t in targets if len(t[0]) == 2
        ]
        assert len(product_targets) > 0
        degrees, delays, target_array = product_targets[0]
        # Manually compute product
        d1, d2 = degrees
        τ1, τ2 = delays
        u_d1 = signal[τ1:]
        u_d2 = signal[τ2:]
        # Need to align them
        max_delay = max(τ1, τ2)
        p_d1 = legendre_basis(d1, u_d1[:len(target_array)])
        p_d2 = legendre_basis(d2, u_d2[:len(target_array)])
        expected = p_d1 * p_d2
        np.testing.assert_allclose(target_array, expected, atol=1e-13)

    def test_generate_targets_max_degree_zero(self):
        """Test with max_degree=0 (only P_0)."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        targets = generate_targets(signal, max_degree=0, max_delay=2)
        # Should have some targets with degree 0
        degree_zero = [t for t in targets if all(d == 0 for d in t[0])]
        assert len(degree_zero) > 0

    def test_generate_targets_max_delay_zero_raises_or_empty(self):
        """Test max_delay=0 produces no targets or raises error."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        targets = generate_targets(signal, max_degree=1, max_delay=0)
        # Should be empty since delays must be > 0
        assert len(targets) == 0


class TestAcceptanceOrthogonality:
    """Acceptance test: Legendre polynomials are orthogonal."""

    def test_legendre_orthogonality_long_signal(self):
        """Test P_1 and P_2 have low correlation over long signal."""
        rng = make_rng(42)
        x = generate_input_signal(10000, rng)
        p1 = legendre_basis(1, x)
        p2 = legendre_basis(2, x)
        corr = np.corrcoef(p1, p2)[0, 1]
        assert abs(corr) < 0.05, f"P1 and P2 not orthogonal: corr={corr}"

    def test_legendre_orthogonality_multiple_pairs(self):
        """Test orthogonality for multiple polynomial pairs."""
        rng = make_rng(42)
        x = generate_input_signal(10000, rng)
        for d1 in range(1, 3):
            for d2 in range(d1 + 1, 3):
                p1 = legendre_basis(d1, x)
                p2 = legendre_basis(d2, x)
                corr = np.corrcoef(p1, p2)[0, 1]
                assert abs(corr) < 0.05, (
                    f"P{d1} and P{d2} not orthogonal: corr={corr}"
                )


class TestAcceptanceTargetGeneration:
    """Acceptance test: Target generation works correctly."""

    def test_target_generation_count(self):
        """Test reasonable number of targets generated."""
        rng = make_rng(42)
        signal = generate_input_signal(100, rng)
        targets = generate_targets(signal, max_degree=2, max_delay=2)
        # Should have at least 4 targets (degree 0-2 x delay 1-2)
        assert len(targets) >= 4

    def test_target_generation_short_signal(self):
        """Test targets truncated correctly for short signals."""
        rng = make_rng(42)
        signal = generate_input_signal(10, rng)
        targets = generate_targets(signal, max_degree=1, max_delay=5)
        # With max_delay=5 and signal length 10, targets should be length 5
        for _, delays, target_array in targets:
            if delays:
                assert len(target_array) == 10 - max(delays)
