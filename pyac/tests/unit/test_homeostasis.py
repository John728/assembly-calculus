"""
Tests for homeostasis/normalization module.

Task 11: Homeostasis Module
Tests the Network.normalize() wrapper around normalize_weights.
"""

from __future__ import annotations

import importlib
import numpy as np
from scipy.sparse import csr_matrix

from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec


network_mod = importlib.import_module("pyac.core.network")
Network = network_mod.Network


def _three_area_spec() -> NetworkSpec:
    """Network spec with three areas: A -> B -> C."""
    return NetworkSpec(
        areas=[
            AreaSpec("A", n=20, k=2, dynamics_type="feedforward"),
            AreaSpec("B", n=20, k=2, dynamics_type="feedforward"),
            AreaSpec("C", n=20, k=2, dynamics_type="feedforward"),
        ],
        fibers=[
            FiberSpec("A", "B", 0.3),
            FiberSpec("B", "C", 0.3),
        ],
        beta=0.1,
    )


def _get_column_sums(weights_dict: dict, area_name: str, n_target: int) -> np.ndarray:
    """
    Compute total column sums across ALL incoming fibers to an area.
    
    Returns array of length n_target where each element is the sum of incoming
    weights to that target neuron from all sources.
    """
    col_sums = np.zeros(n_target, dtype=np.float64)
    for (src, dst), matrix in weights_dict.items():
        if dst == area_name:
            col_sums += np.asarray(matrix.sum(axis=0)).ravel()
    return col_sums


class TestNormalizeMethod:
    """Test Network.normalize() method exists and is callable."""
    
    def test_normalize_method_exists(self):
        """Network should have normalize method."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(123))
        assert hasattr(net, 'normalize')
        assert callable(net.normalize)
    
    def test_normalize_callable_with_no_args(self):
        """normalize() should be callable with no arguments."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(123))
        # Should not raise
        net.normalize()
    
    def test_normalize_callable_with_area_name(self):
        """normalize(area_name) should be callable with area name."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(123))
        # Should not raise
        net.normalize("B")


class TestColumnSumNormalization:
    """Test that normalize() produces column sums of 1.0 for non-zero columns."""
    
    def test_normalize_all_areas_makes_column_sums_one(self):
        """After normalize(), all non-zero columns should sum to 1.0."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(456))
        
        # Perturb weights to break normalization
        net.weights[("A", "B")].data *= 2.5
        
        net.normalize()
        
        # Check B's incoming columns sum to 1.0
        col_sums_b = _get_column_sums(net.weights, "B", 20)
        nonzero_cols = col_sums_b > 0
        assert np.allclose(col_sums_b[nonzero_cols], 1.0, atol=1e-14)
        
        # Check C's incoming columns sum to 1.0
        col_sums_c = _get_column_sums(net.weights, "C", 20)
        nonzero_cols = col_sums_c > 0
        assert np.allclose(col_sums_c[nonzero_cols], 1.0, atol=1e-14)
    
    def test_normalize_single_area_only(self):
        """normalize(area_name) should only normalize that area's incoming weights."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(789))
        
        # Perturb both matrices
        net.weights[("A", "B")].data *= 3.0
        net.weights[("B", "C")].data *= 3.0
        
        # Only normalize B
        net.normalize("B")
        
        # B's incoming columns should be normalized
        col_sums_b = _get_column_sums(net.weights, "B", 20)
        nonzero_cols = col_sums_b > 0
        assert np.allclose(col_sums_b[nonzero_cols], 1.0, atol=1e-14)
        
        # C's incoming columns should still be perturbed
        col_sums_c = _get_column_sums(net.weights, "C", 20)
        nonzero_cols = col_sums_c > 0
        assert not np.allclose(col_sums_c[nonzero_cols], 1.0, atol=1e-14)
    
    def test_zero_columns_remain_zero(self):
        """Disconnected neurons (zero columns) should remain zero after normalization."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(111))
        
        # Get initial zero columns
        col_sums_before = _get_column_sums(net.weights, "B", 20)
        zero_cols_before = np.where(col_sums_before == 0)[0]
        
        # Normalize
        net.normalize("B")
        
        # Check zero columns stayed zero
        col_sums_after = _get_column_sums(net.weights, "B", 20)
        zero_cols_after = np.where(col_sums_after == 0)[0]
        
        assert np.array_equal(zero_cols_before, zero_cols_after)


class TestIdempotency:
    """Test that normalize() is idempotent: calling twice = same result as once."""
    
    def test_normalize_idempotent_all_areas(self):
        """Calling normalize() twice should produce identical weights."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(222))
        
        # Perturb weights
        net.weights[("A", "B")].data *= 2.7
        net.weights[("B", "C")].data *= 1.3
        
        # Normalize once and capture state
        net.normalize()
        weights_after_once = {k: v.copy() for k, v in net.weights.items()}
        
        # Normalize again
        net.normalize()
        weights_after_twice = net.weights
        
        # Should be identical
        for key in weights_after_once.keys():
            diff = (weights_after_once[key] - weights_after_twice[key]).data
            assert np.allclose(diff, 0, atol=1e-14)
    
    def test_normalize_idempotent_single_area(self):
        """Calling normalize(area_name) twice should produce identical weights."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(333))
        
        # Perturb only B's incoming weights
        net.weights[("A", "B")].data *= 4.2
        
        # Normalize once and capture state
        net.normalize("B")
        weights_after_once = {k: v.copy() for k, v in net.weights.items()}
        
        # Normalize again
        net.normalize("B")
        weights_after_twice = net.weights
        
        # Should be identical
        for key in weights_after_once.keys():
            diff = (weights_after_once[key] - weights_after_twice[key]).data
            assert np.allclose(diff, 0, atol=1e-14)


class TestSparsityPreservation:
    """Test that normalize() doesn't create new nonzeros."""
    
    def test_normalize_preserves_sparsity_all_areas(self):
        """normalize() should never create new nonzeros in the sparsity pattern."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(444))
        
        # Count nonzeros before
        nnz_before = {k: v.nnz for k, v in net.weights.items()}
        
        # Perturb and normalize
        net.weights[("A", "B")].data *= 1.5
        net.weights[("B", "C")].data *= 2.2
        net.normalize()
        
        # Count nonzeros after
        nnz_after = {k: v.nnz for k, v in net.weights.items()}
        
        # Should be identical (no new nonzeros)
        assert nnz_before == nnz_after
    
    def test_normalize_preserves_sparsity_single_area(self):
        """normalize(area_name) should never create new nonzeros."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(555))
        
        # Count nonzeros before
        nnz_before = {k: v.nnz for k, v in net.weights.items()}
        
        # Perturb and normalize B only
        net.weights[("A", "B")].data *= 3.3
        net.normalize("B")
        
        # Count nonzeros after
        nnz_after = {k: v.nnz for k, v in net.weights.items()}
        
        # Should be identical
        assert nnz_before == nnz_after


class TestMultipleIncomingFibers:
    """Test normalization with multiple incoming fibers to same area."""
    
    def test_normalize_with_multiple_incoming_fibers(self):
        """normalize() should handle multiple incoming fibers (sum columns across all sources)."""
        spec = NetworkSpec(
            areas=[
                AreaSpec("A", n=10, k=2, dynamics_type="feedforward"),
                AreaSpec("B", n=10, k=2, dynamics_type="feedforward"),
                AreaSpec("C", n=10, k=2, dynamics_type="feedforward"),
            ],
            fibers=[
                FiberSpec("A", "C", 0.3),  # Two sources to C
                FiberSpec("B", "C", 0.3),
            ],
            beta=0.1,
        )
        net = Network(spec, make_rng(666))
        
        # Perturb both incoming fibers
        net.weights[("A", "C")].data *= 2.0
        net.weights[("B", "C")].data *= 1.5
        
        # Normalize
        net.normalize("C")
        
        # C's column sums should be 1.0 (aggregated from both A and B)
        col_sums_c = _get_column_sums(net.weights, "C", 10)
        nonzero_cols = col_sums_c > 0
        assert np.allclose(col_sums_c[nonzero_cols], 1.0, atol=1e-14)


class TestInitialState:
    """Test that network starts with normalized weights from constructor."""
    
    def test_network_starts_normalized(self):
        """Network should start with normalized weights after construction."""
        spec = _three_area_spec()
        net = Network(spec, make_rng(777))
        
        # Check B is normalized
        col_sums_b = _get_column_sums(net.weights, "B", 20)
        nonzero_cols = col_sums_b > 0
        assert np.allclose(col_sums_b[nonzero_cols], 1.0, atol=1e-14)
        
        # Check C is normalized
        col_sums_c = _get_column_sums(net.weights, "C", 20)
        nonzero_cols = col_sums_c > 0
        assert np.allclose(col_sums_c[nonzero_cols], 1.0, atol=1e-14)
