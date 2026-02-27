"""
TDD tests for weights.py module: init_weights and normalize_weights.
"""
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr

from pyac.core.weights import init_weights, normalize_weights
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec


class TestInitWeights:
    """Tests for init_weights function."""

    def test_init_weights_preserves_connectivity_structure(self):
        """init_weights should copy connectivity structure with all values = 1.0."""
        # Create simple connectivity dict
        conn_data = np.array([1, 1, 1])
        conn_indices = np.array([0, 1, 2])
        conn_indptr = np.array([0, 2, 3])
        conn = csr_matrix((conn_data, conn_indices, conn_indptr), shape=(2, 3), dtype=np.float64)
        
        connectivity = {('A', 'B'): conn}
        
        weights = init_weights(connectivity, mode='ones')
        
        # Check structure preserved
        assert ('A', 'B') in weights
        assert isspmatrix_csr(weights[('A', 'B')])
        assert weights[('A', 'B')].shape == conn.shape
        assert weights[('A', 'B')].nnz == conn.nnz
        assert weights[('A', 'B')].dtype == np.float64

    def test_init_weights_sets_all_values_to_one(self):
        """All non-zero values should be 1.0."""
        conn = csr_matrix([[1, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
        connectivity = {('A', 'B'): conn}
        
        weights = init_weights(connectivity, mode='ones')
        
        assert np.all(weights[('A', 'B')].data == 1.0)
        assert weights[('A', 'B')].nnz == 3

    def test_init_weights_does_not_modify_original_connectivity(self):
        """Original connectivity dict should not be modified."""
        conn = csr_matrix([[1, 0], [0, 1]], dtype=np.float64)
        connectivity = {('A', 'B'): conn}
        original_data = connectivity[('A', 'B')].data.copy()
        
        init_weights(connectivity, mode='ones')
        
        # Original should be unchanged
        assert np.all(connectivity[('A', 'B')].data == original_data)

    def test_init_weights_multiple_connectivity_pairs(self):
        """Should handle multiple connectivity pairs."""
        conn_ab = csr_matrix([[1, 0], [0, 1]], dtype=np.float64)
        conn_bc = csr_matrix([[1, 1, 0], [0, 0, 1]], dtype=np.float64)
        
        connectivity = {
            ('A', 'B'): conn_ab,
            ('B', 'C'): conn_bc,
        }
        
        weights = init_weights(connectivity, mode='ones')
        
        assert len(weights) == 2
        assert np.all(weights[('A', 'B')].data == 1.0)
        assert np.all(weights[('B', 'C')].data == 1.0)

    def test_init_weights_empty_matrix(self):
        """Should handle empty (no edges) matrices."""
        conn = csr_matrix((5, 5), dtype=np.float64)
        connectivity = {('A', 'A'): conn}
        
        weights = init_weights(connectivity, mode='ones')
        
        assert weights[('A', 'A')].nnz == 0
        assert weights[('A', 'A')].shape == (5, 5)


class TestNormalizeWeights:
    """Tests for normalize_weights function."""

    def test_normalize_weights_single_fiber_column_normalization(self):
        """Single fiber: columns should sum to 1.0 after normalization."""
        # Create a simple 3x3 matrix where column 0 has two edges, column 1 has one
        conn = csr_matrix([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float64)
        
        weights = {('A', 'B'): conn.copy()}
        
        # Create minimal network spec
        spec = NetworkSpec(
            areas=[AreaSpec(name='A', n=3, k=1), AreaSpec(name='B', n=3, k=1)],
            fibers=[],
        )
        
        normalize_weights(weights, 'B', spec)
        
        # Check column sums
        col_sums = weights[('A', 'B')].sum(axis=0).A1
        assert abs(col_sums[0] - 1.0) < 1e-10, f"col 0: {col_sums[0]}"
        assert abs(col_sums[1] - 1.0) < 1e-10, f"col 1: {col_sums[1]}"
        assert abs(col_sums[2] - 1.0) < 1e-10, f"col 2: {col_sums[2]}"

    def test_normalize_weights_zero_column_handling(self):
        """Columns with no incoming edges should remain zero (not NaN)."""
        # Column 1 has no edges (zero column)
        conn = csr_matrix([[1, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
        
        weights = {('A', 'B'): conn.copy()}
        
        spec = NetworkSpec(
            areas=[AreaSpec(name='A', n=3, k=1), AreaSpec(name='B', n=3, k=1)],
            fibers=[],
        )
        
        normalize_weights(weights, 'B', spec)
        
        col_sums = weights[('A', 'B')].sum(axis=0).A1
        
        # Non-zero columns should sum to 1.0
        assert abs(col_sums[0] - 1.0) < 1e-10
        # Zero column should stay zero (not NaN)
        assert col_sums[1] == 0.0
        assert abs(col_sums[2] - 1.0) < 1e-10
        # Check no NaNs
        assert not np.any(np.isnan(weights[('A', 'B')].data))

    def test_normalize_weights_multiple_fibers_normalizes_independently(self):
        """Multiple fibers to same area: each matrix normalized independently."""
        conn_ab = csr_matrix([[2, 0], [0, 1]], dtype=np.float64)
        conn_cb = csr_matrix([[1, 0], [0, 2]], dtype=np.float64)
        
        weights = {
            ('A', 'B'): conn_ab.copy(),
            ('C', 'B'): conn_cb.copy(),
        }
        
        spec = NetworkSpec(
            areas=[
                AreaSpec(name='A', n=2, k=1),
                AreaSpec(name='B', n=2, k=1),
                AreaSpec(name='C', n=2, k=1),
            ],
            fibers=[],
        )
        
        normalize_weights(weights, 'B', spec)
        
        ab_col0 = np.asarray(weights[('A', 'B')][:, 0].sum())
        ab_col1 = np.asarray(weights[('A', 'B')][:, 1].sum())
        cb_col0 = np.asarray(weights[('C', 'B')][:, 0].sum())
        cb_col1 = np.asarray(weights[('C', 'B')][:, 1].sum())
        
        assert abs(ab_col0 - 1.0) < 1e-10
        assert abs(ab_col1 - 1.0) < 1e-10
        assert abs(cb_col0 - 1.0) < 1e-10
        assert abs(cb_col1 - 1.0) < 1e-10

    def test_normalize_weights_preserves_csr_format(self):
        """Output should remain in CSR format."""
        conn = csr_matrix([[1, 0], [1, 1]], dtype=np.float64)
        weights = {('A', 'B'): conn.copy()}
        
        spec = NetworkSpec(
            areas=[AreaSpec(name='A', n=2, k=1), AreaSpec(name='B', n=2, k=1)],
            fibers=[],
        )
        
        normalize_weights(weights, 'B', spec)
        
        assert isspmatrix_csr(weights[('A', 'B')])

    def test_normalize_weights_preserves_float64_dtype(self):
        """Output dtype should be float64."""
        conn = csr_matrix([[1, 0], [1, 1]], dtype=np.float64)
        weights = {('A', 'B'): conn.copy()}
        
        spec = NetworkSpec(
            areas=[AreaSpec(name='A', n=2, k=1), AreaSpec(name='B', n=2, k=1)],
            fibers=[],
        )
        
        normalize_weights(weights, 'B', spec)
        
        assert weights[('A', 'B')].dtype == np.float64

    def test_normalize_weights_dense_distribution(self):
        """Densely connected matrix should normalize correctly."""
        # All edges present, varying weights
        conn = csr_matrix([[2, 3], [2, 3]], dtype=np.float64)
        weights = {('A', 'B'): conn.copy()}
        
        spec = NetworkSpec(
            areas=[AreaSpec(name='A', n=2, k=1), AreaSpec(name='B', n=2, k=1)],
            fibers=[],
        )
        
        normalize_weights(weights, 'B', spec)
        
        col_sums = weights[('A', 'B')].sum(axis=0).A1
        assert abs(col_sums[0] - 1.0) < 1e-10
        assert abs(col_sums[1] - 1.0) < 1e-10

    def test_normalize_weights_single_edge_per_column(self):
        """One-to-one mapping (identity-like): each column sums to 1.0."""
        conn = csr_matrix(np.eye(5), dtype=np.float64)
        weights = {('A', 'B'): conn.copy()}
        
        spec = NetworkSpec(
            areas=[AreaSpec(name='A', n=5, k=1), AreaSpec(name='B', n=5, k=1)],
            fibers=[],
        )
        
        normalize_weights(weights, 'B', spec)
        
        col_sums = weights[('A', 'B')].sum(axis=0).A1
        assert np.allclose(col_sums, 1.0)
