import numpy as np
from scipy.sparse import isspmatrix_csr

from pyac.core.connectivity import build_connectivity, _random_sparse_matrix
from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec


class TestRandomSparseMatrix:
    def test_returns_csr_float64_with_binary_data(self):
        rng = make_rng(7)
        mat = _random_sparse_matrix(25, 40, 0.2, rng)

        assert isspmatrix_csr(mat)
        assert mat.shape == (25, 40)
        assert mat.dtype == np.float64
        assert np.all(mat.data == 1.0)

    def test_zero_density_returns_empty_matrix(self):
        rng = make_rng(11)
        mat = _random_sparse_matrix(30, 20, 0.0, rng)

        assert isspmatrix_csr(mat)
        assert mat.nnz == 0
        assert mat.shape == (30, 20)

    def test_unit_density_returns_full_binary_matrix(self):
        rng = make_rng(13)
        n_rows = 9
        n_cols = 7
        mat = _random_sparse_matrix(n_rows, n_cols, 1.0, rng)

        assert mat.nnz == n_rows * n_cols
        assert np.all(mat.data == 1.0)

    def test_same_seed_is_deterministic(self):
        mat_a = _random_sparse_matrix(80, 60, 0.15, make_rng(42))
        mat_b = _random_sparse_matrix(80, 60, 0.15, make_rng(42))

        assert mat_a.shape == mat_b.shape
        assert mat_a.nnz == mat_b.nnz
        assert (mat_a - mat_b).nnz == 0

    def test_density_is_within_expected_statistical_band(self):
        n_rows = 1000
        n_cols = 1000
        p = 0.1
        mat = _random_sparse_matrix(n_rows, n_cols, p, make_rng(12345))

        observed_density = mat.nnz / (n_rows * n_cols)
        assert 0.08 <= observed_density <= 0.12


class TestBuildConnectivity:
    def test_builds_fiber_and_recurrent_connectivity(self):
        spec = NetworkSpec(
            areas=[
                AreaSpec("A", n=100, k=10, p_recurrent=0.05),
                AreaSpec("B", n=70, k=7, p_recurrent=0.0),
                AreaSpec("C", n=50, k=5, p_recurrent=0.1),
            ],
            fibers=[
                FiberSpec("A", "B", 0.2),
                FiberSpec("B", "C", 0.3),
            ],
        )

        conn = build_connectivity(spec, make_rng(2024))

        assert ("A", "B") in conn
        assert ("B", "C") in conn
        assert ("A", "A") in conn
        assert ("C", "C") in conn
        assert ("B", "B") not in conn

    def test_matrices_have_expected_shapes(self):
        spec = NetworkSpec(
            areas=[
                AreaSpec("X", n=31, k=3, p_recurrent=0.0),
                AreaSpec("Y", n=47, k=4, p_recurrent=0.25),
            ],
            fibers=[FiberSpec("X", "Y", 0.4)],
        )

        conn = build_connectivity(spec, make_rng(77))

        assert conn[("X", "Y")].shape == (31, 47)
        assert conn[("Y", "Y")].shape == (47, 47)

    def test_all_outputs_are_csr_float64_and_binary(self):
        spec = NetworkSpec(
            areas=[
                AreaSpec("I", n=40, k=4, p_recurrent=0.2),
                AreaSpec("J", n=35, k=3, p_recurrent=0.1),
            ],
            fibers=[FiberSpec("I", "J", 0.3)],
        )

        conn = build_connectivity(spec, make_rng(8))

        for mat in conn.values():
            assert isspmatrix_csr(mat)
            assert mat.dtype == np.float64
            assert np.all(mat.data == 1.0)

    def test_determinism_same_seed_produces_identical_connectivity(self):
        spec = NetworkSpec(
            areas=[
                AreaSpec("A", n=120, k=12, p_recurrent=0.1),
                AreaSpec("B", n=90, k=9, p_recurrent=0.2),
                AreaSpec("C", n=60, k=6, p_recurrent=0.0),
            ],
            fibers=[
                FiberSpec("A", "B", 0.05),
                FiberSpec("B", "C", 0.07),
                FiberSpec("A", "C", 0.02),
            ],
        )

        conn_a = build_connectivity(spec, make_rng(31415))
        conn_b = build_connectivity(spec, make_rng(31415))

        assert set(conn_a.keys()) == set(conn_b.keys())
        for key in conn_a:
            assert conn_a[key].shape == conn_b[key].shape
            assert conn_a[key].nnz == conn_b[key].nnz
            assert (conn_a[key] - conn_b[key]).nnz == 0

    def test_returns_empty_dict_when_no_fibers_and_no_recurrence(self):
        spec = NetworkSpec(
            areas=[AreaSpec("A", n=10, k=2, p_recurrent=0.0)],
            fibers=[],
        )

        conn = build_connectivity(spec, make_rng(99))
        assert conn == {}
