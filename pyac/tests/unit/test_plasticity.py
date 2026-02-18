import importlib

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr


hebbian_update = importlib.import_module("pyac.core.plasticity").hebbian_update


def _sample_weights() -> csr_matrix:
    row = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    col = np.array([1, 2, 0, 2, 3, 1], dtype=np.int64)
    data = np.ones(6, dtype=np.float64)
    return csr_matrix((data, (row, col)), shape=(4, 4), dtype=np.float64)


class TestHebbianUpdate:
    def test_updates_only_cofiring_pairs(self):
        weights = _sample_weights()

        hebbian_update(
            weights,
            pre_firing=np.array([0, 1], dtype=np.int64),
            post_firing=np.array([2, 3], dtype=np.int64),
            beta=0.5,
        )

        assert abs(weights[0, 2] - 1.5) < 1e-10
        assert abs(weights[1, 2] - 1.5) < 1e-10
        assert abs(weights[0, 1] - 1.0) < 1e-10
        assert abs(weights[2, 3] - 1.0) < 1e-10

    def test_multiplies_existing_nonzero_entries_by_one_plus_beta(self):
        weights = _sample_weights()
        before = weights.copy()

        hebbian_update(
            weights,
            pre_firing=np.array([0, 1], dtype=np.int64),
            post_firing=np.array([2], dtype=np.int64),
            beta=0.25,
        )

        factor = 1.25
        assert abs(weights[0, 2] - before[0, 2] * factor) < 1e-10
        assert abs(weights[1, 2] - before[1, 2] * factor) < 1e-10
        assert abs(weights[1, 0] - before[1, 0]) < 1e-10

    def test_beta_zero_is_noop(self):
        weights = _sample_weights()
        original = weights.copy()

        hebbian_update(
            weights,
            pre_firing=np.array([0, 1], dtype=np.int64),
            post_firing=np.array([0, 1], dtype=np.int64),
            beta=0.0,
        )

        assert (weights - original).nnz == 0

    def test_empty_pre_or_post_is_noop(self):
        weights_a = _sample_weights()
        original_a = weights_a.copy()

        hebbian_update(
            weights_a,
            pre_firing=np.array([], dtype=np.int64),
            post_firing=np.array([1, 2], dtype=np.int64),
            beta=0.5,
        )

        assert (weights_a - original_a).nnz == 0

        weights_b = _sample_weights()
        original_b = weights_b.copy()

        hebbian_update(
            weights_b,
            pre_firing=np.array([0, 1], dtype=np.int64),
            post_firing=np.array([], dtype=np.int64),
            beta=0.5,
        )

        assert (weights_b - original_b).nnz == 0

    def test_preserves_csr_and_does_not_change_nnz(self):
        weights = _sample_weights()
        nnz_before = weights.nnz

        hebbian_update(
            weights,
            pre_firing=np.array([0, 1], dtype=np.int64),
            post_firing=np.array([2, 3], dtype=np.int64),
            beta=0.5,
        )

        assert isspmatrix_csr(weights)
        assert weights.nnz == nnz_before
