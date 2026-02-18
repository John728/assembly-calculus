from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _as_index_array(indices: np.ndarray) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError("firing indices must be 1D")
    if array.size == 0:
        return array.astype(np.int64, copy=False)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("firing indices must be integer dtype")
    return array.astype(np.int64, copy=False)


def hebbian_update(
    weights: csr_matrix,
    pre_firing: np.ndarray,
    post_firing: np.ndarray,
    beta: float,
) -> None:
    pre_indices = _as_index_array(pre_firing)
    post_indices = _as_index_array(post_firing)

    if beta == 0.0 or pre_indices.size == 0 or post_indices.size == 0:
        return

    n_rows, n_cols = weights.get_shape()
    if np.any(pre_indices < 0) or np.any(pre_indices >= n_rows):
        raise IndexError("pre_firing indices out of bounds")
    if np.any(post_indices < 0) or np.any(post_indices >= n_cols):
        raise IndexError("post_firing indices out of bounds")

    index = np.ix_(pre_indices, post_indices)
    submatrix = weights[index]
    if submatrix.nnz == 0:
        return

    submatrix.data *= 1.0 + beta
    weights[index] = submatrix
