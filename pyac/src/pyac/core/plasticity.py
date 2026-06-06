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

    # Reconstruct row indices from CSR indptr (vectorized, no copy)
    nnz = weights.data.size
    row_indices = np.empty(nnz, dtype=np.int64)
    if nnz > 0:
        diff = np.diff(weights.indptr)
        row_indices = np.repeat(np.arange(n_rows, dtype=np.int64), diff)

    # Mask: row is in pre_indices AND column is in post_indices
    row_mask = np.isin(row_indices, pre_indices)
    col_mask = np.isin(weights.indices, post_indices)
    mask = row_mask & col_mask

    if mask.any():
        weights.data[mask] *= 1.0 + beta
