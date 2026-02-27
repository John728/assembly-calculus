"""
Weight initialization and normalization for Assembly Calculus.

Two core operations:
1. init_weights: Copy connectivity structure with all values set to 1.0
2. normalize_weights: Column-normalize each incoming weight matrix independently
"""

import numpy as np
from scipy.sparse import csr_matrix

from pyac.core.types import NetworkSpec


def init_weights(
    connectivity: dict[tuple[str, str], csr_matrix],
    mode: str = 'ones',
) -> dict[tuple[str, str], csr_matrix]:
    """
    Initialize weights from connectivity structure.
    
    Creates a new weight matrix for each connection, preserving the sparsity pattern
    of the connectivity matrix but setting all values to 1.0.
    
    Args:
        connectivity: Dict mapping (src, dst) to CSR connectivity matrices.
        mode: Initialization mode. Currently only 'ones' is supported.
    
    Returns:
        New dict with same keys as connectivity, values are CSR matrices with
        same sparsity pattern but all data values set to 1.0 (float64).
    """
    if mode != 'ones':
        raise ValueError(f"mode must be 'ones', got '{mode}'")
    
    weights = {}
    for key, conn_matrix in connectivity.items():
        nnz = conn_matrix.nnz
        new_data = np.ones(nnz, dtype=np.float64)
        new_matrix = csr_matrix(
            (new_data, conn_matrix.indices, conn_matrix.indptr),
            shape=conn_matrix.shape,
            dtype=np.float64,
        )
        weights[key] = new_matrix
    
    return weights


def normalize_weights(
    weights: dict[tuple[str, str], csr_matrix],
    area_name: str,
    network_spec: NetworkSpec,
) -> None:
    """
    Column-normalize each incoming weight matrix to an area independently.
    
    For each incoming weight matrix, each column (target neuron) is divided
    by the column sum of *that matrix alone*, so that every column sums to 1.
    This matches the original paper's normalization:  A /= A.sum(axis=0)
    and  W /= W.sum(axis=0)  applied independently.
    
    Handles zero-sum columns (disconnected neurons) by keeping them at zero.
    
    Modifies weights in place.
    
    Args:
        weights: Dict mapping (src, dst) to CSR weight matrices.
        area_name: Target area name to normalize incoming weights for.
        network_spec: NetworkSpec describing the network topology.
    """
    incoming_keys = [k for k in weights.keys() if k[1] == area_name]
    
    if not incoming_keys:
        return
    
    for key in incoming_keys:
        mat = weights[key]
        col_sums = np.asarray(mat.sum(axis=0)).ravel()
        col_sums_safe = col_sums.copy()
        col_sums_safe[col_sums_safe == 0.0] = 1.0
        
        new_data = mat.data.copy()
        for i, val in enumerate(mat.data):
            col_idx = mat.indices[i]
            new_data[i] = val / col_sums_safe[col_idx]
        
        weights[key].data = new_data
