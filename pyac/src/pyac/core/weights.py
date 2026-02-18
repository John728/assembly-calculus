"""
Weight initialization and normalization for Assembly Calculus.

Two core operations:
1. init_weights: Copy connectivity structure with all values set to 1.0
2. normalize_weights: Column-normalize weights across ALL incoming fibers to an area
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
    Column-normalize weights across ALL incoming fibers to an area.
    
    For each target neuron (column) in the area, computes the sum of all
    incoming weights from all source areas, then divides each matrix's
    contribution by this total. Handles zero-sum columns (disconnected neurons)
    by keeping them at zero (avoiding NaN from divide-by-zero).
    
    Modifies weights in place.
    
    Args:
        weights: Dict mapping (src, dst) to CSR weight matrices.
        area_name: Target area name to normalize incoming weights for.
        network_spec: NetworkSpec describing the network topology.
    """
    incoming_keys = [k for k in weights.keys() if k[1] == area_name]
    
    if not incoming_keys:
        return
    
    first_matrix = weights[incoming_keys[0]]
    n_target_neurons: int = first_matrix.shape[1]  # type: ignore[index]
    
    col_sums = np.zeros(n_target_neurons, dtype=np.float64)
    for key in incoming_keys:
        mat = weights[key]
        col_sums_mat = mat.sum(axis=0)
        col_sums += np.asarray(col_sums_mat).ravel()
    
    col_sums_safe = col_sums.copy()
    col_sums_safe[col_sums_safe == 0.0] = 1.0
    
    for key in incoming_keys:
        mat = weights[key]
        col_sums_broadcast = col_sums_safe[np.newaxis, :]
        
        new_data = mat.data.copy()
        for i, val in enumerate(mat.data):
            col_idx = mat.indices[i]
            new_data[i] = val / col_sums_broadcast[0, col_idx]
        
        weights[key].data = new_data
