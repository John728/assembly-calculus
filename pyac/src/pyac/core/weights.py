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
    norm_type = getattr(network_spec, 'norm_type', 'l1')
    if norm_type == 'none':
        return

    incoming_keys = [k for k in weights.keys() if k[1] == area_name]
    
    if not incoming_keys:
        return
    
    for key in incoming_keys:
        mat = weights[key]
        if norm_type == 'l1':
            col_sums = np.asarray(mat.sum(axis=0)).ravel()
            col_sums_safe = np.where(col_sums == 0.0, 1.0, col_sums)
            weights[key].data = mat.data / col_sums_safe[mat.indices]
        elif norm_type == 'l2':
            # Oja's rule inspired L2 normalization
            mat_sq = mat.copy()
            mat_sq.data = mat_sq.data ** 2
            col_norms = np.sqrt(np.asarray(mat_sq.sum(axis=0)).ravel())
            col_norms_safe = np.where(col_norms == 0.0, 1.0, col_norms)
            weights[key].data = mat.data / col_norms_safe[mat.indices]
        elif norm_type == 'max':
            # Max normalization
            col_max = np.asarray(mat.max(axis=0).todense()).ravel()
            col_max_safe = np.where(col_max == 0.0, 1.0, col_max)
            weights[key].data = mat.data / col_max_safe[mat.indices]
        elif norm_type == 'softmax':
            # Softmax-like normalization across columns (approximate)
            # Find max per column for stability
            col_max = np.asarray(mat.max(axis=0).todense()).ravel()
            shifted_data = mat.data - col_max[mat.indices]
            exp_data = np.exp(shifted_data)
            
            exp_mat = csr_matrix((exp_data, mat.indices, mat.indptr), shape=mat.shape)
            col_sums = np.asarray(exp_mat.sum(axis=0)).ravel()
            col_sums_safe = np.where(col_sums == 0.0, 1.0, col_sums)
            
            weights[key].data = exp_data / col_sums_safe[mat.indices]
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
