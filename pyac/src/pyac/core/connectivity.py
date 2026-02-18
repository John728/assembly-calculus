from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.sparse import coo_matrix, csr_matrix

from pyac.core.types import NetworkSpec


def _random_sparse_matrix(
    n_rows: int,
    n_cols: int,
    density: float,
    rng: Generator,
) -> csr_matrix:
    if n_rows < 0 or n_cols < 0:
        raise ValueError("matrix dimensions must be >= 0")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0, 1]")

    total = n_rows * n_cols
    if total == 0:
        return csr_matrix((n_rows, n_cols), dtype=np.float64)

    n_edges = int(rng.binomial(total, density))
    if n_edges == 0:
        return csr_matrix((n_rows, n_cols), dtype=np.float64)

    if n_edges == total:
        flat_indices = np.arange(total, dtype=np.int64)
    else:
        flat_indices = rng.choice(total, size=n_edges, replace=False)

    row = flat_indices // n_cols
    col = flat_indices % n_cols
    data = np.ones(n_edges, dtype=np.float64)
    coo = coo_matrix((data, (row, col)), shape=(n_rows, n_cols), dtype=np.float64)
    return csr_matrix(coo)


def build_connectivity(
    spec: NetworkSpec,
    rng: Generator,
) -> dict[tuple[str, str], csr_matrix]:
    area_sizes = {area.name: area.n for area in spec.areas}
    matrices: dict[tuple[str, str], csr_matrix] = {}

    for fiber in spec.fibers:
        matrices[(fiber.src, fiber.dst)] = _random_sparse_matrix(
            n_rows=area_sizes[fiber.src],
            n_cols=area_sizes[fiber.dst],
            density=fiber.p_fiber,
            rng=rng,
        )

    for area in spec.areas:
        if area.p_recurrent > 0.0:
            matrices[(area.name, area.name)] = _random_sparse_matrix(
                n_rows=area.n,
                n_cols=area.n,
                density=area.p_recurrent,
                rng=rng,
            )

    return matrices
