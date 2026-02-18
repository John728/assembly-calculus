"""
Pytest configuration and fixtures for pyac tests.

Provides deterministic RNG and network specs for unit and integration tests.
"""

import pytest
import numpy as np


@pytest.fixture
def deterministic_rng():
    """
    Create a deterministic RNG seeded with 12345.
    
    Used throughout test suite to ensure reproducible results.
    """
    from pyac.core.rng import make_rng
    return make_rng(12345)


@pytest.fixture
def small_network_spec():
    """
    Small network spec for quick unit tests.
    
    Two areas with 100 neurons each, connected by fiber.
    """
    from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec
    
    return NetworkSpec(
        areas=[AreaSpec('A', 100, 10), AreaSpec('B', 100, 10)],
        fibers=[FiberSpec('A', 'B', 0.05)],
        beta=0.1
    )


@pytest.fixture
def tiny_network_spec():
    """
    Tiny network spec for smoke tests and rapid iteration.
    
    Two areas with 20 neurons each, connected by fiber.
    """
    from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec
    
    return NetworkSpec(
        areas=[AreaSpec('A', 20, 5), AreaSpec('B', 20, 5)],
        fibers=[FiberSpec('A', 'B', 0.1)],
        beta=0.5
    )
