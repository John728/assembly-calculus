"""
Core types for Assembly Calculus: AreaSpec, FiberSpec, NetworkSpec, Assembly, StepResult, TraceSpec, Trace.

Pure data containers with validation. No behavior logic.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AreaSpec:
    """Specification for an Area (neuron population)."""

    name: str
    n: int
    k: int
    p_recurrent: float = 0.0
    dynamics_type: str = "feedforward"
    dynamics_params: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate AreaSpec constraints."""
        if self.n <= 0:
            raise ValueError("n must be > 0")
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.k > self.n:
            raise ValueError("k must be <= n")
        if not (0 <= self.p_recurrent <= 1):
            raise ValueError("p_recurrent must be in [0, 1]")
        valid_types = {"feedforward", "recurrent", "refracted"}
        if self.dynamics_type not in valid_types:
            raise ValueError(f"dynamics_type must be in {valid_types}")


@dataclass
class FiberSpec:
    """Specification for a connection (fiber) between areas."""

    src: str
    dst: str
    p_fiber: float

    def __post_init__(self):
        """Validate FiberSpec constraints."""
        if self.src == self.dst:
            raise ValueError("src cannot equal dst (no self-connections)")
        if not (0 <= self.p_fiber <= 1):
            raise ValueError("p_fiber must be in [0, 1]")


@dataclass
class NetworkSpec:
    """Specification for a complete network of areas and fibers."""

    areas: list
    fibers: list
    beta: float = 0.1
    homeostasis: bool = True
    step_order: str = "synchronous"

    def __post_init__(self):
        """Validate NetworkSpec constraints."""
        # Check area names are unique
        area_names = [a.name for a in self.areas]
        if len(area_names) != len(set(area_names)):
            raise ValueError("duplicate area names")

        # Check fibers reference existing areas
        area_name_set = set(area_names)
        for fiber in self.fibers:
            if fiber.src not in area_name_set:
                raise ValueError(f"fiber src '{fiber.src}' not in areas")
            if fiber.dst not in area_name_set:
                raise ValueError(f"fiber dst '{fiber.dst}' not in areas")

        # Check beta >= 0
        if self.beta < 0:
            raise ValueError("beta must be >= 0")


@dataclass(frozen=True)
class Assembly:
    """
    Assembly: a subset of neurons (indices) from an area.

    Immutable: indices are sorted and read-only.
    """

    area_name: str
    indices: np.ndarray

    def __post_init__(self):
        """Sort indices and make immutable."""
        # Sort indices
        sorted_indices = np.sort(self.indices).copy()
        # Use object.__setattr__ because this is frozen dataclass
        object.__setattr__(self, "indices", sorted_indices)
        # Make array read-only
        self.indices.flags.writeable = False


@dataclass
class StepResult:
    """Result of one step of computation in an area."""

    area_name: str
    active_indices: np.ndarray
    step_num: int
    converged: bool


@dataclass
class TraceSpec:
    """Specification for a neural trace (eligibility trace)."""

    basis: str
    max_degree: int
    max_delay: int


@dataclass
class Trace:
    """A neural trace with coefficients."""

    spec: TraceSpec
    coefficients: np.ndarray
