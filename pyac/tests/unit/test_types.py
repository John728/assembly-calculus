"""
Test types.py dataclasses with validation.

TDD: Tests written FIRST, validation/behavior checked via pytest.
"""

import pytest
import numpy as np
from pyac.core.types import (
    AreaSpec,
    FiberSpec,
    NetworkSpec,
    Assembly,
    StepResult,
    TraceSpec,
    Trace,
)


# ============================================================================
# AreaSpec Tests
# ============================================================================


class TestAreaSpecValid:
    """Valid AreaSpec construction."""

    def test_areaspec_minimal(self):
        """Minimal AreaSpec with required fields."""
        a = AreaSpec("A", n=100, k=10)
        assert a.name == "A"
        assert a.n == 100
        assert a.k == 10
        assert a.p_recurrent == 0.0
        assert a.dynamics_type == "feedforward"
        assert a.dynamics_params == {}

    def test_areaspec_with_recurrent(self):
        """AreaSpec with recurrent dynamics."""
        a = AreaSpec("B", n=50, k=5, p_recurrent=0.2, dynamics_type="recurrent")
        assert a.name == "B"
        assert a.p_recurrent == 0.2
        assert a.dynamics_type == "recurrent"

    def test_areaspec_with_dynamics_params(self):
        """AreaSpec with custom dynamics parameters."""
        params = {"tau": 10, "threshold": 0.5}
        a = AreaSpec("C", n=30, k=3, dynamics_params=params)
        assert a.dynamics_params == params

    def test_areaspec_refracted_type(self):
        """AreaSpec with refracted dynamics type."""
        a = AreaSpec("D", n=100, k=10, dynamics_type="refracted")
        assert a.dynamics_type == "refracted"


class TestAreaSpecValidation:
    """AreaSpec validation rules."""

    def test_areaspec_n_must_be_positive(self):
        """n must be > 0."""
        with pytest.raises(ValueError, match="n"):
            AreaSpec("A", n=0, k=5)
        with pytest.raises(ValueError, match="n"):
            AreaSpec("A", n=-10, k=5)

    def test_areaspec_k_must_be_positive(self):
        """k must be > 0."""
        with pytest.raises(ValueError, match="k"):
            AreaSpec("A", n=100, k=0)
        with pytest.raises(ValueError, match="k"):
            AreaSpec("A", n=100, k=-5)

    def test_areaspec_k_gt_n_raises(self):
        """k must be <= n."""
        with pytest.raises(ValueError, match="k"):
            AreaSpec("A", n=10, k=20)

    def test_areaspec_p_recurrent_range(self):
        """p_recurrent must be in [0, 1]."""
        with pytest.raises(ValueError, match="p_recurrent"):
            AreaSpec("A", n=100, k=10, p_recurrent=-0.1)
        with pytest.raises(ValueError, match="p_recurrent"):
            AreaSpec("A", n=100, k=10, p_recurrent=1.5)

    def test_areaspec_dynamics_type_valid(self):
        """dynamics_type must be in allowed set."""
        with pytest.raises(ValueError, match="dynamics_type"):
            AreaSpec("A", n=100, k=10, dynamics_type="invalid")


# ============================================================================
# FiberSpec Tests
# ============================================================================


class TestFiberSpecValid:
    """Valid FiberSpec construction."""

    def test_fiberspec_minimal(self):
        """Minimal FiberSpec."""
        f = FiberSpec("A", "B", 0.5)
        assert f.src == "A"
        assert f.dst == "B"
        assert f.p_fiber == 0.5

    def test_fiberspec_boundary_probabilities(self):
        """FiberSpec at probability boundaries."""
        f0 = FiberSpec("A", "B", 0.0)
        assert f0.p_fiber == 0.0
        f1 = FiberSpec("A", "B", 1.0)
        assert f1.p_fiber == 1.0


class TestFiberSpecValidation:
    """FiberSpec validation rules."""

    def test_fiberspec_self_connection_raises(self):
        """src cannot equal dst."""
        with pytest.raises(ValueError, match="src"):
            FiberSpec("A", "A", 0.5)

    def test_fiberspec_p_fiber_range(self):
        """p_fiber must be in [0, 1]."""
        with pytest.raises(ValueError, match="p_fiber"):
            FiberSpec("A", "B", -0.1)
        with pytest.raises(ValueError, match="p_fiber"):
            FiberSpec("A", "B", 1.5)


# ============================================================================
# NetworkSpec Tests
# ============================================================================


class TestNetworkSpecValid:
    """Valid NetworkSpec construction."""

    def test_networkspec_minimal(self):
        """Minimal NetworkSpec."""
        areas = [AreaSpec("A", n=100, k=10), AreaSpec("B", n=50, k=5)]
        fibers = [FiberSpec("A", "B", 0.1)]
        net = NetworkSpec(areas=areas, fibers=fibers)
        assert len(net.areas) == 2
        assert len(net.fibers) == 1
        assert net.beta == 0.1
        assert net.homeostasis is True
        assert net.step_order == "synchronous"

    def test_networkspec_custom_params(self):
        """NetworkSpec with custom parameters."""
        areas = [AreaSpec("A", n=100, k=10)]
        fibers = []
        net = NetworkSpec(
            areas=areas,
            fibers=fibers,
            beta=0.05,
            homeostasis=False,
            step_order="sequential",
        )
        assert net.beta == 0.05
        assert net.homeostasis is False
        assert net.step_order == "sequential"


class TestNetworkSpecValidation:
    """NetworkSpec validation rules."""

    def test_networkspec_fiber_src_must_exist(self):
        """Fiber src must reference existing area."""
        areas = [AreaSpec("A", n=100, k=10)]
        fibers = [FiberSpec("X", "A", 0.1)]
        with pytest.raises(ValueError, match="src"):
            NetworkSpec(areas=areas, fibers=fibers)

    def test_networkspec_fiber_dst_must_exist(self):
        """Fiber dst must reference existing area."""
        areas = [AreaSpec("A", n=100, k=10)]
        fibers = [FiberSpec("A", "X", 0.1)]
        with pytest.raises(ValueError, match="dst"):
            NetworkSpec(areas=areas, fibers=fibers)

    def test_networkspec_no_duplicate_area_names(self):
        """Area names must be unique."""
        areas = [AreaSpec("A", n=100, k=10), AreaSpec("A", n=50, k=5)]
        with pytest.raises(ValueError, match="duplicate"):
            NetworkSpec(areas=areas, fibers=[])

    def test_networkspec_beta_nonnegative(self):
        """beta must be >= 0."""
        areas = [AreaSpec("A", n=100, k=10)]
        with pytest.raises(ValueError, match="beta"):
            NetworkSpec(areas=areas, fibers=[], beta=-0.1)


# ============================================================================
# Assembly Tests
# ============================================================================


class TestAssemblyValid:
    """Valid Assembly construction."""

    def test_assembly_minimal(self):
        """Minimal Assembly."""
        a = Assembly("X", np.array([1, 2, 3]))
        assert a.area_name == "X"
        assert isinstance(a.indices, np.ndarray)

    def test_assembly_single_index(self):
        """Assembly with single index."""
        a = Assembly("Y", np.array([5]))
        assert a.area_name == "Y"
        assert len(a.indices) == 1


class TestAssemblySorting:
    """Assembly sorts indices."""

    def test_assembly_sorts_unsorted_indices(self):
        """Assembly constructor sorts indices."""
        a = Assembly("X", np.array([5, 2, 8, 1]))
        assert list(a.indices) == [1, 2, 5, 8]

    def test_assembly_maintains_sorted_order(self):
        """Already sorted indices stay sorted."""
        a = Assembly("X", np.array([1, 2, 3, 4, 5]))
        assert list(a.indices) == [1, 2, 3, 4, 5]

    def test_assembly_duplicates_preserved(self):
        """Duplicates in input are preserved (no dedup)."""
        a = Assembly("X", np.array([3, 1, 3, 2]))
        assert list(a.indices) == [1, 2, 3, 3]


class TestAssemblyImmutability:
    """Assembly indices are immutable."""

    def test_assembly_indices_readonly(self):
        """Cannot modify indices array after construction."""
        a = Assembly("X", np.array([1, 2, 3]))
        with pytest.raises((ValueError, TypeError)):
            a.indices[0] = 99

    def test_assembly_indices_is_copy(self):
        """Assembly makes own copy of input indices."""
        orig = np.array([3, 1, 2])
        a = Assembly("X", orig)
        # Modify original
        orig[0] = 999
        # Assembly indices unaffected
        assert list(a.indices) == [1, 2, 3]


# ============================================================================
# StepResult Tests
# ============================================================================


class TestStepResult:
    """StepResult simple dataclass."""

    def test_stepresult_valid(self):
        """StepResult construction."""
        sr = StepResult(
            area_name="A",
            active_indices=np.array([1, 2, 3]),
            step_num=5,
            converged=False,
        )
        assert sr.area_name == "A"
        assert list(sr.active_indices) == [1, 2, 3]
        assert sr.step_num == 5
        assert sr.converged is False


# ============================================================================
# TraceSpec Tests
# ============================================================================


class TestTraceSpec:
    """TraceSpec simple dataclass."""

    def test_tracespec_valid(self):
        """TraceSpec construction."""
        ts = TraceSpec(
            basis="legendre",
            max_degree=5,
            max_delay=10,
        )
        assert ts.basis == "legendre"
        assert ts.max_degree == 5
        assert ts.max_delay == 10


# ============================================================================
# Trace Tests
# ============================================================================


class TestTrace:
    """Trace simple dataclass."""

    def test_trace_valid(self):
        """Trace construction."""
        coeffs = np.array([0.1, 0.2, 0.3])
        t = Trace(spec=TraceSpec("legendre", 3, 10), coefficients=coeffs)
        assert t.spec.basis == "legendre"
        assert list(t.coefficients) == [0.1, 0.2, 0.3]
