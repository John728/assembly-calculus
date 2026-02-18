"""
Comprehensive tests for overlap and stability measures.

Tests written first per TDD discipline.
"""

import numpy as np
import pytest

from pyac.core.types import Assembly
from pyac.core.network import Network
from pyac.core.rng import make_rng
from pyac.measures.overlap import assembly_overlap, assembly_intersection_size
from pyac.measures.stability import convergence_trace, is_stable


class TestAssemblyOverlap:
    """Test Jaccard similarity (overlap) between assemblies."""

    def test_overlap_identical_assemblies(self):
        """Same assembly: overlap = 1.0"""
        asm = Assembly("X", np.array([1, 2, 3, 4, 5]))
        overlap = assembly_overlap(asm, asm)
        assert abs(overlap - 1.0) < 1e-10

    def test_overlap_disjoint_assemblies(self):
        """No common elements: overlap = 0.0"""
        asm_a = Assembly("X", np.array([1, 2, 3]))
        asm_b = Assembly("X", np.array([4, 5, 6]))
        overlap = assembly_overlap(asm_a, asm_b)
        assert abs(overlap - 0.0) < 1e-10

    def test_overlap_partial_intersection(self):
        """Acceptance test: Jaccard = |A ∩ B| / |A ∪ B| = 3/7 ≈ 0.4286"""
        asm_a = Assembly("X", np.array([1, 2, 3, 4, 5]))
        asm_b = Assembly("X", np.array([3, 4, 5, 6, 7]))
        overlap = assembly_overlap(asm_a, asm_b)
        expected = 3 / 7
        assert abs(overlap - expected) < 1e-10

    def test_overlap_empty_assembly_a(self):
        """Empty assembly A: overlap = 0.0"""
        asm_a = Assembly("X", np.array([], dtype=np.int64))
        asm_b = Assembly("X", np.array([1, 2, 3]))
        overlap = assembly_overlap(asm_a, asm_b)
        assert abs(overlap - 0.0) < 1e-10

    def test_overlap_empty_assembly_b(self):
        """Empty assembly B: overlap = 0.0"""
        asm_a = Assembly("X", np.array([1, 2, 3]))
        asm_b = Assembly("X", np.array([], dtype=np.int64))
        overlap = assembly_overlap(asm_a, asm_b)
        assert abs(overlap - 0.0) < 1e-10

    def test_overlap_both_empty(self):
        """Both empty: overlap = 0.0 (no union, no intersection)"""
        asm_a = Assembly("X", np.array([], dtype=np.int64))
        asm_b = Assembly("X", np.array([], dtype=np.int64))
        overlap = assembly_overlap(asm_a, asm_b)
        assert abs(overlap - 0.0) < 1e-10

    def test_overlap_area_mismatch_raises(self):
        """Different areas: raise ValueError"""
        asm_a = Assembly("X", np.array([1, 2, 3]))
        asm_b = Assembly("Y", np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="area.*mismatch"):
            assembly_overlap(asm_a, asm_b)

    def test_overlap_unsorted_input_handled(self):
        """Unsorted input arrays are handled (Assembly sorts in __post_init__)"""
        asm_a = Assembly("X", np.array([5, 1, 3, 2, 4]))
        asm_b = Assembly("X", np.array([7, 4, 6, 3, 5]))
        overlap = assembly_overlap(asm_a, asm_b)
        # {1,2,3,4,5} ∩ {3,4,5,6,7} = {3,4,5}, union = {1,2,3,4,5,6,7}
        expected = 3 / 7
        assert abs(overlap - expected) < 1e-10

    def test_overlap_returns_float64(self):
        """Overlap returns numpy float64"""
        asm_a = Assembly("X", np.array([1, 2, 3]))
        asm_b = Assembly("X", np.array([2, 3, 4]))
        overlap = assembly_overlap(asm_a, asm_b)
        assert isinstance(overlap, (float, np.floating))


class TestAssemblyIntersectionSize:
    """Test raw intersection count between assemblies."""

    def test_intersection_identical(self):
        """Same assembly: size = 5"""
        asm = Assembly("X", np.array([1, 2, 3, 4, 5]))
        size = assembly_intersection_size(asm, asm)
        assert size == 5

    def test_intersection_disjoint(self):
        """No overlap: size = 0"""
        asm_a = Assembly("X", np.array([1, 2, 3]))
        asm_b = Assembly("X", np.array([4, 5, 6]))
        size = assembly_intersection_size(asm_a, asm_b)
        assert size == 0

    def test_intersection_partial(self):
        """Partial overlap: size = 3"""
        asm_a = Assembly("X", np.array([1, 2, 3, 4, 5]))
        asm_b = Assembly("X", np.array([3, 4, 5, 6, 7]))
        size = assembly_intersection_size(asm_a, asm_b)
        assert size == 3

    def test_intersection_empty_a(self):
        """Empty A: size = 0"""
        asm_a = Assembly("X", np.array([], dtype=np.int64))
        asm_b = Assembly("X", np.array([1, 2, 3]))
        size = assembly_intersection_size(asm_a, asm_b)
        assert size == 0

    def test_intersection_area_mismatch_raises(self):
        """Different areas: raise ValueError"""
        asm_a = Assembly("X", np.array([1, 2, 3]))
        asm_b = Assembly("Y", np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="area.*mismatch"):
            assembly_intersection_size(asm_a, asm_b)

    def test_intersection_returns_int(self):
        """Intersection size returns int"""
        asm_a = Assembly("X", np.array([1, 2, 3]))
        asm_b = Assembly("X", np.array([2, 3, 4]))
        size = assembly_intersection_size(asm_a, asm_b)
        assert isinstance(size, (int, np.integer))


class TestConvergenceTrace:
    """Test convergence_trace function recording overlap over time."""

    @pytest.fixture
    def deterministic_rng(self):
        """Deterministic RNG for reproducibility"""
        return make_rng(12345)

    @pytest.fixture
    def simple_network_spec(self):
        """Simple network spec from conftest"""
        from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec
        areas = [
            AreaSpec("X", n=50, k=5),
            AreaSpec("Y", n=50, k=5),
        ]
        fibers = [FiberSpec("X", "Y", p_fiber=0.3)]
        return NetworkSpec(areas=areas, fibers=fibers, beta=0.1)

    def test_convergence_trace_basic(self, simple_network_spec, deterministic_rng):
        """Convergence trace returns list of overlaps"""
        network = Network(simple_network_spec, rng=deterministic_rng)
        stimulus = deterministic_rng.normal(size=50)
        
        overlaps = convergence_trace(
            network, "X", "Y", stimulus, t_steps=5
        )
        
        # Should return 5 overlap values (one per step)
        assert len(overlaps) == 5
        # All overlaps should be in [0, 1]
        assert all(0.0 <= ov <= 1.0 for ov in overlaps)

    def test_convergence_trace_t_steps_validation(self, simple_network_spec, deterministic_rng):
        """t_steps must be >= 1"""
        network = Network(simple_network_spec, rng=deterministic_rng)
        stimulus = deterministic_rng.normal(size=50)
        
        with pytest.raises(ValueError, match="t_steps.*positive"):
            convergence_trace(network, "X", "Y", stimulus, t_steps=0)

    def test_convergence_trace_invalid_source_area(self, simple_network_spec, deterministic_rng):
        """Invalid source area raises ValueError"""
        network = Network(simple_network_spec, rng=deterministic_rng)
        stimulus = deterministic_rng.normal(size=50)
        
        with pytest.raises(ValueError, match="unknown area"):
            convergence_trace(network, "INVALID", "Y", stimulus, t_steps=5)

    def test_convergence_trace_invalid_dest_area(self, simple_network_spec, deterministic_rng):
        """Invalid destination area raises ValueError"""
        network = Network(simple_network_spec, rng=deterministic_rng)
        stimulus = deterministic_rng.normal(size=50)
        
        with pytest.raises(ValueError, match="unknown area"):
            convergence_trace(network, "X", "INVALID", stimulus, t_steps=5)

    def test_convergence_trace_same_area(self, simple_network_spec, deterministic_rng):
        """Same source/destination area is allowed"""
        network = Network(simple_network_spec, rng=deterministic_rng)
        stimulus = deterministic_rng.normal(size=50)
        
        overlaps = convergence_trace(network, "X", "X", stimulus, t_steps=3)
        assert len(overlaps) == 3
        assert all(0.0 <= ov <= 1.0 for ov in overlaps)

    def test_convergence_trace_single_step(self, simple_network_spec, deterministic_rng):
        """Single step returns list with one overlap value"""
        network = Network(simple_network_spec, rng=deterministic_rng)
        stimulus = deterministic_rng.normal(size=50)
        
        overlaps = convergence_trace(network, "X", "Y", stimulus, t_steps=1)
        assert len(overlaps) == 1
        assert 0.0 <= overlaps[0] <= 1.0


class TestIsStable:
    """Test stability detection based on overlap threshold."""

    def test_is_stable_high_overlaps(self):
        """All overlaps above threshold: stable = True"""
        overlaps = [0.95, 0.96, 0.97, 0.98, 0.99]
        assert is_stable(overlaps, threshold=0.95, window=5) is True

    def test_is_stable_low_overlaps(self):
        """All overlaps below threshold: stable = False"""
        overlaps = [0.80, 0.82, 0.84, 0.85, 0.86]
        assert is_stable(overlaps, threshold=0.95, window=5) is False

    def test_is_stable_recent_window_above(self):
        """Last 3 overlaps above threshold: stable = True"""
        overlaps = [0.5, 0.6, 0.7, 0.96, 0.97, 0.98]
        assert is_stable(overlaps, threshold=0.95, window=3) is True

    def test_is_stable_recent_window_below(self):
        """Last 3 overlaps below threshold: stable = False"""
        overlaps = [0.96, 0.97, 0.98, 0.80, 0.82, 0.84]
        assert is_stable(overlaps, threshold=0.95, window=3) is False

    def test_is_stable_window_larger_than_overlaps(self):
        """Window larger than list: check all overlaps"""
        overlaps = [0.96, 0.97, 0.98]
        # Window=10 > len=3, so check all 3
        assert is_stable(overlaps, threshold=0.95, window=10) is True

    def test_is_stable_window_one(self):
        """Window=1: check only last overlap"""
        overlaps = [0.5, 0.6, 0.7, 0.96]
        assert is_stable(overlaps, threshold=0.95, window=1) is True
        
        overlaps_unstable = [0.96, 0.97, 0.98, 0.80]
        assert is_stable(overlaps_unstable, threshold=0.95, window=1) is False

    def test_is_stable_empty_list(self):
        """Empty overlaps list: stable = False"""
        assert is_stable([], threshold=0.95, window=5) is False

    def test_is_stable_exact_threshold(self):
        """Overlap exactly at threshold: should be considered stable (>=)"""
        overlaps = [0.95, 0.95, 0.95]
        assert is_stable(overlaps, threshold=0.95, window=3) is True

    def test_is_stable_just_below_threshold(self):
        """Overlap just below threshold: unstable"""
        overlaps = [0.949, 0.949, 0.949]
        assert is_stable(overlaps, threshold=0.95, window=3) is False

    def test_is_stable_default_threshold(self):
        """Test default threshold=0.95"""
        overlaps = [0.96, 0.97, 0.98]
        # Should use default threshold=0.95
        assert is_stable(overlaps, window=3) is True

    def test_is_stable_default_window(self):
        """Test default window=5"""
        overlaps = [0.96, 0.97, 0.98, 0.99, 0.995]
        # Should use default window=5, check all 5 elements
        assert is_stable(overlaps) is True


class TestMeasuresIntegration:
    """Integration tests combining multiple measure functions."""

    @pytest.fixture
    def deterministic_rng(self):
        """Deterministic RNG for reproducibility"""
        return make_rng(12345)

    @pytest.fixture
    def simple_network_spec(self):
        """Simple network spec"""
        from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec
        areas = [
            AreaSpec("X", n=50, k=5),
            AreaSpec("Y", n=50, k=5),
        ]
        fibers = [FiberSpec("X", "Y", p_fiber=0.3)]
        return NetworkSpec(areas=areas, fibers=fibers, beta=0.1)

    def test_convergence_and_stability_integration(self, simple_network_spec, deterministic_rng):
        """Convergence trace → stability check workflow"""
        network = Network(simple_network_spec, rng=deterministic_rng)
        stimulus = deterministic_rng.normal(size=50)
        
        # Get convergence trace
        overlaps = convergence_trace(
            network, "X", "Y", stimulus, t_steps=8
        )
        
        # Check if stabilized
        is_conv = is_stable(overlaps, threshold=0.8, window=3)
        
        # Should produce reasonable result
        assert isinstance(is_conv, bool)
        assert len(overlaps) == 8

    def test_overlap_symmetry(self):
        """overlap(A, B) == overlap(B, A)"""
        asm_a = Assembly("X", np.array([1, 2, 3, 4, 5]))
        asm_b = Assembly("X", np.array([3, 4, 5, 6, 7]))
        
        ov_ab = assembly_overlap(asm_a, asm_b)
        ov_ba = assembly_overlap(asm_b, asm_a)
        
        assert abs(ov_ab - ov_ba) < 1e-14

    def test_intersection_size_symmetry(self):
        """intersection_size(A, B) == intersection_size(B, A)"""
        asm_a = Assembly("X", np.array([1, 2, 3, 4, 5]))
        asm_b = Assembly("X", np.array([3, 4, 5, 6, 7]))
        
        size_ab = assembly_intersection_size(asm_a, asm_b)
        size_ba = assembly_intersection_size(asm_b, asm_a)
        
        assert size_ab == size_ba
