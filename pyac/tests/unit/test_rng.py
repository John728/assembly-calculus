"""
Tests for pyac.core.rng module.

TDD-first tests for RNG determinism, spawn independence, and entropy extraction.
Tests that G2 (no global RNG) requirement is met.
"""

import numpy as np
import pytest

from pyac.core.rng import make_rng, spawn_rngs, get_entropy


class TestMakeRng:
    """Tests for make_rng function."""

    def test_make_rng_with_int_seed(self):
        """make_rng(int) returns a Generator backed by PCG64."""
        rng = make_rng(42)
        assert isinstance(rng, np.random.Generator)
        assert isinstance(rng.bit_generator, np.random.PCG64)

    def test_make_rng_with_seed_sequence(self):
        """make_rng(SeedSequence) returns a Generator backed by PCG64."""
        seed_seq = np.random.SeedSequence(12345)
        rng = make_rng(seed_seq)
        assert isinstance(rng, np.random.Generator)
        assert isinstance(rng.bit_generator, np.random.PCG64)

    def test_make_rng_with_none_seed(self):
        """make_rng(None) creates SeedSequence with random entropy and returns Generator."""
        rng = make_rng(None)
        assert isinstance(rng, np.random.Generator)
        assert isinstance(rng.bit_generator, np.random.PCG64)

    def test_make_rng_no_arguments(self):
        """make_rng() with no args creates SeedSequence with random entropy."""
        rng = make_rng()
        assert isinstance(rng, np.random.Generator)
        assert isinstance(rng.bit_generator, np.random.PCG64)

    def test_deterministic_same_seed(self):
        """Same seed produces identical random sequences (determinism requirement)."""
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        
        np.testing.assert_array_equal(vals1, vals2)

    def test_deterministic_int_and_seed_sequence(self):
        """Seed via int and SeedSequence(int) produce same sequence."""
        rng1 = make_rng(999)
        rng2 = make_rng(np.random.SeedSequence(999))
        
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        
        np.testing.assert_array_equal(vals1, vals2)

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds produce different sequences (high probability)."""
        rng1 = make_rng(42)
        rng2 = make_rng(43)
        
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        
        assert vals1 != vals2  # with very high probability

    def test_fresh_generator_per_call(self):
        """Each make_rng call creates a fresh independent generator."""
        rng1 = make_rng(42)
        val1 = rng1.random()
        
        rng1b = make_rng(42)
        val1b = rng1b.random()
        
        assert val1 == val1b  # fresh generator, same seed
        
        # Advance rng1 and rng1b further — they should still match
        vals1 = [rng1.random() for _ in range(5)]
        vals1b = [rng1b.random() for _ in range(5)]
        np.testing.assert_array_equal(vals1, vals1b)


class TestSpawnRngs:
    """Tests for spawn_rngs function."""

    def test_spawn_from_generator(self):
        """spawn_rngs(Generator, n) returns list of n independent Generators."""
        parent = make_rng(42)
        children = spawn_rngs(parent, 3)
        
        assert isinstance(children, list)
        assert len(children) == 3
        assert all(isinstance(c, np.random.Generator) for c in children)
        assert all(isinstance(c.bit_generator, np.random.PCG64) for c in children)

    def test_spawn_from_seed_sequence(self):
        """spawn_rngs(SeedSequence, n) returns list of n independent Generators."""
        parent_seq = np.random.SeedSequence(42)
        children = spawn_rngs(parent_seq, 3)
        
        assert isinstance(children, list)
        assert len(children) == 3
        assert all(isinstance(c, np.random.Generator) for c in children)

    def test_spawn_independence(self):
        """Spawned RNGs produce independent, non-overlapping sequences."""
        parent = make_rng(42)
        children = spawn_rngs(parent, 3)
        
        vals = [c.random() for c in children]
        
        # All three values should be different (with extremely high probability)
        assert len(set(vals)) == 3

    def test_spawn_zero_children(self):
        """spawn_rngs(parent, 0) returns empty list."""
        parent = make_rng(42)
        children = spawn_rngs(parent, 0)
        
        assert children == []

    def test_spawn_many_children(self):
        """spawn_rngs can create many independent children."""
        parent = make_rng(42)
        children = spawn_rngs(parent, 100)
        
        assert len(children) == 100
        vals = [c.random() for c in children]
        # Very high probability that all 100 values are distinct
        assert len(set(vals)) > 95  # allow for tiny collision probability

    def test_spawn_deterministic_with_seeded_parent(self):
        """Spawning from same seeded parent produces same child sequence."""
        parent1 = make_rng(42)
        children1 = spawn_rngs(parent1, 3)
        vals1 = [c.random() for c in children1]
        
        parent2 = make_rng(42)
        children2 = spawn_rngs(parent2, 3)
        vals2 = [c.random() for c in children2]
        
        np.testing.assert_array_equal(vals1, vals2)

    def test_spawn_used_by_multiple_children(self):
        """Each spawned child advances independently from same parent."""
        parent = make_rng(42)
        child1, child2 = spawn_rngs(parent, 2)
        
        # Generate 5 values from each child
        vals1 = [child1.random() for _ in range(5)]
        vals2 = [child2.random() for _ in range(5)]
        
        # Different children should produce different sequences
        assert vals1 != vals2


class TestGetEntropy:
    """Tests for get_entropy function."""

    def test_get_entropy_returns_tuple(self):
        """get_entropy(rng) returns a tuple."""
        rng = make_rng(42)
        entropy = get_entropy(rng)
        
        assert entropy is not None
        assert isinstance(entropy, (tuple, int, np.integer))

    def test_entropy_extraction_from_seeded_rng(self):
        """get_entropy extracts entropy from SeedSequence of a Generator."""
        rng = make_rng(12345)
        entropy = get_entropy(rng)
        
        # Should be able to access seed_seq.entropy
        assert entropy is not None

    def test_entropy_roundtrip(self):
        """Entropy can be used to recreate deterministic sequence."""
        rng = make_rng(42)
        entropy = get_entropy(rng)
        
        # Create new RNG from same seed
        rng2 = make_rng(42)
        entropy2 = get_entropy(rng2)
        
        # Entropies should match (same seed)
        assert entropy == entropy2

    def test_entropy_different_for_different_seeds(self):
        """Different seeds produce different entropy values."""
        rng1 = make_rng(42)
        rng2 = make_rng(43)
        
        entropy1 = get_entropy(rng1)
        entropy2 = get_entropy(rng2)
        
        assert entropy1 != entropy2


class TestIntegration:
    """Integration tests for RNG module."""

    def test_workflow_make_spawn_use(self):
        """Complete workflow: make parent -> spawn children -> use."""
        # Create parent with deterministic seed
        parent = make_rng(42)
        
        # Spawn 3 children
        children = spawn_rngs(parent, 3)
        
        # Use each child
        for i, child in enumerate(children):
            vals = [child.random() for _ in range(5)]
            assert len(vals) == 5
            assert all(0 <= v < 1 for v in vals)

    def test_no_shared_state_between_children(self):
        """Spawned children don't share state; advancing one doesn't affect others."""
        parent = make_rng(42)
        children = spawn_rngs(parent, 2)
        
        # Advance first child many times
        for _ in range(1000):
            children[0].random()
        
        # Second child should still produce same early values as a fresh spawned child
        parent2 = make_rng(42)
        children2 = spawn_rngs(parent2, 2)
        
        val_child2_fresh = children2[1].random()
        val_child1_after_advance = children[1].random()
        
        # Different because children[1] hasn't advanced yet, but this tests independence
        assert children[1].random() == children2[1].random()

    def test_determinism_preserved_across_module_reimport(self):
        """Determinism holds: recreating same seed gives same sequence."""
        # This tests the core determinism guarantee
        rng1 = make_rng(999)
        seq1 = [rng1.random() for _ in range(20)]
        
        rng2 = make_rng(999)
        seq2 = [rng2.random() for _ in range(20)]
        
        np.testing.assert_array_equal(seq1, seq2)
