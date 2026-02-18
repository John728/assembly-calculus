"""
Test suite for network serialization (save/load via pickle).

TDD approach: All tests written first, then implementation follows.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyac.core.network import Network
from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec


@pytest.fixture
def simple_spec():
    """Two-area network with one fiber for testing."""
    return NetworkSpec(
        areas=[AreaSpec("A", 50, 5), AreaSpec("B", 50, 5)],
        fibers=[FiberSpec("A", "B", 0.1)],
        beta=0.1,
    )


@pytest.fixture
def simple_network(simple_spec):
    """Create a simple network with seed for reproducibility."""
    return Network(simple_spec, make_rng(42))


class TestSaveNetworkBasics:
    """Tests for save_network function."""

    def test_save_network_creates_file(self, simple_network):
        """save_network should create a pickle file at the specified path."""
        from pyac_experiments.serialization import save_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            assert Path(path).exists()

    def test_saved_file_is_valid_pickle(self, simple_network):
        """Saved file should be valid pickle format."""
        from pyac_experiments.serialization import save_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)

            # Should not raise
            with open(path, "rb") as f:
                data = pickle.load(f)
            assert data is not None

    def test_save_network_with_modified_state(self, simple_network):
        """Should be able to save network after running steps."""
        from pyac_experiments.serialization import save_network

        stim = {"A": np.zeros(50)}
        stim["A"][:10] = 1.0
        simple_network.step(external_stimuli=stim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            assert Path(path).exists()


class TestLoadNetworkBasics:
    """Tests for load_network function."""

    def test_load_network_returns_network(self, simple_network):
        """load_network should return a Network instance."""
        from pyac_experiments.serialization import save_network, load_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)
            assert isinstance(loaded, Network)

    def test_load_network_nonexistent_path(self):
        """load_network should raise for nonexistent file."""
        from pyac_experiments.serialization import load_network

        with pytest.raises(FileNotFoundError):
            load_network("/nonexistent/path/network.pkl")


class TestRoundtripStatePreservation:
    """Tests that roundtrip save/load preserves full network state."""

    def test_roundtrip_preserves_spec(self, simple_network):
        """After load, spec should match original."""
        from pyac_experiments.serialization import save_network, load_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            assert loaded.spec.beta == simple_network.spec.beta
            assert loaded.spec.areas[0].n == simple_network.spec.areas[0].n
            assert loaded.spec.areas[0].k == simple_network.spec.areas[0].k

    def test_roundtrip_preserves_area_names(self, simple_network):
        """After load, area names should be identical."""
        from pyac_experiments.serialization import save_network, load_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            assert loaded.area_names == simple_network.area_names

    def test_roundtrip_preserves_weights(self, simple_network):
        """After load, weights matrices should be identical."""
        from pyac_experiments.serialization import save_network, load_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            for key in simple_network.weights.keys():
                orig = simple_network.weights[key]
                new = loaded.weights[key]
                assert np.allclose(orig.data, new.data)
                assert np.array_equal(orig.indices, new.indices)
                assert np.array_equal(orig.indptr, new.indptr)

    def test_roundtrip_preserves_activations(self, simple_network):
        """After load, activations should match."""
        from pyac_experiments.serialization import save_network, load_network

        stim = {"A": np.zeros(50)}
        stim["A"][:10] = 1.0
        simple_network.step(external_stimuli=stim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            for area_name in simple_network.area_names:
                orig_act = simple_network.activations[area_name]
                new_act = loaded.activations[area_name]
                assert np.array_equal(orig_act, new_act)

    def test_roundtrip_preserves_step_count(self, simple_network):
        """After load, step_count should match."""
        from pyac_experiments.serialization import save_network, load_network

        # Run a few steps
        for _ in range(3):
            simple_network.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            assert loaded.step_count == simple_network.step_count

    def test_roundtrip_preserves_strategy_objects(self, simple_network):
        """After load, strategy objects should be present."""
        from pyac_experiments.serialization import save_network, load_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            for area_name in simple_network.area_names:
                assert area_name in loaded.strategies
                orig_strategy = simple_network.strategies[area_name]
                new_strategy = loaded.strategies[area_name]
                assert type(orig_strategy) == type(new_strategy)


class TestRNGStatePreservation:
    """Tests that RNG state is preserved during roundtrip."""

    def test_roundtrip_preserves_rng_state(self, simple_network):
        """After load, RNG continues from same state point."""
        from pyac_experiments.serialization import save_network, load_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            # Both RNGs should produce identical next values since saved at same state
            orig_next = [simple_network.rng.uniform() for _ in range(5)]
            loaded_next = [loaded.rng.uniform() for _ in range(5)]

            assert np.allclose(orig_next, loaded_next)

    def test_rng_state_after_step_preserved(self, simple_network):
        """RNG state after a step should be preserved on reload."""
        from pyac_experiments.serialization import save_network, load_network

        stim = {"A": np.zeros(50)}
        stim["A"][:10] = 1.0
        simple_network.step(external_stimuli=stim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            orig_randoms = [simple_network.rng.uniform() for _ in range(3)]
            loaded_randoms = [loaded.rng.uniform() for _ in range(3)]
            assert np.allclose(orig_randoms, loaded_randoms)


class TestIdenticalStepResults:
    """Tests that loaded network produces identical step results."""

    def test_loaded_network_identical_next_step(self, simple_network):
        """After load, running identical step should produce same assembly indices."""
        from pyac_experiments.serialization import save_network, load_network

        stim = {"A": np.zeros(50)}
        stim["A"][:10] = 1.0
        result1_first = simple_network.step(external_stimuli=stim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            # Run identical next step with same stimulus
            result1_next = simple_network.step(external_stimuli=stim)
            result2_next = loaded.step(external_stimuli=stim)

            # Assembly indices should match exactly
            for area_name in simple_network.area_names:
                assert np.array_equal(
                    result1_next.assemblies[area_name].indices,
                    result2_next.assemblies[area_name].indices,
                )

    def test_multi_step_roundtrip(self, simple_network):
        """Multiple steps before save and after load should produce same results."""
        from pyac_experiments.serialization import save_network, load_network

        stim = {"A": np.zeros(50)}
        stim["A"][:10] = 1.0
        for _ in range(3):
            simple_network.step(external_stimuli=stim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            # Next step should be identical
            result1 = simple_network.step(external_stimuli=stim)
            result2 = loaded.step(external_stimuli=stim)

            for area_name in simple_network.area_names:
                assert np.array_equal(
                    result1.assemblies[area_name].indices,
                    result2.assemblies[area_name].indices,
                )

    def test_roundtrip_without_stimulus(self, simple_network):
        """Roundtrip should work even without external stimuli."""
        from pyac_experiments.serialization import save_network, load_network

        stim = {"A": np.zeros(50)}
        stim["A"][:10] = 1.0
        simple_network.step(external_stimuli=stim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            # Run step without stimulus
            result1 = simple_network.step()
            result2 = loaded.step()

            for area_name in simple_network.area_names:
                assert np.array_equal(
                    result1.assemblies[area_name].indices,
                    result2.assemblies[area_name].indices,
                )


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_save_network_before_any_steps(self, simple_network):
        """Should be able to save pristine network before any steps."""
        from pyac_experiments.serialization import save_network, load_network

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path)
            loaded = load_network(path)

            assert loaded.step_count == 0
            for area_name in simple_network.area_names:
                assert loaded.activations[area_name].size == 0

    def test_save_and_load_multiple_times(self, simple_network):
        """Should be able to save and load multiple times without data corruption."""
        from pyac_experiments.serialization import save_network, load_network

        stim = {"A": np.zeros(50)}
        stim["A"][:10] = 1.0
        simple_network.step(external_stimuli=stim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = str(Path(tmpdir) / "network1.pkl")
            save_network(simple_network, path1)

            loaded1 = load_network(path1)
            path2 = str(Path(tmpdir) / "network2.pkl")
            save_network(loaded1, path2)

            loaded2 = load_network(path2)

            # Final loaded should match original
            for area_name in simple_network.area_names:
                assert np.array_equal(
                    simple_network.activations[area_name],
                    loaded2.activations[area_name],
                )

    def test_save_respects_filepath_format(self, simple_network):
        """Save should work with various path formats."""
        from pyac_experiments.serialization import save_network

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with str path
            path_str = str(Path(tmpdir) / "network.pkl")
            save_network(simple_network, path_str)
            assert Path(path_str).exists()

            # Test with Path object (if function accepts it)
            path_obj = Path(tmpdir) / "network2.pkl"
            save_network(simple_network, str(path_obj))
            assert path_obj.exists()

    def test_load_with_different_seed_networks(self):
        """Load should work correctly regardless of original seed."""
        from pyac_experiments.serialization import save_network, load_network
        from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec

        spec = NetworkSpec(
            areas=[AreaSpec("A", 50, 5), AreaSpec("B", 50, 5)],
            fibers=[FiberSpec("A", "B", 0.1)],
            beta=0.1,
        )

        net1 = Network(spec, make_rng(42))
        net2 = Network(spec, make_rng(123))

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = str(Path(tmpdir) / "net1.pkl")
            path2 = str(Path(tmpdir) / "net2.pkl")

            save_network(net1, path1)
            save_network(net2, path2)

            loaded1 = load_network(path1)
            loaded2 = load_network(path2)

            # Loaded networks should still be different (from different seeds)
            seq1 = [loaded1.rng.uniform() for _ in range(5)]
            seq2 = [loaded2.rng.uniform() for _ in range(5)]
            assert not np.allclose(seq1, seq2)
