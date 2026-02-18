from __future__ import annotations

import importlib
import inspect

import numpy as np

from pyac.core.rng import make_rng

inhibition = importlib.import_module("pyac.core.inhibition")


class TestKCap:
    def test_returns_exactly_k_unique_indices(self):
        rng = make_rng(123)
        input_vector = rng.random(100)

        winners = inhibition.k_cap(input_vector, k=10, rng=rng)

        assert isinstance(winners, np.ndarray)
        assert len(winners) == 10
        assert len(np.unique(winners)) == 10
        assert np.all(np.diff(winners) >= 0)
        assert np.all((0 <= winners) & (winners < len(input_vector)))

    def test_zero_input_still_returns_k_winners(self):
        input_vector = np.zeros(50, dtype=np.float64)

        winners_a = inhibition.k_cap(input_vector, k=5, rng=make_rng(42))
        winners_b = inhibition.k_cap(input_vector, k=5, rng=make_rng(42))

        assert len(winners_a) == 5
        assert len(np.unique(winners_a)) == 5
        assert np.array_equal(winners_a, winners_b)

    def test_deterministic_tie_breaking_same_seed(self):
        input_vector = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float64,
        )

        winners_a = inhibition.k_cap(input_vector, k=5, rng=make_rng(999))
        winners_b = inhibition.k_cap(input_vector, k=5, rng=make_rng(999))

        assert np.array_equal(winners_a, winners_b)

    def test_degenerate_single_neuron_case(self):
        input_vector = np.array([1.0], dtype=np.float64)

        winners = inhibition.k_cap(input_vector, k=1, rng=make_rng(7))

        np.testing.assert_array_equal(winners, np.array([0]))

    def test_returns_top_k_when_inputs_are_distinct(self):
        input_vector = np.array([0.1, 0.8, 0.2, 1.4, 0.9, 0.3], dtype=np.float64)

        winners = inhibition.k_cap(input_vector, k=3, rng=make_rng(5))

        np.testing.assert_array_equal(winners, np.array([1, 3, 4]))

    def test_implementation_uses_argpartition_not_argsort(self):
        source = inspect.getsource(inhibition.k_cap)

        assert "argpartition" in source
        assert "argsort" not in source


class TestInhibitionState:
    def test_unknown_area_defaults_to_not_inhibited(self):
        state = inhibition.InhibitionState()

        assert state.is_inhibited("A") is False

    def test_inhibit_marks_area_inhibited(self):
        state = inhibition.InhibitionState()

        state.inhibit("A")

        assert state.is_inhibited("A") is True

    def test_disinhibit_clears_inhibition(self):
        state = inhibition.InhibitionState()
        state.inhibit("A")

        state.disinhibit("A")

        assert state.is_inhibited("A") is False

    def test_tracks_multiple_areas_independently(self):
        state = inhibition.InhibitionState()

        state.inhibit("A")
        state.disinhibit("B")
        state.inhibit("C")

        assert state.is_inhibited("A") is True
        assert state.is_inhibited("B") is False
        assert state.is_inhibited("C") is True
