from __future__ import annotations

import importlib
import numpy as np

from pyac.core.network import Network
from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec

fit_readout = importlib.import_module("pyac.measures.ipc.readout").fit_readout
run_ipc = importlib.import_module("pyac.measures.ipc.runner").run_ipc


class TestFitReadout:
    def test_fit_readout_high_capacity_for_linear_target(self):
        rng = make_rng(123)
        states = rng.normal(size=(300, 8)).astype(np.float64)
        weights = rng.normal(size=8).astype(np.float64)
        targets = states @ weights

        coefficients, capacity = fit_readout(states, targets, alpha=1.0)

        assert coefficients.shape == (8,)
        assert coefficients.dtype == np.float64
        assert 0.98 <= capacity <= 1.0

    def test_fit_readout_constant_target_returns_zero_capacity(self):
        rng = make_rng(11)
        states = rng.normal(size=(100, 5)).astype(np.float64)
        targets = np.ones(100, dtype=np.float64)

        coefficients, capacity = fit_readout(states, targets, alpha=1.0)

        assert coefficients.shape == (5,)
        assert capacity == 0.0

    def test_fit_readout_large_alpha_shrinks_coefficient_norm(self):
        rng = make_rng(222)
        states = rng.normal(size=(200, 6)).astype(np.float64)
        targets = (states @ np.array([1.0, -2.0, 0.5, 0.0, 1.0, -0.5])).astype(np.float64)

        coef_small, _ = fit_readout(states, targets, alpha=1e-6)
        coef_large, _ = fit_readout(states, targets, alpha=1e3)

        assert np.linalg.norm(coef_large) < np.linalg.norm(coef_small)


class _DummySpecArea:
    def __init__(self, name: str, n: int):
        self.name = name
        self.n = n


class _DummySpec:
    def __init__(self, area_name: str, n: int):
        self.areas = [_DummySpecArea(area_name, n)]


class _DummyNetwork:
    def __init__(self, area_name: str, n: int, activation_schedule: list[np.ndarray]):
        self.spec = _DummySpec(area_name, n)
        self.activations: dict[str, np.ndarray] = {area_name: np.array([], dtype=np.int64)}
        self._area_name = area_name
        self._schedule = activation_schedule
        self._idx = 0

    def step(self, external_stimuli: dict[str, np.ndarray] | None = None) -> None:
        _ = external_stimuli
        self.activations[self._area_name] = self._schedule[self._idx]
        self._idx += 1


class TestRunIpc:
    def test_run_ipc_returns_expected_structure_and_bounds(self):
        spec = NetworkSpec(
            areas=[AreaSpec("A", n=30, k=4), AreaSpec("B", n=30, k=4)],
            fibers=[FiberSpec("A", "B", 0.2)],
            beta=0.1,
        )
        net = Network(spec, make_rng(10))

        result = run_ipc(
            net,
            "B",
            max_degree=2,
            max_delay=4,
            signal_length=150,
            rng=make_rng(999),
        )

        assert set(result.keys()) == {"total_capacity", "per_target_capacities", "degree_breakdown"}
        assert isinstance(result["total_capacity"], float)
        assert isinstance(result["per_target_capacities"], list)
        assert isinstance(result["degree_breakdown"], dict)

        assert result["total_capacity"] >= 0.0
        assert result["total_capacity"] <= 30.0
        assert np.isclose(result["total_capacity"], sum(result["per_target_capacities"]))

        for capacity in result["per_target_capacities"]:
            assert isinstance(capacity, float)
            assert capacity >= 0.0

        for degree, degree_capacity in result["degree_breakdown"].items():
            assert isinstance(degree, int)
            assert degree >= 0
            assert degree_capacity >= 0.0

    def test_run_ipc_is_reproducible_with_same_rng_seed(self):
        spec = NetworkSpec(
            areas=[AreaSpec("A", n=24, k=3), AreaSpec("B", n=24, k=3)],
            fibers=[FiberSpec("A", "B", 0.2)],
            beta=0.1,
        )
        net1 = Network(spec, make_rng(1))
        net2 = Network(spec, make_rng(1))

        result1 = run_ipc(net1, "B", max_degree=1, max_delay=3, signal_length=80, rng=make_rng(555))
        result2 = run_ipc(net2, "B", max_degree=1, max_delay=3, signal_length=80, rng=make_rng(555))

        assert np.isclose(result1["total_capacity"], result2["total_capacity"])
        np.testing.assert_allclose(result1["per_target_capacities"], result2["per_target_capacities"])
        assert result1["degree_breakdown"] == result2["degree_breakdown"]

    def test_run_ipc_converts_sparse_indices_to_dense_binary_states(self, monkeypatch):
        schedule = [
            np.array([0, 2], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([0, 1, 3], dtype=np.int64),
            np.array([], dtype=np.int64),
        ]
        net = _DummyNetwork("X", n=4, activation_schedule=schedule)

        monkeypatch.setattr(
            "pyac.measures.ipc.basis.generate_input_signal",
            lambda length, rng: np.linspace(-1.0, 1.0, length, dtype=np.float64),
        )
        monkeypatch.setattr(
            "pyac.measures.ipc.basis.generate_targets",
            lambda input_signal, max_degree, max_delay: [
                ((1,), (), np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64))
            ],
        )

        captured_states: list[np.ndarray] = []

        def _fake_fit_readout(states: np.ndarray, targets: np.ndarray, alpha: float):
            _ = targets
            _ = alpha
            captured_states.append(states.copy())
            return np.zeros(states.shape[1], dtype=np.float64), 0.25

        monkeypatch.setattr("pyac.measures.ipc.readout.fit_readout", _fake_fit_readout)

        result = run_ipc(net, "X", max_degree=1, max_delay=1, signal_length=4, rng=make_rng(9))

        assert len(captured_states) == 1
        expected = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(captured_states[0], expected)
        assert result["total_capacity"] == 0.25
        assert result["degree_breakdown"] == {1: 0.25}
