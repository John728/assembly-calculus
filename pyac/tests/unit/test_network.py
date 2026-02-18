from __future__ import annotations

import importlib

import numpy as np

from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec


network_mod = importlib.import_module("pyac.core.network")
Network = network_mod.Network


def _two_area_spec() -> NetworkSpec:
    return NetworkSpec(
        areas=[
            AreaSpec("A", n=40, k=4, dynamics_type="feedforward"),
            AreaSpec("B", n=40, k=4, dynamics_type="feedforward"),
        ],
        fibers=[FiberSpec("A", "B", 0.2)],
        beta=0.1,
    )


def _stimulus(area_n: int, hot: int) -> np.ndarray:
    stim = np.zeros(area_n, dtype=np.float64)
    stim[:hot] = 1.0
    return stim


class TestNetworkConstruction:
    def test_builds_connectivity_weights_and_area_state(self):
        spec = _two_area_spec()
        net = Network(spec, make_rng(123))

        assert net.spec is spec
        assert set(net.connectivity.keys()) == {("A", "B")}
        assert set(net.weights.keys()) == {("A", "B")}
        assert "A" in net.strategies
        assert "B" in net.strategies
        assert net.activations["A"].dtype == np.int64
        assert net.activations["B"].dtype == np.int64
        assert net.activations["A"].size == 0
        assert net.activations["B"].size == 0
        assert net.step_count == 0


class TestNetworkStep:
    def test_single_step_with_external_stimulus_produces_k_sized_assemblies(self):
        spec = _two_area_spec()
        net = Network(spec, make_rng(42))

        result = net.step({"A": _stimulus(40, 8)})

        assert result.step == 1
        assert set(result.assemblies.keys()) == {"A", "B"}
        assert len(result.assemblies["A"].indices) == 4
        assert len(result.assemblies["B"].indices) == 4

    def test_step_without_external_stimulus_still_selects_k(self):
        spec = _two_area_spec()
        net = Network(spec, make_rng(11))

        result = net.step()

        assert len(result.assemblies["A"].indices) == 4
        assert len(result.assemblies["B"].indices) == 4

    def test_plasticity_toggle_controls_weight_updates(self):
        spec = _two_area_spec()
        net_off = Network(spec, make_rng(9))
        before_off = net_off.weights[("A", "B")].copy()
        net_off.step({"A": _stimulus(40, 8)}, plasticity_on=False)
        assert (net_off.weights[("A", "B")] - before_off).nnz == 0

        net_on = Network(spec, make_rng(9))
        before_on = net_on.weights[("A", "B")].copy()
        net_on.step({"A": _stimulus(40, 8)}, plasticity_on=True)
        assert (net_on.weights[("A", "B")] - before_on).nnz > 0

    def test_inhibit_and_disinhibit_control_area_updates(self):
        spec = _two_area_spec()
        net = Network(spec, make_rng(202))
        initial_b = net.activations["B"].copy()

        net.inhibit("B")
        result_inhibited = net.step({"A": _stimulus(40, 8)})
        assert result_inhibited.assemblies["B"].indices.size == 0
        assert np.array_equal(net.activations["B"], initial_b)

        net.disinhibit("B")
        result_disinhibited = net.step({"A": _stimulus(40, 8)})
        assert len(result_disinhibited.assemblies["B"].indices) == 4


class TestNetworkRolloutAndHelpers:
    def test_rollout_records_requested_number_of_steps(self):
        spec = _two_area_spec()
        net = Network(spec, make_rng(777))

        trace = net.rollout(
            stimulus_schedule=[{"A": _stimulus(40, 8)}],
            t_internal=10,
            trace_spec=object(),
        )

        assert trace is not None
        assert len(trace.steps) == 10
        assert trace.steps[0].step == 1
        assert trace.steps[-1].step == 10

    def test_determinism_same_seed_same_assemblies(self):
        spec = _two_area_spec()
        stim = {"A": _stimulus(40, 8)}

        net_a = Network(spec, make_rng(31415))
        net_b = Network(spec, make_rng(31415))

        res_a = net_a.step(stim)
        res_b = net_b.step(stim)

        np.testing.assert_array_equal(
            res_a.assemblies["A"].indices,
            res_b.assemblies["A"].indices,
        )
        np.testing.assert_array_equal(
            res_a.assemblies["B"].indices,
            res_b.assemblies["B"].indices,
        )

    def test_get_assembly_returns_current_area_assembly(self):
        spec = _two_area_spec()
        net = Network(spec, make_rng(1234))

        net.step({"A": _stimulus(40, 8)})
        assembly = net.get_assembly("A")

        np.testing.assert_array_equal(assembly.indices, net.activations["A"])
