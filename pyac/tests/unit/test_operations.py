from __future__ import annotations

import importlib

import numpy as np

from pyac.core.rng import make_rng
from pyac.core.types import Assembly
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec


network_mod = importlib.import_module("pyac.core.network")
Network = network_mod.Network
operations_mod = importlib.import_module("pyac.core.operations")
project = operations_mod.project
reciprocal_project = operations_mod.reciprocal_project
associate = operations_mod.associate
merge = operations_mod.merge


def _stimulus(area_n: int, hot: int) -> np.ndarray:
    stim = np.zeros(area_n, dtype=np.float64)
    stim[:hot] = 1.0
    return stim


def _project_spec() -> NetworkSpec:
    return NetworkSpec(
        areas=[
            AreaSpec("A", n=80, k=8, dynamics_type="feedforward"),
            AreaSpec("B", n=80, k=8, dynamics_type="feedforward"),
            AreaSpec("C", n=80, k=8, dynamics_type="feedforward"),
        ],
        fibers=[
            FiberSpec("A", "B", 0.15),
            FiberSpec("B", "A", 0.15),
            FiberSpec("C", "B", 0.15),
        ],
        beta=0.2,
    )


def _reciprocal_spec() -> NetworkSpec:
    return NetworkSpec(
        areas=[
            AreaSpec("A", n=100, k=10, dynamics_type="feedforward"),
            AreaSpec("B", n=100, k=10, dynamics_type="feedforward"),
            AreaSpec("C", n=100, k=10, dynamics_type="feedforward"),
        ],
        fibers=[
            FiberSpec("A", "B", 0.1),
            FiberSpec("B", "A", 0.1),
            FiberSpec("C", "A", 0.1),
        ],
        beta=0.3,
    )


def _associate_spec() -> NetworkSpec:
    return NetworkSpec(
        areas=[AreaSpec("A", n=120, k=12, p_recurrent=1.0, dynamics_type="recurrent")],
        fibers=[],
        beta=0.4,
    )


def _merge_spec() -> NetworkSpec:
    return NetworkSpec(
        areas=[
            AreaSpec("A", n=200, k=20, dynamics_type="feedforward"),
            AreaSpec("B", n=200, k=20, dynamics_type="feedforward"),
            AreaSpec("C", n=200, k=20, dynamics_type="feedforward"),
        ],
        fibers=[
            FiberSpec("A", "B", 0.2),
            FiberSpec("C", "B", 0.2),
        ],
        beta=0.5,
    )


def _probe_area_response(net: Network, area_name: str, cue: Assembly) -> np.ndarray:
    net.activations[area_name] = cue.indices.copy()
    net.step(plasticity_on=False)
    return net.activations[area_name].copy()


class TestProject:
    def test_project_with_stimulus_returns_k_sized_dst_assembly(self):
        spec = _project_spec()
        net = Network(spec, make_rng(42))

        asm = project(
            net,
            src_area="A",
            dst_area="B",
            stimulus=_stimulus(80, 20),
            t_internal=15,
        )

        assert asm.area_name == "B"
        assert len(asm.indices) == 8

    def test_project_without_stimulus_still_returns_k_sized_assembly(self):
        spec = _project_spec()
        net = Network(spec, make_rng(88))

        asm = project(net, src_area="A", dst_area="B", stimulus=None, t_internal=10)

        assert asm.area_name == "B"
        assert len(asm.indices) == 8

    def test_project_is_deterministic_for_same_seed(self):
        spec = _project_spec()
        stim = _stimulus(80, 24)

        asm_a = project(Network(spec, make_rng(1234)), "A", "B", stimulus=stim, t_internal=20)
        asm_b = project(Network(spec, make_rng(1234)), "A", "B", stimulus=stim, t_internal=20)

        np.testing.assert_array_equal(asm_a.indices, asm_b.indices)

    def test_project_restores_inhibition_state(self):
        spec = _project_spec()
        net = Network(spec, make_rng(99))
        net.inhibit("A")
        net.disinhibit("B")
        net.inhibit("C")

        before = {
            name: net.inhibition_state.is_inhibited(name)
            for name in net.area_names
        }

        project(net, "A", "B", stimulus=_stimulus(80, 20), t_internal=8)

        after = {
            name: net.inhibition_state.is_inhibited(name)
            for name in net.area_names
        }
        assert after == before

    def test_project_clamp_src_true_keeps_src_activations_fixed(self):
        spec = _project_spec()
        net = Network(spec, make_rng(555))
        net.activations["A"] = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

        initial = net.activations["A"].copy()
        project(net, "A", "B", stimulus=None, t_internal=12, clamp_src=True)

        np.testing.assert_array_equal(net.activations["A"], initial)

    def test_project_clamp_src_false_allows_src_to_update(self):
        spec = _project_spec()
        net = Network(spec, make_rng(555))
        net.activations["A"] = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

        initial = net.activations["A"].copy()
        project(net, "A", "B", stimulus=None, t_internal=12, clamp_src=False)

        assert not np.array_equal(net.activations["A"], initial)


class TestReciprocalProject:
    def test_reciprocal_project_returns_both_assemblies(self):
        spec = _reciprocal_spec()
        net = Network(spec, make_rng(7))

        asm_a, asm_b = reciprocal_project(
            net,
            area_a="A",
            area_b="B",
            stimulus_a=_stimulus(100, 30),
            stimulus_b=_stimulus(100, 25),
            t_internal=12,
        )

        assert asm_a.area_name == "A"
        assert asm_b.area_name == "B"
        assert len(asm_a.indices) == 10
        assert len(asm_b.indices) == 10

    def test_reciprocal_project_restores_inhibition_state(self):
        spec = _reciprocal_spec()
        net = Network(spec, make_rng(101))
        net.inhibit("A")
        net.inhibit("C")

        before = {
            name: net.inhibition_state.is_inhibited(name)
            for name in net.area_names
        }

        reciprocal_project(
            net,
            area_a="A",
            area_b="B",
            stimulus_a=_stimulus(100, 25),
            stimulus_b=None,
            t_internal=10,
        )

        after = {
            name: net.inhibition_state.is_inhibited(name)
            for name in net.area_names
        }
        assert after == before

    def test_reciprocal_project_enables_bidirectional_weight_updates(self):
        spec = _reciprocal_spec()
        net = Network(spec, make_rng(8080))

        before_ab = net.weights[("A", "B")].copy()
        before_ba = net.weights[("B", "A")].copy()

        reciprocal_project(
            net,
            area_a="A",
            area_b="B",
            stimulus_a=_stimulus(100, 28),
            stimulus_b=_stimulus(100, 24),
            t_internal=15,
            plasticity_on=True,
        )

        assert (net.weights[("A", "B")] - before_ab).nnz > 0
        assert (net.weights[("B", "A")] - before_ba).nnz > 0


class TestAssociate:
    def test_associate_modifies_recurrent_weights(self):
        spec = _associate_spec()
        net = Network(spec, make_rng(2026))
        asm_x = Assembly("A", np.arange(0, 12, dtype=np.int64))
        asm_y = Assembly("A", np.arange(60, 72, dtype=np.int64))

        before = net.strategies["A"].recurrent_weights.copy()
        associate(net, "A", asm_x, asm_y, t_internal=18, plasticity_on=True)

        assert (net.strategies["A"].recurrent_weights - before).nnz > 0

    def test_associate_produces_overlapping_responses(self):
        spec = _associate_spec()
        asm_x = Assembly("A", np.arange(0, 12, dtype=np.int64))
        asm_y = Assembly("A", np.arange(60, 72, dtype=np.int64))

        baseline_net = Network(spec, make_rng(77))
        baseline_x = _probe_area_response(baseline_net, "A", asm_x)
        baseline_y = _probe_area_response(baseline_net, "A", asm_y)
        baseline_overlap = len(np.intersect1d(baseline_x, asm_y.indices)) + len(
            np.intersect1d(baseline_y, asm_x.indices)
        )

        trained_net = Network(spec, make_rng(77))
        associate(trained_net, "A", asm_x, asm_y, t_internal=22, plasticity_on=True)
        trained_x = _probe_area_response(trained_net, "A", asm_x)
        trained_y = _probe_area_response(trained_net, "A", asm_y)
        trained_overlap = len(np.intersect1d(trained_x, asm_y.indices)) + len(
            np.intersect1d(trained_y, asm_x.indices)
        )

        assert trained_overlap > 0
        assert trained_overlap >= baseline_overlap

    def test_associate_restores_inhibition_state(self):
        spec = _merge_spec()
        net = Network(spec, make_rng(500))
        net.inhibit("A")
        net.disinhibit("B")
        net.inhibit("C")

        before = {
            name: net.inhibition_state.is_inhibited(name)
            for name in net.area_names
        }

        asm_x = Assembly("A", np.arange(0, 20, dtype=np.int64))
        asm_y = Assembly("A", np.arange(80, 100, dtype=np.int64))
        associate(net, "A", asm_x, asm_y, t_internal=12)

        after = {
            name: net.inhibition_state.is_inhibited(name)
            for name in net.area_names
        }
        assert after == before


class TestMerge:
    def _build_parent_assemblies(self, net: Network) -> tuple[Assembly, Assembly, Assembly, Assembly]:
        stim_x = np.zeros(200, dtype=np.float64)
        stim_x[:40] = 1.0
        stim_y = np.zeros(200, dtype=np.float64)
        stim_y[100:140] = 1.0

        asm_dst_x = project(net, "A", "B", stimulus=stim_x, t_internal=20)
        asm_src_x = net.get_assembly("A")
        asm_dst_y = project(net, "A", "B", stimulus=stim_y, t_internal=20)
        asm_src_y = net.get_assembly("A")
        return asm_src_x, asm_src_y, asm_dst_x, asm_dst_y

    def test_merge_returns_k_sized_dst_assembly(self):
        spec = _merge_spec()
        net = Network(spec, make_rng(42))
        asm_src_x, asm_src_y, _, _ = self._build_parent_assemblies(net)

        asm_merged = merge(net, "A", asm_src_x, asm_src_y, "B", t_internal=20)

        assert asm_merged.area_name == "B"
        assert len(asm_merged.indices) == 20

    def test_merge_overlaps_both_parent_projections(self):
        spec = _merge_spec()
        net = Network(spec, make_rng(4242))
        asm_src_x, asm_src_y, asm_dst_x, asm_dst_y = self._build_parent_assemblies(net)

        asm_merged = merge(net, "A", asm_src_x, asm_src_y, "B", t_internal=20)

        overlap_x = len(np.intersect1d(asm_merged.indices, asm_dst_x.indices))
        overlap_y = len(np.intersect1d(asm_merged.indices, asm_dst_y.indices))
        assert overlap_x > 0
        assert overlap_y > 0

    def test_merge_is_deterministic(self):
        spec = _merge_spec()

        net_a = Network(spec, make_rng(9001))
        asm_src_x_a, asm_src_y_a, _, _ = self._build_parent_assemblies(net_a)
        merged_a = merge(net_a, "A", asm_src_x_a, asm_src_y_a, "B", t_internal=18)

        net_b = Network(spec, make_rng(9001))
        asm_src_x_b, asm_src_y_b, _, _ = self._build_parent_assemblies(net_b)
        merged_b = merge(net_b, "A", asm_src_x_b, asm_src_y_b, "B", t_internal=18)

        np.testing.assert_array_equal(merged_a.indices, merged_b.indices)

    def test_merge_restores_inhibition_state(self):
        spec = _merge_spec()
        net = Network(spec, make_rng(31337))
        net.inhibit("A")
        net.disinhibit("B")
        net.inhibit("C")

        before = {
            name: net.inhibition_state.is_inhibited(name)
            for name in net.area_names
        }

        asm_src_x, asm_src_y, _, _ = self._build_parent_assemblies(net)
        merge(net, "A", asm_src_x, asm_src_y, "B", t_internal=10)

        after = {
            name: net.inhibition_state.is_inhibited(name)
            for name in net.area_names
        }
        assert after == before
