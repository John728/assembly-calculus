from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec


Network = importlib.import_module("pyac.core.network").Network


@dataclass
class _TraceSpec:
    areas: list[str] | None = None


def _hot(n: int, k: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    v[:k] = 1.0
    return v


class TestNetworkIntegration:
    def test_three_area_chain_orchestrates_all_areas(self):
        spec = NetworkSpec(
            areas=[
                AreaSpec("A", n=60, k=6),
                AreaSpec("B", n=60, k=6),
                AreaSpec("C", n=60, k=6),
            ],
            fibers=[
                FiberSpec("A", "B", 0.15),
                FiberSpec("B", "C", 0.15),
            ],
            beta=0.1,
        )
        net = Network(spec, make_rng(10))

        result = net.step({"A": _hot(60, 12)})

        assert len(result.assemblies["A"].indices) == 6
        assert len(result.assemblies["B"].indices) == 6
        assert len(result.assemblies["C"].indices) == 6

    def test_recurrent_dynamics_contributes_after_first_step(self):
        spec = NetworkSpec(
            areas=[AreaSpec("R", n=50, k=5, p_recurrent=0.2, dynamics_type="recurrent")],
            fibers=[],
            beta=0.2,
        )
        net = Network(spec, make_rng(22))

        net.step({"R": _hot(50, 10)})
        contrib = net.strategies["R"].dynamics_contribution(
            area_name="R",
            activations=net.activations["R"],
            context={"step": net.step_count},
        )

        assert np.any(contrib > 0.0)

    def test_refracted_dynamics_accumulates_negative_bias(self):
        spec = NetworkSpec(
            areas=[AreaSpec("X", n=40, k=4, dynamics_type="refracted")],
            fibers=[],
            beta=0.3,
        )
        net = Network(spec, make_rng(23))

        net.step({"X": _hot(40, 8)}, plasticity_on=True)
        contribution = net.strategies["X"].dynamics_contribution(
            area_name="X",
            activations=net.activations["X"],
            context={"step": net.step_count},
        )

        assert np.any(contribution < 0.0)

    def test_multi_step_plasticity_increases_coactivated_weights(self):
        spec = NetworkSpec(
            areas=[AreaSpec("A", n=50, k=5), AreaSpec("B", n=50, k=5)],
            fibers=[FiberSpec("A", "B", 0.2)],
            beta=0.2,
        )
        net = Network(spec, make_rng(7))

        key = ("A", "B")
        initial_data = net.weights[key].data.copy()
        for _ in range(8):
            net.step({"A": _hot(50, 10)}, plasticity_on=True)

        assert np.any(net.weights[key].data > initial_data)

    def test_trace_filter_records_only_requested_areas(self):
        spec = NetworkSpec(
            areas=[AreaSpec("A", n=30, k=3), AreaSpec("B", n=30, k=3)],
            fibers=[FiberSpec("A", "B", 0.2)],
            beta=0.1,
        )
        net = Network(spec, make_rng(123))

        trace = net.rollout(
            stimulus_schedule=[{"A": _hot(30, 6)}],
            t_internal=5,
            plasticity_on=False,
            trace_spec=_TraceSpec(areas=["A"]),
        )

        assert trace is not None
        assert len(trace.steps) == 5
        for step_result in trace.steps:
            assert set(step_result.assemblies.keys()) == {"A"}
