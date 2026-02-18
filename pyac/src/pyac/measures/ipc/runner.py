from __future__ import annotations

from importlib import import_module
import numpy as np
from numpy.random import Generator
from typing import Protocol


class _AreaLike(Protocol):
    name: str
    n: int


class _SpecLike(Protocol):
    areas: list[_AreaLike]


class _NetworkLike(Protocol):
    spec: _SpecLike
    activations: dict[str, np.ndarray]

    def step(self, external_stimuli: dict[str, np.ndarray] | None = None) -> object:
        ...


def _get_area_size(network: _NetworkLike, area_name: str) -> int:
    for area in network.spec.areas:
        if area.name == area_name:
            return int(area.n)
    raise ValueError(f"unknown area: {area_name}")


def _get_input_area(network: _NetworkLike, area_name: str) -> str:
    for area in network.spec.areas:
        if area.name != area_name:
            return str(area.name)
    return str(area_name)


def run_ipc(
    network: _NetworkLike,
    area_name: str,
    max_degree: int = 3,
    max_delay: int = 10,
    signal_length: int = 1000,
    rng: Generator | None = None,
) -> dict[str, object]:
    if rng is None:
        raise ValueError("rng must be provided")

    basis_module = import_module("pyac.measures.ipc.basis")
    readout_module = import_module("pyac.measures.ipc.readout")
    generate_input_signal = basis_module.generate_input_signal
    generate_targets = basis_module.generate_targets
    fit_readout = readout_module.fit_readout

    n_area = _get_area_size(network, area_name)
    input_area = _get_input_area(network, area_name)
    input_signal = generate_input_signal(signal_length, rng)

    states = np.zeros((signal_length, n_area), dtype=np.float64)
    for t in range(signal_length):
        network.step(external_stimuli={input_area: input_signal[t]})
        active = np.asarray(network.activations[area_name], dtype=np.int64)
        if active.size > 0:
            states[t, active] = 1.0

    target_specs = generate_targets(input_signal, max_degree=max_degree, max_delay=max_delay)

    per_target_capacities: list[float] = []
    degree_breakdown: dict[int, float] = {}

    for degrees, delays, target in target_specs:
        delay = max(delays) if delays else 0
        target_arr = np.asarray(target, dtype=np.float64)
        state_slice = states[delay : delay + len(target_arr)]
        _, capacity = fit_readout(state_slice, target_arr, alpha=1.0)
        capacity = float(max(capacity, 0.0))

        per_target_capacities.append(capacity)
        total_degree = int(sum(degrees))
        degree_breakdown[total_degree] = degree_breakdown.get(total_degree, 0.0) + capacity

    total_capacity = float(sum(per_target_capacities))
    if total_capacity > float(n_area):
        total_capacity = float(n_area)

    return {
        "total_capacity": total_capacity,
        "per_target_capacities": per_target_capacities,
        "degree_breakdown": degree_breakdown,
    }
