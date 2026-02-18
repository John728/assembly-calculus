from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyac.core.types import Assembly

if TYPE_CHECKING:
    from pyac.core.network import Network


def _snapshot_inhibition_state(network: Network) -> dict[str, bool]:
    return {
        area_name: network.inhibition_state.is_inhibited(area_name)
        for area_name in network.area_names
    }


def _restore_inhibition_state(network: Network, state: dict[str, bool]) -> None:
    for area_name in network.area_names:
        if state.get(area_name, False):
            network.inhibit(area_name)
        else:
            network.disinhibit(area_name)


def _set_active_areas(network: Network, active_areas: set[str]) -> None:
    for area_name in network.area_names:
        if area_name in active_areas:
            network.disinhibit(area_name)
        else:
            network.inhibit(area_name)


def _validate_area_name(network: Network, area_name: str) -> None:
    if area_name not in network.area_names:
        raise ValueError(f"unknown area name: {area_name}")


def _validate_t_internal(t_internal: int) -> None:
    if t_internal < 0:
        raise ValueError("t_internal must be >= 0")


def _validate_assembly_in_area(assembly: Assembly, area_name: str) -> None:
    if assembly.area_name != area_name:
        raise ValueError(
            f"assembly belongs to area '{assembly.area_name}', expected '{area_name}'"
        )


def project(
    network: Network,
    src_area: str,
    dst_area: str,
    stimulus: np.ndarray | None = None,
    t_internal: int = 10,
    plasticity_on: bool = True,
    clamp_src: bool = True,
) -> Assembly:
    _validate_area_name(network, src_area)
    _validate_area_name(network, dst_area)
    _validate_t_internal(t_internal)

    saved_inhibition_state = _snapshot_inhibition_state(network)

    clamped_src_assembly = network.activations[src_area].copy() if clamp_src else None

    try:
        _set_active_areas(network, {src_area, dst_area})

        for step_idx in range(t_internal):
            external_stimuli = None
            if step_idx == 0 and stimulus is not None:
                external_stimuli = {src_area: np.asarray(stimulus, dtype=np.float64)}

            _ = network.step(external_stimuli=external_stimuli, plasticity_on=plasticity_on)

            if clamp_src:
                if clamped_src_assembly is None or clamped_src_assembly.size == 0:
                    clamped_src_assembly = network.activations[src_area].copy()
                network.activations[src_area] = clamped_src_assembly.copy()

        return network.get_assembly(dst_area)
    finally:
        _restore_inhibition_state(network, saved_inhibition_state)


def reciprocal_project(
    network: Network,
    area_a: str,
    area_b: str,
    stimulus_a: np.ndarray | None = None,
    stimulus_b: np.ndarray | None = None,
    t_internal: int = 10,
    plasticity_on: bool = True,
) -> tuple[Assembly, Assembly]:
    _validate_area_name(network, area_a)
    _validate_area_name(network, area_b)
    _validate_t_internal(t_internal)

    saved_inhibition_state = _snapshot_inhibition_state(network)

    try:
        _set_active_areas(network, {area_a, area_b})

        for step_idx in range(t_internal):
            external_stimuli: dict[str, np.ndarray] | None = None
            if step_idx == 0:
                external_stimuli = {}
                if stimulus_a is not None:
                    external_stimuli[area_a] = np.asarray(stimulus_a, dtype=np.float64)
                if stimulus_b is not None:
                    external_stimuli[area_b] = np.asarray(stimulus_b, dtype=np.float64)
                if not external_stimuli:
                    external_stimuli = None

            _ = network.step(external_stimuli=external_stimuli, plasticity_on=plasticity_on)

        return network.get_assembly(area_a), network.get_assembly(area_b)
    finally:
        _restore_inhibition_state(network, saved_inhibition_state)


def associate(
    network: Network,
    area: str,
    assembly_x: Assembly,
    assembly_y: Assembly,
    t_internal: int = 10,
    plasticity_on: bool = True,
) -> None:
    _validate_area_name(network, area)
    _validate_t_internal(t_internal)
    _validate_assembly_in_area(assembly_x, area)
    _validate_assembly_in_area(assembly_y, area)

    saved_inhibition_state = _snapshot_inhibition_state(network)

    try:
        _set_active_areas(network, {area})

        for step_idx in range(t_internal):
            current_assembly = assembly_x if (step_idx % 2 == 0) else assembly_y
            network.activations[area] = current_assembly.indices.copy()
            _ = network.step(plasticity_on=plasticity_on)
            network.activations[area] = current_assembly.indices.copy()
    finally:
        _restore_inhibition_state(network, saved_inhibition_state)


def merge(
    network: Network,
    src_area: str,
    assembly_x: Assembly,
    assembly_y: Assembly,
    dst_area: str,
    t_internal: int = 10,
    plasticity_on: bool = True,
) -> Assembly:
    _validate_area_name(network, src_area)
    _validate_area_name(network, dst_area)
    _validate_t_internal(t_internal)
    _validate_assembly_in_area(assembly_x, src_area)
    _validate_assembly_in_area(assembly_y, src_area)

    saved_inhibition_state = _snapshot_inhibition_state(network)
    merged_src_indices = np.union1d(assembly_x.indices, assembly_y.indices).astype(np.int64, copy=False)

    try:
        _set_active_areas(network, {src_area, dst_area})

        for _ in range(t_internal):
            network.activations[src_area] = merged_src_indices.copy()
            _ = network.step(plasticity_on=plasticity_on)
            network.activations[src_area] = merged_src_indices.copy()

        return network.get_assembly(dst_area)
    finally:
        _restore_inhibition_state(network, saved_inhibition_state)
