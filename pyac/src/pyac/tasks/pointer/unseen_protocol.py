from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pyac.core.network import Network
from pyac.core.types import AreaSpec, Assembly, FiberSpec, NetworkSpec
from pyac.measures.overlap import assembly_intersection_size

from pyac.tasks.pointer.data import follow_pointer


@dataclass
class UnseenPointerTask:
    list_length: int
    assembly_size: int
    area_map: dict[str, str]
    node_assemblies: dict[str, dict[int, Assembly]]
    episodic_fiber: tuple[str, str]
    episodic_baseline: Any


def _area_assemblies(area_name: str, list_length: int, assembly_size: int) -> dict[int, Assembly]:
    assemblies: dict[int, Assembly] = {}
    for node_idx in range(list_length):
        start = node_idx * assembly_size
        indices = np.arange(start, start + assembly_size, dtype=np.int64)
        assemblies[node_idx] = Assembly(area_name=area_name, indices=indices)
    return assemblies


def _set_identity_projection(network: Network, src_assemblies: dict[int, Assembly], dst_assemblies: dict[int, Assembly], fiber: tuple[str, str], strength: float = 1.0) -> None:
    weights = network.weights[fiber]
    weights.data[:] = 0.0
    for node_idx, src_assembly in src_assemblies.items():
        dst_assembly = dst_assemblies[node_idx]
        rows = np.repeat(src_assembly.indices, len(dst_assembly.indices))
        cols = np.tile(dst_assembly.indices, len(src_assembly.indices))
        weights[rows, cols] = strength
    weights.eliminate_zeros()


def build_unseen_pointer_network(
    *,
    list_length: int,
    assembly_size: int = 12,
    density: float = 0.35,
    plasticity: float = 0.2,
    rng: np.random.Generator,
) -> tuple[Network, UnseenPointerTask]:
    area_n = list_length * assembly_size
    spec = NetworkSpec(
        areas=[
            AreaSpec(name="current", n=area_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="key", n=area_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="value", n=area_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="readout", n=area_n, k=assembly_size, dynamics_type="feedforward"),
        ],
        fibers=[
            FiberSpec(src="current", dst="key", p_fiber=1.0),
            FiberSpec(src="key", dst="value", p_fiber=1.0),
            FiberSpec(src="value", dst="current", p_fiber=1.0),
            FiberSpec(src="value", dst="readout", p_fiber=1.0),
        ],
        beta=plasticity,
    )
    network = Network(spec, rng)

    task = UnseenPointerTask(
        list_length=list_length,
        assembly_size=assembly_size,
        area_map={"current": "current", "key": "key", "value": "value", "readout": "readout"},
        node_assemblies={
            area_name: _area_assemblies(area_name, list_length, assembly_size)
            for area_name in ["current", "key", "value", "readout"]
        },
        episodic_fiber=("key", "value"),
        episodic_baseline=network.weights[("key", "value")].copy(),
    )
    _set_identity_projection(network, task.node_assemblies["current"], task.node_assemblies["key"], ("current", "key"))
    _set_identity_projection(network, task.node_assemblies["value"], task.node_assemblies["current"], ("value", "current"))
    _set_identity_projection(network, task.node_assemblies["value"], task.node_assemblies["readout"], ("value", "readout"))
    task.episodic_baseline = network.weights[("key", "value")].copy()
    reset_episode_memory(network, task)
    return network, task


def reset_episode_memory(network: Network, task: UnseenPointerTask) -> None:
    weights = network.weights[task.episodic_fiber]
    baseline = task.episodic_baseline.copy()
    weights.data = baseline.data.copy()
    weights.indices = baseline.indices.copy()
    weights.indptr = baseline.indptr.copy()
    for area_name in network.area_names:
        network.activations[area_name] = np.array([], dtype=np.int64)


def episodic_binding_mass(network: Network, task: UnseenPointerTask) -> float:
    return float(network.weights[task.episodic_fiber].sum() - task.episodic_baseline.sum())


def write_list_episode(
    network: Network,
    task: UnseenPointerTask,
    pointer: np.ndarray,
    write_rounds: int = 1,
    binding_strength: float = 10.0,
) -> dict[str, int | str]:
    reset_episode_memory(network, task)
    current_area = task.area_map["current"]
    key_area = task.area_map["key"]
    value_area = task.area_map["value"]
    readout_area = task.area_map["readout"]
    network.inhibit(current_area)
    network.inhibit(readout_area)
    write_steps = 0
    for _ in range(write_rounds):
        for src_node, dst_node in enumerate(pointer.tolist()):
            for area_name in network.area_names:
                network.activations[area_name] = np.array([], dtype=np.int64)
            src_assembly = task.node_assemblies[key_area][src_node]
            dst_assembly = task.node_assemblies[value_area][int(dst_node)]
            key_stimulus = np.zeros(network.areas_by_name[key_area].n, dtype=np.float64)
            value_stimulus = np.zeros(network.areas_by_name[value_area].n, dtype=np.float64)
            key_stimulus[src_assembly.indices] = binding_strength
            value_stimulus[dst_assembly.indices] = binding_strength
            network.step(external_stimuli={key_area: key_stimulus, value_area: value_stimulus}, plasticity_on=True)
            write_steps += 1
    network.disinhibit(current_area)
    network.disinhibit(readout_area)
    return {"write_mode": "plastic", "write_steps": write_steps}


def _decode_node(task: UnseenPointerTask, area_name: str, active_indices: np.ndarray) -> int:
    assembly = Assembly(area_name=area_name, indices=active_indices)
    best_node = 0
    best_score = -1
    for node_idx, prototype in task.node_assemblies[area_name].items():
        score = assembly_intersection_size(assembly, prototype)
        if score > best_score:
            best_score = score
            best_node = node_idx
    return int(best_node)


def _clear_activations(network: Network) -> None:
    for area_name in network.area_names:
        network.activations[area_name] = np.array([], dtype=np.int64)


def query_one_hop(network: Network, task: UnseenPointerTask, pointer: np.ndarray, start_node: int) -> int:
    write_list_episode(network, task, pointer)
    return query_written_episode(network, task, start_node)


def query_written_episode(network: Network, task: UnseenPointerTask, start_node: int) -> int:
    _clear_activations(network)
    current_area = task.area_map["current"]
    key_area = task.area_map["key"]
    value_area = task.area_map["value"]
    readout_area = task.area_map["readout"]
    key_assembly = task.node_assemblies[key_area][start_node]
    key_stimulus = np.zeros(network.areas_by_name[key_area].n, dtype=np.float64)
    key_stimulus[key_assembly.indices] = 10.0
    network.inhibit(current_area)
    network.inhibit(readout_area)
    network.step(external_stimuli={key_area: key_stimulus}, plasticity_on=False)
    network.disinhibit(current_area)
    network.disinhibit(readout_area)
    return _decode_node(task, value_area, network.activations[value_area])


def rollout_unseen_pointer(
    network: Network,
    task: UnseenPointerTask,
    pointer: np.ndarray,
    *,
    start_node: int,
    hops: int,
    internal_steps: int | None = None,
) -> dict[str, object]:
    write_stats = write_list_episode(network, task, pointer)
    current_area = task.area_map["current"]
    readout_area = task.area_map["readout"]
    current_assembly = task.node_assemblies[current_area][start_node]
    current_stimulus = np.zeros(network.areas_by_name[current_area].n, dtype=np.float64)
    current_stimulus[current_assembly.indices] = 10.0

    _clear_activations(network)
    network.inhibit(readout_area)
    current_state_nodes = [int(start_node)]
    network.step(external_stimuli={current_area: current_stimulus}, plasticity_on=False)
    rollout_steps = int(internal_steps) if internal_steps is not None else int(hops)
    for _ in range(rollout_steps):
        network.step(plasticity_on=False)
        decoded = _decode_node(task, current_area, network.activations[current_area])
        current_state_nodes.append(int(decoded))
    network.disinhibit(readout_area)

    intermediate_nodes = current_state_nodes[1:]
    current_node = intermediate_nodes[-1] if intermediate_nodes else int(start_node)
    return {
        "start_node": int(start_node),
        "hops": int(hops),
        "intermediate_nodes": intermediate_nodes,
        "current_state_nodes": current_state_nodes,
        "final_prediction": int(current_node),
        "external_cue_count": 1,
        "internal_steps": rollout_steps,
        "write_mode": write_stats["write_mode"],
        "write_steps": write_stats["write_steps"],
    }


def evaluate_unseen_rollout(
    network: Network,
    task: UnseenPointerTask,
    lists: list[np.ndarray],
    *,
    hops: int,
    internal_steps: int | None,
    samples_per_list: int,
    rng: np.random.Generator,
) -> float:
    correct = 0
    total = 0
    for pointer in lists:
        pointer_arr = np.asarray(pointer, dtype=np.int64)
        for _ in range(samples_per_list):
            start_node = int(rng.integers(0, task.list_length))
            trace = rollout_unseen_pointer(network, task, pointer_arr, start_node=start_node, hops=hops, internal_steps=internal_steps)
            target = follow_pointer(pointer_arr, start=start_node, hops=hops)
            final_prediction = trace["final_prediction"]
            if not isinstance(final_prediction, int):
                raise TypeError(f"Expected int final_prediction, got {type(final_prediction).__name__}")
            correct += int(final_prediction == target)
            total += 1
    return correct / max(total, 1)


def evaluate_unseen_one_hop(
    network: Network,
    task: UnseenPointerTask,
    lists: list[np.ndarray],
    samples_per_list: int,
    rng: np.random.Generator,
) -> float:
    correct = 0
    total = 0
    for pointer in lists:
        pointer_arr = np.asarray(pointer, dtype=np.int64)
        for _ in range(samples_per_list):
            start_node = int(rng.integers(0, task.list_length))
            prediction = query_one_hop(network, task, pointer_arr, start_node)
            target = follow_pointer(pointer_arr, start=start_node, hops=1)
            correct += int(prediction == target)
            total += 1
    return correct / max(total, 1)
