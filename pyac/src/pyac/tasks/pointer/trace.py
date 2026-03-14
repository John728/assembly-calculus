from __future__ import annotations

from typing import Any

import numpy as np

from pyac.core.network import Network
from pyac.tasks.pointer.data import follow_pointer
from pyac.tasks.pointer.protocol import PointerTask, _decode_node, _reset_network, _stimulus_from_indices
from pyac.tasks.pointer.unseen_protocol import (
    UnseenPointerTask,
    _clear_activations,
    _decode_node as _decode_unseen_node,
    write_list_episode,
)


def _active_assembly_labels(task: PointerTask, active_indices: np.ndarray) -> list[str]:
    labels: list[str] = []
    active_set = set(int(index) for index in active_indices.tolist())
    for (list_idx, node_idx), assembly in task.state_assemblies.items():
        if any(int(index) in active_set for index in assembly.indices.tolist()):
            labels.append(f"L{list_idx}:N{node_idx}")
    return labels


def _assembly_weight_matrix(network: Network, task: PointerTask) -> dict[str, Any]:
    state_area = task.area_map["state"]
    weights = network.weights[(state_area, state_area)]
    labels: list[str] = []
    matrix = np.zeros((task.num_tokens, task.num_tokens), dtype=np.float64)

    ordered_keys = sorted(task.state_assemblies.keys())
    for row_idx, src_key in enumerate(ordered_keys):
        src_assembly = task.state_assemblies[src_key]
        labels.append(f"L{src_key[0]}:N{src_key[1]}")
        for col_idx, dst_key in enumerate(ordered_keys):
            dst_assembly = task.state_assemblies[dst_key]
            submatrix = weights[np.ix_(src_assembly.indices, dst_assembly.indices)]
            matrix[row_idx, col_idx] = float(submatrix.sum())

    return {"labels": labels, "values": matrix.tolist()}


def _assembly_spans(task: PointerTask) -> list[dict[str, int | str]]:
    spans: list[dict[str, int | str]] = []
    for list_idx, node_idx in sorted(task.state_assemblies.keys()):
        assembly = task.state_assemblies[(list_idx, node_idx)]
        spans.append(
            {
                "label": f"L{list_idx}:N{node_idx}",
                "list_idx": int(list_idx),
                "node_idx": int(node_idx),
                "start": int(np.min(assembly.indices)),
                "end": int(np.max(assembly.indices)),
            }
        )
    return spans


def _active_unseen_labels(task: UnseenPointerTask, area_name: str, active_indices: np.ndarray) -> list[str]:
    labels: list[str] = []
    active_set = set(int(index) for index in active_indices.tolist())
    for node_idx, assembly in task.node_assemblies[area_name].items():
        if any(int(index) in active_set for index in assembly.indices.tolist()):
            labels.append(f"N{node_idx}")
    return labels


def _unseen_assembly_weight_matrix(network: Network, task: UnseenPointerTask) -> dict[str, Any]:
    key_area, value_area = task.episodic_fiber
    weights = network.weights[task.episodic_fiber]
    labels = [f"N{node_idx}" for node_idx in sorted(task.node_assemblies[key_area].keys())]
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)

    for src_node in sorted(task.node_assemblies[key_area].keys()):
        src_assembly = task.node_assemblies[key_area][src_node]
        for dst_node in sorted(task.node_assemblies[value_area].keys()):
            dst_assembly = task.node_assemblies[value_area][dst_node]
            submatrix = weights[np.ix_(src_assembly.indices, dst_assembly.indices)]
            matrix[src_node, dst_node] = float(submatrix.sum())

    return {"labels": labels, "values": matrix.tolist()}


def _unseen_assembly_spans(task: UnseenPointerTask) -> list[dict[str, int | str]]:
    spans: list[dict[str, int | str]] = []
    current_area = task.area_map["current"]
    for node_idx, assembly in sorted(task.node_assemblies[current_area].items()):
        spans.append(
            {
                "label": f"N{node_idx}",
                "list_idx": 0,
                "node_idx": int(node_idx),
                "start": int(np.min(assembly.indices)),
                "end": int(np.max(assembly.indices)),
            }
        )
    return spans


def _record_unseen_rollout_trace(
    network: Network,
    task: UnseenPointerTask,
    lists: list[np.ndarray],
    *,
    list_idx: int,
    start_node: int,
    hops: int,
) -> dict[str, Any]:
    pointer = np.asarray(lists[list_idx], dtype=np.int64)
    write_list_episode(network, task, pointer)

    current_area = task.area_map["current"]
    readout_area = task.area_map["readout"]
    current_assembly = task.node_assemblies[current_area][start_node]
    current_stimulus = np.zeros(network.areas_by_name[current_area].n, dtype=np.float64)
    current_stimulus[current_assembly.indices] = 10.0

    _clear_activations(network)
    network.inhibit(readout_area)

    steps: list[dict[str, Any]] = []
    network.step(external_stimuli={current_area: current_stimulus}, plasticity_on=False)
    active = network.activations[current_area].copy()
    steps.append(
        {
            "time": 0,
            "active_neurons": active.tolist(),
            "active_assemblies": _active_unseen_labels(task, current_area, active),
        }
    )

    for hop_idx in range(hops):
        network.step(plasticity_on=False)
        active = network.activations[current_area].copy()
        steps.append(
            {
                "time": hop_idx + 1,
                "active_neurons": active.tolist(),
                "active_assemblies": _active_unseen_labels(task, current_area, active),
            }
        )

    network.disinhibit(readout_area)
    final_prediction = _decode_unseen_node(task, current_area, network.activations[current_area])
    target_node = follow_pointer(pointer, start=start_node, hops=hops)
    expected_edges = [{"src": f"N{src}", "dst": f"N{int(dst)}"} for src, dst in enumerate(pointer.tolist())]
    rollout_path_labels = [step["active_assemblies"][0] for step in steps if step["active_assemblies"]]

    return {
        "list_idx": int(list_idx),
        "start_node": int(start_node),
        "hops": int(hops),
        "pointer": pointer.tolist(),
        "steps": steps,
        "rollout_path_labels": rollout_path_labels,
        "final_prediction": int(final_prediction),
        "target_node": int(target_node),
        "expected_edges": expected_edges,
        "assembly_spans": _unseen_assembly_spans(task),
        "assembly_weight_matrix": _unseen_assembly_weight_matrix(network, task),
    }


def record_rollout_trace(
    network: Network,
    task: PointerTask | UnseenPointerTask,
    lists: list[np.ndarray],
    *,
    list_idx: int,
    start_node: int,
    hops: int,
    settle_steps: int = 1,
) -> dict[str, Any]:
    if isinstance(task, UnseenPointerTask):
        return _record_unseen_rollout_trace(
            network,
            task,
            lists,
            list_idx=list_idx,
            start_node=start_node,
            hops=hops,
        )

    input_area = task.area_map["input"]
    state_area = task.area_map["state"]
    key = task.key_for(list_idx, start_node)
    start_assembly = task.state_assemblies[key]
    state_stimulus = _stimulus_from_indices(network.areas_by_name[state_area].n, start_assembly.indices)

    _reset_network(network)
    network.inhibit(input_area)

    steps: list[dict[str, Any]] = []
    for step_idx in range(settle_steps):
        ext = {state_area: state_stimulus} if step_idx == 0 else None
        network.step(external_stimuli=ext, plasticity_on=False)
        active = network.activations[state_area].copy()
        steps.append(
            {
                "time": step_idx,
                "active_neurons": active.tolist(),
                "active_assemblies": _active_assembly_labels(task, active),
            }
        )

    for hop_idx in range(hops):
        network.step(plasticity_on=False)
        active = network.activations[state_area].copy()
        steps.append(
            {
                "time": settle_steps + hop_idx,
                "active_neurons": active.tolist(),
                "active_assemblies": _active_assembly_labels(task, active),
            }
        )

    network.disinhibit(input_area)
    final_assembly = network.get_assembly(state_area)
    final_prediction = _decode_node(task, list_idx, final_assembly)
    pointer = np.asarray(lists[list_idx], dtype=np.int64)
    target_node = follow_pointer(pointer, start=start_node, hops=hops)
    expected_edges = [{"src": f"L{list_idx}:N{src}", "dst": f"L{list_idx}:N{int(dst)}"} for src, dst in enumerate(pointer.tolist())]
    rollout_path_labels = [step["active_assemblies"][0] for step in steps if step["active_assemblies"]]

    return {
        "list_idx": list_idx,
        "start_node": start_node,
        "hops": hops,
        "pointer": pointer.tolist(),
        "steps": steps,
        "rollout_path_labels": rollout_path_labels,
        "final_prediction": int(final_prediction),
        "target_node": int(target_node),
        "expected_edges": expected_edges,
        "assembly_spans": _assembly_spans(task),
        "assembly_weight_matrix": _assembly_weight_matrix(network, task),
    }
