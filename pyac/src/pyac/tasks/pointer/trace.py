from __future__ import annotations

from typing import Any

import numpy as np

from pyac.core.network import Network
from pyac.tasks.pointer.data import follow_pointer
from pyac.tasks.pointer.protocol import PointerTask, _decode_node, _reset_network, _stimulus_from_indices


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


def record_rollout_trace(
    network: Network,
    task: PointerTask,
    lists: list[np.ndarray],
    *,
    list_idx: int,
    start_node: int,
    hops: int,
    settle_steps: int = 1,
) -> dict[str, Any]:
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
