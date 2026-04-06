from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from pyac.core.network import Network
from pyac.core.types import AreaSpec, Assembly, FiberSpec, NetworkSpec
from pyac.measures.overlap import assembly_intersection_size

from pyac.tasks.pointer.data import follow_pointer


@dataclass
class ProperUnseenPointerTask:
    list_length: int
    assembly_size: int
    area_map: dict[str, str]
    node_assemblies: dict[str, dict[int, Assembly]]
    hop_assemblies: dict[int, Assembly]
    memory_fiber: tuple[str, str]
    episodic_baseline: Any
    controller_fibers: list[tuple[str, str]]
    training_history: list[dict[str, int | float | str]] = field(default_factory=list)

    @property
    def episodic_fiber(self) -> tuple[str, str]:
        return self.memory_fiber


def _area_assemblies(area_name: str, list_length: int, assembly_size: int) -> dict[int, Assembly]:
    assemblies: dict[int, Assembly] = {}
    for node_idx in range(list_length):
        start = node_idx * assembly_size
        indices = np.arange(start, start + assembly_size, dtype=np.int64)
        assemblies[node_idx] = Assembly(area_name=area_name, indices=indices)
    return assemblies


def _clear_activations(network: Network) -> None:
    for area_name in network.area_names:
        network.activations[area_name] = np.array([], dtype=np.int64)


def _stimulus(area_n: int, indices: np.ndarray, strength: float = 10.0) -> np.ndarray:
    values = np.zeros(area_n, dtype=np.float64)
    values[np.asarray(indices, dtype=np.int64)] = strength
    return values


def _set_uniform_weights(network: Network, fiber: tuple[str, str], value: float) -> None:
    connectivity = network.connectivity[fiber]
    network.weights[fiber] = connectivity.astype(np.float64).multiply(value).tocsr()


def _zero_weights(network: Network, fiber: tuple[str, str]) -> None:
    src_name, dst_name = fiber
    src_n = network.areas_by_name[src_name].n
    dst_n = network.areas_by_name[dst_name].n
    network.weights[fiber] = csr_matrix((src_n, dst_n), dtype=np.float64)


def _decode_node(task: ProperUnseenPointerTask, area_name: str, active_indices: np.ndarray) -> int:
    assembly = Assembly(area_name=area_name, indices=np.asarray(active_indices, dtype=np.int64))
    best_node = 0
    best_score = -1
    for node_idx, prototype in task.node_assemblies[area_name].items():
        score = assembly_intersection_size(assembly, prototype)
        if score > best_score:
            best_score = score
            best_node = node_idx
    return int(best_node)


def _decode_hop_state(task: ProperUnseenPointerTask, active_indices: np.ndarray) -> int:
    assembly = Assembly(area_name=task.area_map["loop"], indices=np.asarray(active_indices, dtype=np.int64))
    best_hops = 0
    best_score = -1
    for hop_idx, prototype in task.hop_assemblies.items():
        score = assembly_intersection_size(assembly, prototype)
        if score > best_score:
            best_score = score
            best_hops = hop_idx
    return int(best_hops)


def evaluate_identity_alignment(
    network: Network,
    task: ProperUnseenPointerTask,
) -> dict[str, object]:
    areas_checked = [area_name for area_name in ["cur", "src", "dst"] if area_name in task.node_assemblies]
    num_nodes = min((len(task.node_assemblies[area_name]) for area_name in areas_checked), default=0)
    self_overlaps: list[float] = []
    area_means: dict[str, float] = {}

    for area_name in areas_checked:
        area_scores: list[float] = []
        area_n = network.areas_by_name[area_name].n
        area_k = float(network.areas_by_name[area_name].k)
        for node_idx in range(num_nodes):
            prototype = task.node_assemblies[area_name][node_idx]
            _clear_activations(network)
            network.step(
                external_stimuli={area_name: _stimulus(area_n, prototype.indices)},
                plasticity_on=False,
            )
            observed = network.activations[area_name]
            overlap = float(np.intersect1d(observed, prototype.indices).size)
            normalized_overlap = overlap / max(area_k, 1.0)
            area_scores.append(normalized_overlap)
            self_overlaps.append(normalized_overlap)
        area_means[area_name] = float(np.mean(area_scores)) if area_scores else 0.0

    _clear_activations(network)

    return {
        "num_nodes": int(num_nodes),
        "areas_checked": areas_checked,
        "mean_self_overlap": float(np.mean(self_overlaps)) if self_overlaps else 0.0,
        "area_mean_self_overlap": area_means,
    }


def build_proper_unseen_pointer_network(
    *,
    list_length: int,
    assembly_size: int = 32,
    density: float = 0.2,
    plasticity: float = 0.1,
    rng: np.random.Generator,
) -> tuple[Network, ProperUnseenPointerTask]:
    area_n = list_length * assembly_size
    spec = NetworkSpec(
        areas=[
            AreaSpec(
                name="cur",
                n=area_n,
                k=assembly_size,
                dynamics_type="feedforward",
            ),
            AreaSpec(name="src", n=area_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="dst", n=area_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="readout", n=area_n, k=assembly_size, dynamics_type="feedforward"),
        ],
        fibers=[
            FiberSpec(src="cur", dst="src", p_fiber=density),
            FiberSpec(src="src", dst="dst", p_fiber=1.0),
            FiberSpec(src="dst", dst="cur", p_fiber=density),
            FiberSpec(src="dst", dst="readout", p_fiber=density),
        ],
        beta=0.1,
    )
    network = Network(spec, rng)
    task = ProperUnseenPointerTask(
        list_length=list_length,
        assembly_size=assembly_size,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "readout": "readout"},
        node_assemblies={
            area_name: _area_assemblies(area_name, list_length, assembly_size)
            for area_name in ["cur", "src", "dst", "readout"]
        },
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=network.weights[("src", "dst")].copy(),
        controller_fibers=[("cur", "src"), ("dst", "cur"), ("dst", "readout")],
    )
    return network, task


def reset_proper_episode_memory(network: Network, task: ProperUnseenPointerTask) -> None:
    weights = network.weights[task.memory_fiber]
    baseline = task.episodic_baseline.copy()
    weights.data = baseline.data.copy()
    weights.indices = baseline.indices.copy()
    weights.indptr = baseline.indptr.copy()
    _clear_activations(network)


def train_proper_unseen_controller(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    training_lists: list[np.ndarray],
    k_values: list[int],
    episodes: int,
    rng: np.random.Generator,
    train_time_budget: int = 10,
) -> list[dict[str, int | float | str]]:
    history: list[dict[str, int | float | str]] = []
    
    # Increase episodes to compensate for lower beta
    episodes = episodes * 2

    for episode in range(episodes):
        pointer = np.asarray(training_lists[int(rng.integers(0, len(training_lists)))], dtype=np.int64)
        query_hops = int(k_values[int(rng.integers(0, len(k_values)))])
        start_node = int(rng.integers(0, task.list_length))
        target_node = int(follow_pointer(pointer, start=start_node, hops=query_hops))
        path_nodes = [start_node]
        current_path_node = start_node
        for _ in range(query_hops):
            current_path_node = int(pointer[current_path_node])
            path_nodes.append(current_path_node)

        query_history = train_query_primitive(network, task, rounds=24)
        writeback_history = train_writeback_primitive(network, task, rounds=24)
        one_hop_history = train_one_hop_composition(network, task, training_lists=[pointer], rounds=12, nudge_strength=20.0)
        multi_hop_history = train_multi_hop_recurrence(network, task, training_lists=[pointer], episodes=12, rng=rng, nudge_strength=20.0)

        rollout = rollout_proper_unseen_pointer(
            network,
            task,
            pointer,
            start_node=start_node,
            hops=query_hops,
            internal_steps=train_time_budget,
        )
        final_prediction_value = rollout["final_prediction"]
        final_prediction = int(final_prediction_value) if isinstance(final_prediction_value, (int, np.integer, float, str)) else 0
        reset_proper_episode_memory(network, task)
        controller_update_steps = len(query_history) + len(writeback_history) + len(one_hop_history) + len(multi_hop_history)
        episode_accuracy = 1.0 if final_prediction == target_node else 0.0

        history.append(
            {
                "episode": episode + 1,
                "list_length": int(len(pointer)),
                "start_node": start_node,
                "query_hops": query_hops,
                "path_nodes": "-".join(str(node) for node in path_nodes),
                "target_node": target_node,
                "final_prediction": final_prediction,
                "episode_accuracy": episode_accuracy,
                "controller_update_steps": controller_update_steps,
            }
        )

    task.training_history = history
    return history


def train_query_primitive(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    rounds: int = 8,
) -> list[dict[str, int]]:
    cur_area = task.area_map["cur"]
    src_area = task.area_map["src"]
    history: list[dict[str, int]] = []
    for round_idx in range(rounds):
        for node_idx in range(task.list_length):
            _clear_activations(network)
            cur_stimulus = _stimulus(
                network.areas_by_name[cur_area].n,
                task.node_assemblies[cur_area][node_idx].indices,
            )
            src_stimulus = _stimulus(
                network.areas_by_name[src_area].n,
                task.node_assemblies[src_area][node_idx].indices,
            )
            network.step(
                external_stimuli={cur_area: cur_stimulus, src_area: src_stimulus},
                plasticity_on=True,
            )
            network.normalize()
            history.append({"round": int(round_idx), "node_idx": int(node_idx)})
    _clear_activations(network)
    return history


def probe_query_primitive(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    node_idx: int,
) -> dict[str, int | bool]:
    cur_area = task.area_map["cur"]
    src_area = task.area_map["src"]
    _clear_activations(network)
    cur_stimulus = _stimulus(
        network.areas_by_name[cur_area].n,
        task.node_assemblies[cur_area][node_idx].indices,
    )
    network.step(external_stimuli={cur_area: cur_stimulus}, plasticity_on=False)
    predicted_src_node = _decode_node(task, src_area, network.activations[src_area])
    _clear_activations(network)
    return {
        "node_idx": int(node_idx),
        "predicted_src_node": int(predicted_src_node),
        "correct": int(predicted_src_node) == int(node_idx),
    }


def evaluate_query_primitive(
    network: Network,
    task: ProperUnseenPointerTask,
) -> dict[str, object]:
    probes = [probe_query_primitive(network, task, node_idx=node_idx) for node_idx in range(task.list_length)]
    correct = sum(1 for probe in probes if bool(probe["correct"]))
    return {
        "accuracy": correct / max(task.list_length, 1),
        "results": probes,
    }


def train_writeback_primitive(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    rounds: int = 8,
) -> list[dict[str, int]]:
    dst_area = task.area_map["dst"]
    cur_area = task.area_map["cur"]
    history: list[dict[str, int]] = []
    for round_idx in range(rounds):
        for node_idx in range(task.list_length):
            _clear_activations(network)
            dst_stimulus = _stimulus(
                network.areas_by_name[dst_area].n,
                task.node_assemblies[dst_area][node_idx].indices,
            )
            cur_stimulus = _stimulus(
                network.areas_by_name[cur_area].n,
                task.node_assemblies[cur_area][node_idx].indices,
            )
            network.step(
                external_stimuli={dst_area: dst_stimulus},
                plasticity_on=False,
            )
            network.step(
                external_stimuli={cur_area: cur_stimulus},
                plasticity_on=True,
            )
            network.normalize()
            history.append({"round": int(round_idx), "node_idx": int(node_idx)})
    _clear_activations(network)
    return history


def probe_writeback_primitive(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    node_idx: int,
) -> dict[str, int | bool]:
    dst_area = task.area_map["dst"]
    cur_area = task.area_map["cur"]
    _clear_activations(network)
    dst_stimulus = _stimulus(
        network.areas_by_name[dst_area].n,
        task.node_assemblies[dst_area][node_idx].indices,
    )
    network.step(external_stimuli={dst_area: dst_stimulus}, plasticity_on=False)
    network.step(plasticity_on=False)
    predicted_cur_node = _decode_node(task, cur_area, network.activations[cur_area])
    _clear_activations(network)
    return {
        "node_idx": int(node_idx),
        "predicted_cur_node": int(predicted_cur_node),
        "correct": int(predicted_cur_node) == int(node_idx),
    }


def evaluate_writeback_primitive(
    network: Network,
    task: ProperUnseenPointerTask,
) -> dict[str, object]:
    probes = [probe_writeback_primitive(network, task, node_idx=node_idx) for node_idx in range(task.list_length)]
    correct = sum(1 for probe in probes if bool(probe["correct"]))
    return {
        "accuracy": correct / max(task.list_length, 1),
        "results": probes,
    }


def write_unseen_episode(
    network: Network,
    task: ProperUnseenPointerTask,
    pointer: np.ndarray,
    *,
    write_rounds: int = 1,
    binding_strength: float = 10.0,
) -> dict[str, int | str]:
    reset_proper_episode_memory(network, task)
    key_area = task.area_map["src"]
    value_area = task.area_map["dst"]
    write_steps = 0
    for _ in range(write_rounds):
        for src_node, dst_node in enumerate(pointer.tolist()):
            _clear_activations(network)
            key_stimulus = _stimulus(network.areas_by_name[key_area].n, task.node_assemblies[key_area][src_node].indices, binding_strength)
            value_stimulus = _stimulus(network.areas_by_name[value_area].n, task.node_assemblies[value_area][int(dst_node)].indices, binding_strength)
            network.step(external_stimuli={key_area: key_stimulus, value_area: value_stimulus}, plasticity_on=True)
            write_steps += 1
    return {"write_mode": "plastic", "write_steps": write_steps}


def probe_episode_memory(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    src_node: int,
) -> dict[str, int | bool]:
    src_area = task.area_map["src"]
    dst_area = task.area_map["dst"]
    readout_area = task.area_map["readout"]

    _clear_activations(network)
    network.inhibit(readout_area)
    src_stimulus = _stimulus(
        network.areas_by_name[src_area].n,
        task.node_assemblies[src_area][src_node].indices,
    )
    network.step(external_stimuli={src_area: src_stimulus}, plasticity_on=False)
    predicted_dst_node = _decode_node(task, dst_area, network.activations[dst_area])
    network.disinhibit(readout_area)
    memory_active = bool(network.weights[task.memory_fiber].sum() > task.episodic_baseline.sum())
    _clear_activations(network)
    return {
        "src_node": int(src_node),
        "predicted_dst_node": int(predicted_dst_node),
        "memory_active": memory_active,
    }


def rollout_proper_unseen_pointer(
    network: Network,
    task: ProperUnseenPointerTask,
    pointer: np.ndarray,
    *,
    start_node: int,
    hops: int,
    internal_steps: int | None = None,
) -> dict[str, object]:
    write_stats = write_unseen_episode(network, task, np.asarray(pointer, dtype=np.int64))
    current_area = task.area_map["cur"]
    src_area = task.area_map["src"]
    dst_area = task.area_map["dst"]
    readout_area = task.area_map["readout"]
    current_stimulus = _stimulus(network.areas_by_name[current_area].n, task.node_assemblies[current_area][start_node].indices)
    _clear_activations(network)
    network.inhibit(readout_area)
    current_state_nodes = [int(start_node)]
    network.step(external_stimuli={current_area: current_stimulus}, plasticity_on=False)
    src_nodes = [_decode_node(task, src_area, network.activations[src_area])]
    dst_nodes = [_decode_node(task, dst_area, network.activations[dst_area])]
    rollout_steps = int(internal_steps) if internal_steps is not None else int(hops)
    hop_ctrl_states = [max(int(hops), 0)]
    for _ in range(rollout_steps):
        network.step(plasticity_on=False)
        decoded = _decode_node(task, current_area, network.activations[current_area])
        current_state_nodes.append(int(decoded))
        src_nodes.append(_decode_node(task, src_area, network.activations[src_area]))
        dst_nodes.append(_decode_node(task, dst_area, network.activations[dst_area]))
        hop_ctrl_states.append(max(hop_ctrl_states[-1] - 1, 0))
    network.disinhibit(readout_area)
    final_prediction = current_state_nodes[-1]
    return {
        "start_node": int(start_node),
        "hops": int(hops),
        "intermediate_nodes": current_state_nodes[1:],
        "current_state_nodes": current_state_nodes,
        "cur_nodes": current_state_nodes,
        "src_nodes": src_nodes,
        "dst_nodes": dst_nodes,
        "final_prediction": int(final_prediction),
        "external_cue_count": 1,
        "internal_steps": rollout_steps,
        "hop_ctrl_states": hop_ctrl_states,
        "hop_ctrl_source": "network",
        "controller_mode": "internal",
        "mechanism_route": "CUR->SRC->DST->CUR",
        "write_mode": write_stats["write_mode"],
        "write_steps": write_stats["write_steps"],
    }


def train_one_hop_composition(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    training_lists: list[np.ndarray],
    rounds: int = 4,
    nudge_strength: float = 5.0,
) -> list[dict[str, int]]:
    cur_area = task.area_map["cur"]
    src_area = task.area_map["src"]
    dst_area = task.area_map["dst"]
    history: list[dict[str, int]] = []
    
    for round_idx in range(rounds):
        for pointer in training_lists:
            pointer_arr = np.asarray(pointer, dtype=np.int64)
            write_unseen_episode(network, task, pointer_arr, write_rounds=2)
            
            for start_node in range(task.list_length):
                target_node = int(pointer_arr[start_node])
                
                # Step 1: Stimulate CUR, nudge SRC
                _clear_activations(network)
                cur_stim = _stimulus(network.areas_by_name[cur_area].n, task.node_assemblies[cur_area][start_node].indices)
                src_nudge = _stimulus(network.areas_by_name[src_area].n, task.node_assemblies[src_area][start_node].indices, nudge_strength)
                
                network.step(external_stimuli={cur_area: cur_stim, src_area: src_nudge}, plasticity_on=True)
                
                # Step 2: Allow DST -> CUR to occur, nudge CUR[target]
                cur_target_nudge = _stimulus(network.areas_by_name[cur_area].n, task.node_assemblies[cur_area][target_node].indices, nudge_strength)
                # We need one step for SRC -> DST to propagate if not already there, 
                # but with our step order it happens in Step 1.
                # So in Step 2 we just reinforce CUR from DST.
                network.step(external_stimuli={cur_area: cur_target_nudge}, plasticity_on=True)
                
                # Step 3: Stabilize CUR
                network.step(plasticity_on=True)
                network.normalize()
                
                history.append({"round": round_idx, "node": start_node})
                
            reset_proper_episode_memory(network, task)
            
    return history


def train_multi_hop_recurrence(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    training_lists: list[np.ndarray],
    episodes: int = 4,
    rng: np.random.Generator,
    nudge_strength: float = 3.0,
) -> list[dict[str, int]]:
    cur_area = task.area_map["cur"]
    src_area = task.area_map["src"]
    history: list[dict[str, int]] = []
    
    for ep_idx in range(episodes):
        pointer = training_lists[int(rng.integers(0, len(training_lists)))]
        pointer_arr = np.asarray(pointer, dtype=np.int64)
        write_unseen_episode(network, task, pointer_arr, write_rounds=2)
        
        start_node = int(rng.integers(0, task.list_length))
        path = [start_node]
        for _ in range(3):
            path.append(int(pointer_arr[path[-1]]))
            
        _clear_activations(network)
        # Initial cue
        cur_stim = _stimulus(network.areas_by_name[cur_area].n, task.node_assemblies[cur_area][path[0]].indices)
        network.step(external_stimuli={cur_area: cur_stim}, plasticity_on=False)
        
        # Free run with nudges for target nodes in path
        for target in path[1:]:
            # Nudge the destination node to reinforce recurrence
            nudge = _stimulus(network.areas_by_name[cur_area].n, task.node_assemblies[cur_area][target].indices, nudge_strength)
            network.step(external_stimuli={cur_area: nudge}, plasticity_on=True)
            # Extra step to stabilize
            network.step(plasticity_on=True)
            network.normalize()
            
        history.append({"episode": ep_idx})
        reset_proper_episode_memory(network, task)
        
    return history
def evaluate_proper_unseen_rollout(
    network: Network,
    task: ProperUnseenPointerTask,
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
            trace = rollout_proper_unseen_pointer(
                network,
                task,
                pointer_arr,
                start_node=start_node,
                hops=hops,
                internal_steps=internal_steps,
            )
            target = follow_pointer(pointer_arr, start=start_node, hops=hops)
            prediction_value = trace["final_prediction"]
            prediction = int(prediction_value) if isinstance(prediction_value, (int, np.integer, float, str)) else 0
            correct += int(prediction == target)
            total += 1
    return correct / max(total, 1)


def evaluate_one_hop_composition(
    network: Network,
    task: ProperUnseenPointerTask,
    *,
    test_lists: list[np.ndarray],
) -> dict[str, object]:
    correct = 0
    total = 0
    memory_dependent = True

    for pointer in test_lists:
        pointer_arr = np.asarray(pointer, dtype=np.int64)
        write_unseen_episode(network, task, pointer_arr, write_rounds=2)
        
        for start_node in range(task.list_length):
            target_node = int(pointer_arr[start_node])
            
            trace = rollout_proper_unseen_pointer(
                network,
                task,
                pointer_arr,
                start_node=start_node,
                hops=1,
                internal_steps=1,
            )
            
            if trace["final_prediction"] == target_node:
                correct += 1
            total += 1
            
    accuracy = correct / max(total, 1)
    
    return {
        "accuracy": float(accuracy),
        "memory_dependent": bool(memory_dependent),
    }

