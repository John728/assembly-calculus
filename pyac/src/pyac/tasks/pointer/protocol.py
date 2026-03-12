from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyac.core.network import Network
from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, Assembly, FiberSpec, NetworkSpec
from pyac.measures.overlap import assembly_intersection_size

from pyac.tasks.pointer.data import follow_pointer, sample_pointer_examples


@dataclass
class PointerTask:
    num_lists: int
    list_length: int
    assembly_size: int
    area_map: dict[str, str]
    token_to_key: list[tuple[int, int]]
    input_assemblies: dict[tuple[int, int], Assembly]
    state_assemblies: dict[tuple[int, int], Assembly] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return self.num_lists * self.list_length

    def key_for(self, list_idx: int, node_idx: int) -> tuple[int, int]:
        return int(list_idx), int(node_idx)


def _stimulus_from_indices(area_n: int, indices: np.ndarray, amplitude: float = 10.0) -> np.ndarray:
    stimulus = np.zeros(area_n, dtype=np.float64)
    stimulus[np.asarray(indices, dtype=np.int64)] = amplitude
    return stimulus


def _reset_network(network: Network) -> None:
    for area_name in network.area_names:
        network.activations[area_name] = np.array([], dtype=np.int64)


def _all_state_keys(task: PointerTask) -> list[tuple[int, int]]:
    return [task.key_for(list_idx, node_idx) for list_idx in range(task.num_lists) for node_idx in range(task.list_length)]


def build_pointer_network(
    num_lists: int,
    list_length: int,
    assembly_size: int = 12,
    density: float = 0.35,
    plasticity: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[Network, PointerTask]:
    if num_lists <= 0:
        raise ValueError("num_lists must be > 0")
    if list_length <= 1:
        raise ValueError("list_length must be > 1")

    rng = make_rng(0) if rng is None else rng
    num_tokens = num_lists * list_length
    input_n = num_tokens * assembly_size
    state_n = num_tokens * assembly_size

    spec = NetworkSpec(
        areas=[
            AreaSpec(name="input", n=input_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="state", n=state_n, k=assembly_size, p_recurrent=density, dynamics_type="recurrent"),
        ],
        fibers=[FiberSpec(src="input", dst="state", p_fiber=density)],
        beta=plasticity,
    )
    network = Network(spec, rng)

    token_to_key: list[tuple[int, int]] = []
    input_assemblies: dict[tuple[int, int], Assembly] = {}
    state_assemblies: dict[tuple[int, int], Assembly] = {}
    for token_id in range(num_tokens):
        start = token_id * assembly_size
        indices = np.arange(start, start + assembly_size, dtype=np.int64)
        state_indices = np.arange(start, start + assembly_size, dtype=np.int64)
        list_idx = token_id // list_length
        node_idx = token_id % list_length
        key = (list_idx, node_idx)
        token_to_key.append(key)
        input_assemblies[key] = Assembly(area_name="input", indices=indices)
        state_assemblies[key] = Assembly(area_name="state", indices=state_indices)

    task = PointerTask(
        num_lists=num_lists,
        list_length=list_length,
        assembly_size=assembly_size,
        area_map={"input": "input", "state": "state"},
        token_to_key=token_to_key,
        input_assemblies=input_assemblies,
        state_assemblies=state_assemblies,
    )
    return network, task


def train_node_assemblies(
    network: Network,
    task: PointerTask,
    presentation_rounds: int = 6,
    settle_steps: int = 4,
) -> dict[tuple[int, int], Assembly]:
    state_area = task.area_map["state"]
    input_area = task.area_map["input"]
    input_n = network.areas_by_name[input_area].n
    state_n = network.areas_by_name[state_area].n

    for _ in range(presentation_rounds):
        for key in _all_state_keys(task):
            _reset_network(network)
            input_assembly = task.input_assemblies[key]
            state_assembly = task.state_assemblies[key]
            input_stimulus = _stimulus_from_indices(input_n, input_assembly.indices)
            state_stimulus = _stimulus_from_indices(state_n, state_assembly.indices)

            for step_idx in range(settle_steps):
                ext = {input_area: input_stimulus, state_area: state_stimulus} if step_idx == 0 else {state_area: state_stimulus}
                network.step(
                    external_stimuli=ext,
                    plasticity_on=True,
                )
                network.activations[input_area] = input_assembly.indices.copy()

    state_weights = network.weights[(state_area, state_area)]
    state_weights.setdiag(0.0)
    state_weights.eliminate_zeros()
    network.normalize(state_area)

    return task.state_assemblies


def train_seen_transitions(
    network: Network,
    task: PointerTask,
    lists: list[np.ndarray],
    transition_rounds: int = 12,
    association_steps: int = 3,
    teacher_strength: float = 12.0,
) -> None:
    state_area = task.area_map["state"]
    state_n = network.areas_by_name[state_area].n

    if not task.state_assemblies:
        raise ValueError("train_node_assemblies must run before transition training")

    for _ in range(transition_rounds):
        for list_idx, pointer in enumerate(lists):
            pointer_arr = np.asarray(pointer, dtype=np.int64)
            for src_node, dst_node in enumerate(pointer_arr.tolist()):
                src_key = task.key_for(list_idx, src_node)
                dst_key = task.key_for(list_idx, int(dst_node))
                src_assembly = task.state_assemblies[src_key]
                dst_assembly = task.state_assemblies[dst_key]
                dst_stimulus = _stimulus_from_indices(state_n, dst_assembly.indices, amplitude=teacher_strength)

                for _ in range(association_steps):
                    _reset_network(network)
                    network.inhibit(task.area_map["input"])
                    network.activations[state_area] = src_assembly.indices.copy()
                    network.step(external_stimuli={state_area: dst_stimulus}, plasticity_on=True)
                    network.disinhibit(task.area_map["input"])

        weights = network.weights[(state_area, state_area)]
        weights.setdiag(0.0)
        weights.eliminate_zeros()
        network.normalize(state_area)


def _decode_node(task: PointerTask, list_idx: int, final_assembly: Assembly) -> int:
    best_node = 0
    best_score = -1
    for (prototype_list_idx, node_idx), prototype in task.state_assemblies.items():
        if prototype_list_idx != list_idx:
            continue
        score = assembly_intersection_size(final_assembly, prototype)
        if score > best_score:
            best_score = score
            best_node = node_idx
    return int(best_node)


def rollout_pointer(
    network: Network,
    task: PointerTask,
    list_idx: int,
    start_node: int,
    hops: int,
    settle_steps: int = 1,
) -> int:
    input_area = task.area_map["input"]
    state_area = task.area_map["state"]
    key = task.key_for(list_idx, start_node)
    start_assembly = task.state_assemblies[key]
    state_stimulus = _stimulus_from_indices(network.areas_by_name[state_area].n, start_assembly.indices)

    _reset_network(network)
    network.inhibit(input_area)
    for step_idx in range(settle_steps):
        ext = {state_area: state_stimulus} if step_idx == 0 else None
        network.step(external_stimuli=ext, plasticity_on=False)

    for _ in range(hops):
        network.step(plasticity_on=False)
    network.disinhibit(input_area)

    final_assembly = network.get_assembly(state_area)
    return _decode_node(task, list_idx, final_assembly)


def evaluate_seen_lists(
    network: Network,
    task: PointerTask,
    lists: list[np.ndarray],
    samples_per_list: int,
    k: int,
    rng: np.random.Generator,
    settle_steps: int = 1,
) -> float:
    examples = sample_pointer_examples(lists, samples_per_list=samples_per_list, k=k, rng=rng)
    correct = 0
    for example in examples:
        prediction = rollout_pointer(
            network,
            task,
            list_idx=int(example["list_idx"]),
            start_node=int(example["start"]),
            hops=int(example["k"]),
            settle_steps=settle_steps,
        )
        target = follow_pointer(np.asarray(example["pointer"], dtype=np.int64), start=int(example["start"]), hops=int(example["k"]))
        correct += int(prediction == target)
    return correct / max(len(examples), 1)
