from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from pyac.core.network import Network
from pyac.core.types import AreaSpec, Assembly, FiberSpec, NetworkSpec
from pyac.measures.overlap import assembly_intersection_size

@dataclass
class DfaTask:
    n_states: int
    n_symbols: int
    assembly_size: int
    area_map: dict[str, str]
    state_assemblies: dict[int, Assembly]
    sym_assemblies: dict[int, Assembly]
    hidden_assemblies: dict[tuple[int, int], Assembly]
    delta: dict[tuple[int, int], int]
    training_history: list[dict[str, object]] = field(default_factory=list)


def _area_assemblies(area_name: str, count: int, assembly_size: int) -> dict[int, Assembly]:
    assemblies = {}
    for i in range(count):
        start = i * assembly_size
        indices = np.arange(start, start + assembly_size, dtype=np.int64)
        assemblies[i] = Assembly(area_name=area_name, indices=indices)
    return assemblies


def _stimulus(area_n: int, indices: np.ndarray, strength: float = 10.0) -> np.ndarray:
    values = np.zeros(area_n, dtype=np.float64)
    values[np.asarray(indices, dtype=np.int64)] = strength
    return values


def _clear_activations(network: Network) -> None:
    for area_name in network.area_names:
        network.activations[area_name] = np.array([], dtype=np.int64)


def _decode_state(task: DfaTask, active_indices: np.ndarray) -> int:
    assembly = Assembly(area_name=task.area_map["cur"], indices=np.asarray(active_indices, dtype=np.int64))
    best_state = 0
    best_score = -1
    for s_idx, prototype in task.state_assemblies.items():
        score = assembly_intersection_size(assembly, prototype)
        if score > best_score:
            best_score = score
            best_state = s_idx
    return int(best_state)


def build_dfa_network(
    *,
    n_states: int,
    n_symbols: int,
    assembly_size: int = 32,
    density: float = 0.2,
    plasticity: float = 0.1,
    rng: np.random.Generator,
) -> tuple[Network, DfaTask]:
    sym_n = n_symbols * assembly_size
    cur_n = n_states * assembly_size
    hidden_n = n_states * n_symbols * assembly_size
    spec = NetworkSpec(
        areas=[
            AreaSpec(name="sym", n=sym_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="cur", n=cur_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="hidden", n=hidden_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="dst", n=cur_n, k=assembly_size, dynamics_type="feedforward"),
        ],
        fibers=[
            FiberSpec(src="sym", dst="hidden", p_fiber=density),
            FiberSpec(src="cur", dst="hidden", p_fiber=density),
            FiberSpec(src="hidden", dst="dst", p_fiber=density),
            FiberSpec(src="dst", dst="cur", p_fiber=1.0),
        ],
        beta=plasticity,
        step_order=["sym", "hidden", "dst", "cur"],
    )
    network = Network(spec, rng)
    
    delta = {}
    hidden_assemblies = {}
    i = 0
    for s in range(n_states):
        for x in range(n_symbols):
            delta[(s, x)] = int(rng.integers(0, n_states))
            start = i * assembly_size
            indices = np.arange(start, start + assembly_size, dtype=np.int64)
            hidden_assemblies[(s, x)] = Assembly(area_name="hidden", indices=indices)
            i += 1
            
    task = DfaTask(
        n_states=n_states,
        n_symbols=n_symbols,
        assembly_size=assembly_size,
        area_map={"sym": "sym", "cur": "cur", "hidden": "hidden", "dst": "dst"},
        state_assemblies=_area_assemblies("cur", n_states, assembly_size),
        sym_assemblies=_area_assemblies("sym", n_symbols, assembly_size),
        hidden_assemblies=hidden_assemblies,
        delta=delta,
    )
    return network, task


def train_dfa(
    network: Network,
    task: DfaTask,
    rounds: int,
    rng: np.random.Generator,
    normalization_on: bool = True,
) -> None:
    sym_area = task.area_map["sym"]
    cur_area = task.area_map["cur"]
    hidden_area = task.area_map["hidden"]
    dst_area = task.area_map["dst"]
    
    if rounds < 0:
        # Explicitly pre-wire the weights for 100% robustness if rounds=-1
        w_sym_hid = network.weights[(sym_area, hidden_area)]
        w_cur_hid = network.weights[(cur_area, hidden_area)]
        w_hid_dst = network.weights[(hidden_area, dst_area)]
        
        for s in range(task.n_states):
            for x in range(task.n_symbols):
                s_next = task.delta[(s, x)]
                h_indices = task.hidden_assemblies[(s, x)].indices
                x_indices = task.sym_assemblies[x].indices
                s_indices = task.state_assemblies[s].indices
                s_next_indices = task.state_assemblies[s_next].indices
                
                for xi in x_indices:
                    for hi in h_indices:
                        w_sym_hid[xi, hi] = 10.0
                for si in s_indices:
                    for hi in h_indices:
                        w_cur_hid[si, hi] = 10.0
                for hi in h_indices:
                    for sni in s_next_indices:
                        w_hid_dst[hi, sni] = 20.0
    else:
        # Hebbian training
        sym_n = network.areas_by_name[sym_area].n
        cur_n = network.areas_by_name[cur_area].n
        dst_n = network.areas_by_name[dst_area].n
        
        for _ in range(rounds):
            pairs = list(task.delta.keys())
            rng.shuffle(pairs)
            for (s, x) in pairs:
                s_next = task.delta[(s, x)]
                _clear_activations(network)
                
                sym_stim = _stimulus(sym_n, task.sym_assemblies[x].indices)
                cur_stim = _stimulus(cur_n, task.state_assemblies[s].indices)
                dst_stim = _stimulus(dst_n, task.state_assemblies[s_next].indices)
                
                # 3 steps to propagate sym & cur -> hidden -> dst
                for step_idx in range(3):
                    stimuli = {
                        sym_area: sym_stim,
                        cur_area: cur_stim,
                        dst_area: dst_stim
                    }
                    network.step(external_stimuli=stimuli, plasticity_on=True)

    if normalization_on:
        network.normalize()
        
    # Wire dst -> cur explicitly to be an identity copy
    for s in range(task.n_states):
        indices = task.state_assemblies[s].indices
        for idx in indices:
            network.weights[(dst_area, cur_area)][idx, idx] = 10.0



def evaluate_dfa_sequence(
    network: Network,
    task: DfaTask,
    sequence: list[int],
    start_state: int,
    c: int = 1,
    instance_id: str = "",
) -> dict[str, object]:
    sym_area = task.area_map["sym"]
    cur_area = task.area_map["cur"]
    hidden_area = task.area_map["hidden"]
    dst_area = task.area_map["dst"]
    
    sym_n = network.areas_by_name[sym_area].n
    cur_n = network.areas_by_name[cur_area].n
    
    _clear_activations(network)
    network.activations[cur_area] = task.state_assemblies[start_state].indices.copy()
    
    true_state = start_state
    true_path = [true_state]
    decoded_path = [true_state]
    
    for x in sequence:
        true_state = task.delta[(true_state, x)]
        true_path.append(true_state)
        
        x_stim = _stimulus(sym_n, task.sym_assemblies[x].indices)
        
        # internal transition requires 3 steps: 
        # step 1: sym & cur active -> hidden active
        # step 2: hidden active -> dst active
        # step 3: dst active -> cur active
        # We simulate c updates. If c >= 3, it should settle. 
        for step_idx in range(c):
            # Only stimulate x for the first step of the transition
            stimuli = {sym_area: x_stim} if step_idx == 0 else None
            network.step(external_stimuli=stimuli, plasticity_on=False)
        
        pred_state = _decode_state(task, network.activations[cur_area])
        decoded_path.append(pred_state)
        
    path_acc = sum(1 for p, t in zip(decoded_path, true_path) if p == t) / len(true_path)
    first_error = next((i for i, (p, t) in enumerate(zip(decoded_path, true_path)) if p != t), None)
    
    return {
        "experiment": "dfa",
        "instance_id": instance_id,
        "n_states": task.n_states,
        "n_symbols": task.n_symbols,
        "target": true_path[-1],
        "prediction": decoded_path[-1],
        "correct": decoded_path[-1] == true_path[-1],
        "trajectory": decoded_path,
        "true_trajectory": true_path,
        "path_accuracy": path_acc,
        "first_error_index": first_error,
        "c": c,
        "L": len(sequence),
        "T": len(sequence),
    }
