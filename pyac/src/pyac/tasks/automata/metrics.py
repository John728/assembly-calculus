from __future__ import annotations

import numpy as np
from numpy.random import Generator

from pyac.core.network import Network
from pyac.core.rng import spawn_rngs
from pyac.core.types import Assembly
from pyac.tasks.automata.dfa import DFA
from pyac.tasks.automata.protocol import build_fsm_network, eval_fsm, train_fsm


def _make_fixed_assemblies(
    states: tuple[str, ...],
    alphabet: tuple[str, ...],
    n_neurons: int,
    k: int,
    rng: Generator,
) -> dict[str, Assembly]:
    n_required = (len(states) + len(alphabet)) * k
    if n_required > n_neurons:
        raise ValueError("n_neurons must fit disjoint assemblies")

    all_indices = rng.permutation(n_neurons)[:n_required]
    assemblies: dict[str, Assembly] = {}

    offset = 0
    for state in states:
        block = np.sort(all_indices[offset : offset + k]).astype(np.int64)
        assemblies[state] = Assembly(area_name="state", indices=block)
        offset += k

    for symbol in alphabet:
        block = np.sort(all_indices[offset : offset + k]).astype(np.int64)
        assemblies[symbol] = Assembly(area_name="symbol", indices=block)
        offset += k

    return assemblies


def accuracy_vs_presentations(
    dfa: DFA,
    n_neurons: int,
    k: int,
    density: float,
    plasticity: float,
    n_presentations_list: list[int],
    test_strings: list[str],
    rng: Generator,
) -> dict[int, float]:
    if not n_presentations_list:
        raise ValueError("n_presentations_list must not be empty")
    if not test_strings:
        raise ValueError("test_strings must not be empty")

    asm_rng, sweep_rng = spawn_rngs(rng, 2)
    assemblies = _make_fixed_assemblies(dfa.states, dfa.alphabet, n_neurons, k, asm_rng)

    results: dict[int, float] = {}

    for n_pres in n_presentations_list:
        if n_pres <= 0:
            raise ValueError("n_presentations_list must contain only positive integers")

        net_rng, train_rng, eval_rng = spawn_rngs(sweep_rng, 3)

        network, area_mapping = build_fsm_network(
            dfa=dfa,
            n_neurons=n_neurons,
            k=k,
            density=density,
            plasticity=plasticity,
            rng=net_rng,
        )

        train_fsm(
            network=network,
            dfa=dfa,
            area_mapping=area_mapping,
            assemblies=assemblies,
            n_presentations=n_pres,
            rng=train_rng,
        )

        accuracy = eval_fsm(
            network=network,
            dfa=dfa,
            area_mapping=area_mapping,
            assemblies=assemblies,
            test_strings=test_strings,
            rng=eval_rng,
        )

        results[n_pres] = accuracy

    return results


def state_confusion_matrix(
    predicted_states: list[str],
    true_states: list[str],
) -> np.ndarray:
    if len(predicted_states) != len(true_states):
        raise ValueError("predicted_states and true_states must have same length")
    if not predicted_states:
        raise ValueError("predicted_states must not be empty")

    unique_states = sorted(set(predicted_states) | set(true_states))
    n_states = len(unique_states)
    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}

    matrix = np.zeros((n_states, n_states), dtype=np.int64)

    for pred, true in zip(predicted_states, true_states):
        true_idx = state_to_idx[true]
        pred_idx = state_to_idx[pred]
        matrix[true_idx, pred_idx] += 1

    return matrix
