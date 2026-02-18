from __future__ import annotations

import numpy as np
from numpy.random import Generator

from pyac.core.network import Network
from pyac.core.rng import make_rng
from pyac.core.types import AreaSpec, Assembly, FiberSpec, NetworkSpec
from pyac.measures.overlap import assembly_overlap
from pyac.tasks.automata.dfa import DFA


def _validate_prob(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_area_mapping(network: Network, area_mapping: dict[str, str]) -> None:
    required = {"symbol", "state", "arc"}
    missing = required - set(area_mapping.keys())
    if missing:
        raise ValueError(f"area_mapping missing keys: {sorted(missing)}")

    for key in required:
        area_name = area_mapping[key]
        if area_name not in network.area_names:
            raise ValueError(f"area_mapping[{key!r}] references unknown area: {area_name}")


def _validate_assemblies(dfa: DFA, assemblies: dict[str, Assembly]) -> None:
    required = set(dfa.states) | set(dfa.alphabet)
    missing = required - set(assemblies.keys())
    if missing:
        raise ValueError(f"assemblies missing entries: {sorted(missing)}")


def _stimulus_from_assembly(
    n_neurons: int,
    assembly: Assembly,
    amplitude: float = 5.0,
) -> np.ndarray:
    stimulus = np.zeros(n_neurons, dtype=np.float64)
    if assembly.indices.size:
        stimulus[assembly.indices] = amplitude
    return stimulus


def _best_matching_state(
    predicted: Assembly,
    states: tuple[str, ...],
    assemblies: dict[str, Assembly],
) -> str:
    best_state = states[0]
    best_overlap = -1.0
    for state in states:
        candidate = Assembly(area_name=predicted.area_name, indices=assemblies[state].indices)
        overlap = assembly_overlap(predicted, candidate)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state = state
    return best_state


def _reset_network_state(network: Network, clear_refracted_bias: bool) -> None:
    for area_name in network.area_names:
        network.activations[area_name] = np.array([], dtype=np.int64)

    if not clear_refracted_bias:
        return

    for strategy in network.strategies.values():
        bias = getattr(strategy, "bias", None)
        if isinstance(bias, np.ndarray):
            bias.fill(0.0)


def _run_transition(
    network: Network,
    area_mapping: dict[str, str],
    state_assembly: Assembly,
    symbol_assembly: Assembly,
    clamp_state_assembly: Assembly | None,
    plasticity_on: bool,
) -> None:
    symbol_area = area_mapping["symbol"]
    state_area = area_mapping["state"]

    symbol_n = network.areas_by_name[symbol_area].n
    state_n = network.areas_by_name[state_area].n

    encode_stimuli = {
        symbol_area: _stimulus_from_assembly(symbol_n, symbol_assembly),
        state_area: _stimulus_from_assembly(state_n, state_assembly),
    }
    _ = network.step(external_stimuli=encode_stimuli, plasticity_on=plasticity_on)

    decode_stimuli: dict[str, np.ndarray] = {}
    if clamp_state_assembly is not None:
        decode_stimuli[state_area] = _stimulus_from_assembly(state_n, clamp_state_assembly)

    network.inhibit(area_mapping["arc"])
    try:
        _ = network.step(external_stimuli=decode_stimuli, plasticity_on=plasticity_on)
    finally:
        network.disinhibit(area_mapping["arc"])


def build_fsm_network(
    dfa: DFA,
    n_neurons: int,
    k: int,
    density: float,
    plasticity: float,
    rng: Generator,
) -> tuple[Network, dict[str, str]]:
    del dfa

    if n_neurons <= 0:
        raise ValueError("n_neurons must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > n_neurons:
        raise ValueError("k must be <= n_neurons")
    if density <= 0.0:
        raise ValueError("density must be in (0, 1]")
    _validate_prob("density", density)
    _validate_prob("plasticity", plasticity)

    symbol_name = "symbol"
    state_name = "state"
    arc_name = "arc"

    spec = NetworkSpec(
        areas=[
            AreaSpec(symbol_name, n=n_neurons, k=k, dynamics_type="feedforward"),
            AreaSpec(state_name, n=n_neurons, k=k, dynamics_type="feedforward"),
            AreaSpec(arc_name, n=n_neurons, k=k, dynamics_type="refracted"),
        ],
        fibers=[
            FiberSpec(symbol_name, arc_name, density),
            FiberSpec(state_name, arc_name, density),
            FiberSpec(arc_name, state_name, density),
        ],
        beta=plasticity,
    )
    network = Network(spec, rng)
    return network, {"symbol": symbol_name, "state": state_name, "arc": arc_name}


def train_fsm(
    network: Network,
    dfa: DFA,
    area_mapping: dict[str, str],
    assemblies: dict[str, Assembly],
    n_presentations: int = 20,
    rng: Generator | None = None,
) -> None:
    if n_presentations <= 0:
        raise ValueError("n_presentations must be > 0")

    _ = make_rng(0) if rng is None else rng
    _validate_area_mapping(network, area_mapping)
    _validate_assemblies(dfa, assemblies)

    state_area = area_mapping["state"]
    arc_area = area_mapping["arc"]
    prototypes: dict[tuple[str, str], Assembly] = {}

    transitions = list(dfa.transitions.items())
    for _presentation in range(n_presentations):
        for (state, symbol), next_state in transitions:
            _reset_network_state(network, clear_refracted_bias=True)
            _run_transition(
                network=network,
                area_mapping=area_mapping,
                state_assembly=assemblies[state],
                symbol_assembly=assemblies[symbol],
                clamp_state_assembly=assemblies[next_state],
                plasticity_on=True,
            )
            prototypes[(state, symbol)] = network.get_assembly(arc_area)
        network.normalize(state_area)
        network.normalize(arc_area)

    setattr(network, "fsm_arc_prototypes", prototypes)


def eval_fsm(
    network: Network,
    dfa: DFA,
    area_mapping: dict[str, str],
    assemblies: dict[str, Assembly],
    test_strings: list[str],
    rng: Generator,
) -> float:
    _ = rng

    if not test_strings:
        raise ValueError("test_strings must not be empty")

    _validate_area_mapping(network, area_mapping)
    _validate_assemblies(dfa, assemblies)

    alphabet_set = set(dfa.alphabet)
    for test_string in test_strings:
        for symbol in test_string:
            if symbol not in alphabet_set:
                raise ValueError(f"test string contains unknown symbol: {symbol}")

    correct = 0
    state_area = area_mapping["state"]
    arc_area = area_mapping["arc"]
    prototypes = getattr(network, "fsm_arc_prototypes", None)

    for test_string in test_strings:
        _reset_network_state(network, clear_refracted_bias=True)
        current_state = dfa.initial_state

        for symbol in test_string:
            _run_transition(
                network=network,
                area_mapping=area_mapping,
                state_assembly=assemblies[current_state],
                symbol_assembly=assemblies[symbol],
                clamp_state_assembly=None,
                plasticity_on=False,
            )
            if isinstance(prototypes, dict) and prototypes:
                predicted_arc = network.get_assembly(arc_area)
                best_pair = next(iter(prototypes.keys()))
                best_overlap = -1.0
                for pair, prototype in prototypes.items():
                    candidate = Assembly(area_name=predicted_arc.area_name, indices=prototype.indices)
                    overlap = assembly_overlap(predicted_arc, candidate)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_pair = pair
                current_state = dfa.transitions[best_pair]
            else:
                predicted_assembly = network.get_assembly(state_area)
                current_state = _best_matching_state(predicted_assembly, dfa.states, assemblies)

        expected_state = dfa.initial_state
        for symbol in test_string:
            expected_state = dfa.transitions[(expected_state, symbol)]

        if current_state == expected_state:
            correct += 1

    return float(correct) / float(len(test_strings))
