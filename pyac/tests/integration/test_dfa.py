from __future__ import annotations

import importlib

import numpy as np

from pyac.core.rng import make_rng
from pyac.core.types import Assembly

_dfa_mod = importlib.import_module("pyac.tasks.automata.dfa")
_protocol_mod = importlib.import_module("pyac.tasks.automata.protocol")

make_div3_dfa = _dfa_mod.make_div3_dfa
build_fsm_network = _protocol_mod.build_fsm_network
train_fsm = _protocol_mod.train_fsm
eval_fsm = _protocol_mod.eval_fsm


def _make_assemblies(
    states: tuple[str, ...],
    alphabet: tuple[str, ...],
    n_neurons: int,
    k: int,
    seed: int,
) -> dict[str, Assembly]:
    rng = make_rng(seed)
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


def test_div3_dfa_structure() -> None:
    dfa = make_div3_dfa()

    assert dfa.states == ("q0", "q1", "q2")
    assert dfa.alphabet == ("0", "1")
    assert dfa.transitions[("q0", "0")] == "q0"
    assert dfa.transitions[("q0", "1")] == "q1"
    assert dfa.transitions[("q1", "0")] == "q2"
    assert dfa.transitions[("q1", "1")] == "q0"
    assert dfa.transitions[("q2", "0")] == "q1"
    assert dfa.transitions[("q2", "1")] == "q2"
    assert dfa.initial_state == "q0"
    assert dfa.accept_states == frozenset({"q0"})


def test_build_fsm_network_creates_areas() -> None:
    dfa = make_div3_dfa()
    net, mapping = build_fsm_network(
        dfa,
        n_neurons=120,
        k=10,
        density=0.2,
        plasticity=0.5,
        rng=make_rng(7),
    )

    assert set(mapping.keys()) == {"symbol", "state", "arc"}
    assert set(net.area_names) == {mapping["symbol"], mapping["state"], mapping["arc"]}

    assert net.areas_by_name[mapping["symbol"]].dynamics_type == "feedforward"
    assert net.areas_by_name[mapping["state"]].dynamics_type == "feedforward"
    assert net.areas_by_name[mapping["arc"]].dynamics_type == "refracted"

    fibers = {(fiber.src, fiber.dst) for fiber in net.spec.fibers}
    assert (mapping["symbol"], mapping["arc"]) in fibers
    assert (mapping["state"], mapping["arc"]) in fibers
    assert (mapping["arc"], mapping["state"]) in fibers


def test_accuracy_improves_with_training() -> None:
    dfa = make_div3_dfa()
    net, mapping = build_fsm_network(
        dfa,
        n_neurons=150,
        k=10,
        density=0.1,
        plasticity=0.7,
        rng=make_rng(42),
    )
    assemblies = _make_assemblies(dfa.states, dfa.alphabet, n_neurons=150, k=10, seed=123)

    test_strings = [
        "0",
        "1",
        "10",
        "11",
        "100",
        "101",
        "110",
        "111",
        "1001",
        "1111",
        "10101",
        "1100",
    ]

    acc_before = eval_fsm(
        net,
        dfa,
        mapping,
        assemblies,
        test_strings,
        rng=make_rng(99),
    )

    train_fsm(
        net,
        dfa,
        mapping,
        assemblies,
        n_presentations=20,
        rng=make_rng(77),
    )

    acc_after = eval_fsm(
        net,
        dfa,
        mapping,
        assemblies,
        test_strings,
        rng=make_rng(99),
    )

    assert acc_after > acc_before
    assert acc_after >= 0.8


def test_eval_fsm_deterministic() -> None:
    dfa = make_div3_dfa()
    assemblies = _make_assemblies(dfa.states, dfa.alphabet, n_neurons=120, k=10, seed=555)
    test_strings = ["0", "1", "11", "100", "110", "1001", "11111"]

    net_a, mapping_a = build_fsm_network(
        dfa,
        n_neurons=120,
        k=10,
        density=0.2,
        plasticity=0.6,
        rng=make_rng(314),
    )
    net_b, mapping_b = build_fsm_network(
        dfa,
        n_neurons=120,
        k=10,
        density=0.2,
        plasticity=0.6,
        rng=make_rng(314),
    )

    train_fsm(net_a, dfa, mapping_a, assemblies, n_presentations=15, rng=make_rng(2718))
    train_fsm(net_b, dfa, mapping_b, assemblies, n_presentations=15, rng=make_rng(2718))

    acc_a = eval_fsm(net_a, dfa, mapping_a, assemblies, test_strings, rng=make_rng(1234))
    acc_b = eval_fsm(net_b, dfa, mapping_b, assemblies, test_strings, rng=make_rng(1234))

    assert acc_a == acc_b


_metrics_mod = importlib.import_module("pyac.tasks.automata.metrics")
accuracy_vs_presentations = _metrics_mod.accuracy_vs_presentations
state_confusion_matrix = _metrics_mod.state_confusion_matrix


def test_accuracy_vs_presentations_sweep() -> None:
    """Metrics: accuracy_vs_presentations sweeps correctly."""
    dfa = make_div3_dfa()
    n_pres_list = [1, 10, 20]
    test_strings = ["0", "1", "10", "11"]

    results = accuracy_vs_presentations(
        dfa=dfa,
        n_neurons=100,
        k=10,
        density=0.2,
        plasticity=0.6,
        n_presentations_list=n_pres_list,
        test_strings=test_strings,
        rng=make_rng(42),
    )

    # Check structure
    assert set(results.keys()) == set(n_pres_list)
    assert all(0.0 <= acc <= 1.0 for acc in results.values())

    # Check monotonicity (more presentations -> better or equal accuracy)
    accuracies = [results[n] for n in sorted(n_pres_list)]
    # Final better than initial
    assert accuracies[-1] >= accuracies[0]


def test_accuracy_vs_presentations_deterministic() -> None:
    """Metrics: accuracy_vs_presentations is deterministic with same seed."""
    dfa = make_div3_dfa()
    n_pres_list = [5, 15]
    test_strings = ["0", "1", "11"]

    results_a = accuracy_vs_presentations(
        dfa=dfa,
        n_neurons=80,
        k=8,
        density=0.15,
        plasticity=0.5,
        n_presentations_list=n_pres_list,
        test_strings=test_strings,
        rng=make_rng(999),
    )
    results_b = accuracy_vs_presentations(
        dfa=dfa,
        n_neurons=80,
        k=8,
        density=0.15,
        plasticity=0.5,
        n_presentations_list=n_pres_list,
        test_strings=test_strings,
        rng=make_rng(999),
    )

    assert results_a == results_b


def test_state_confusion_matrix_structure() -> None:
    """Metrics: confusion matrix has correct shape."""
    predicted = ["s0", "s1", "s0", "s2"]
    true = ["s0", "s0", "s1", "s2"]

    matrix = state_confusion_matrix(predicted, true)

    assert matrix.shape == (3, 3)  # 3 unique states
    assert matrix.sum() == 4  # 4 predictions
    assert matrix.dtype == np.int64


def test_state_confusion_matrix_values() -> None:
    """Metrics: confusion matrix counts correct."""
    predicted = ["A", "B", "A"]
    true = ["A", "A", "B"]

    matrix = state_confusion_matrix(predicted, true)
    # Rows=true, Cols=pred (sorted: A, B)
    # True A -> Pred A: 1, True A -> Pred B: 1
    # True B -> Pred A: 1, True B -> Pred B: 0
    assert matrix[0, 0] == 1  # A->A
    assert matrix[0, 1] == 1  # A->B
    assert matrix[1, 0] == 1  # B->A
    assert matrix[1, 1] == 0  # B->B


def test_state_confusion_matrix_single_state() -> None:
    """Metrics: confusion matrix handles single state."""
    predicted = ["X", "X", "X"]
    true = ["X", "X", "X"]

    matrix = state_confusion_matrix(predicted, true)

    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 3
    assert matrix.sum() == 3
