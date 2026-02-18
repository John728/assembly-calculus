from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DFA:
    states: tuple[str, ...]
    alphabet: tuple[str, ...]
    transitions: dict[tuple[str, str], str]
    initial_state: str
    accept_states: frozenset[str]

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("states must not be empty")
        if not self.alphabet:
            raise ValueError("alphabet must not be empty")
        if len(set(self.states)) != len(self.states):
            raise ValueError("states must be unique")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet symbols must be unique")

        state_set = set(self.states)
        alphabet_set = set(self.alphabet)

        if self.initial_state not in state_set:
            raise ValueError("initial_state must be in states")
        if not self.accept_states.issubset(state_set):
            raise ValueError("accept_states must be a subset of states")

        expected_count = len(self.states) * len(self.alphabet)
        if len(self.transitions) != expected_count:
            raise ValueError("transitions must define exactly one edge per state-symbol pair")

        for (state, symbol), next_state in self.transitions.items():
            if state not in state_set:
                raise ValueError(f"transition references unknown state: {state}")
            if symbol not in alphabet_set:
                raise ValueError(f"transition references unknown symbol: {symbol}")
            if next_state not in state_set:
                raise ValueError(f"transition has unknown next_state: {next_state}")


def make_div3_dfa() -> DFA:
    return DFA(
        states=("q0", "q1", "q2"),
        alphabet=("0", "1"),
        transitions={
            ("q0", "0"): "q0",
            ("q0", "1"): "q1",
            ("q1", "0"): "q2",
            ("q1", "1"): "q0",
            ("q2", "0"): "q1",
            ("q2", "1"): "q2",
        },
        initial_state="q0",
        accept_states=frozenset({"q0"}),
    )
