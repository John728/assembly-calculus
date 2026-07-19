from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t


LENGTHS = (5, 10, 20, 40, 80)
DWELL_TIMES = (1, 2, 3, 4, 6)
PRIMARY_SEEDS = (42, 43, 44, 45, 46)
RELIABILITY_SEEDS = tuple(range(42, 62))
HOLD_SEEDS = tuple(range(42, 62))
COLOURS = ("#0077BB", "#EE7733", "#009988", "#CC3311", "#AA4499")
HOLD_SYMBOL = 2
HOLD_SEQUENCE_LENGTH = 40
HOLD_MAX_LAG = 100
HOLD_SEQUENCES_PER_SEED = 100
RELIABILITY_FIT_PATHS = 500
RELIABILITY_TEST_PATHS = 500
RELIABILITY_FIT_HORIZON = 40
RELIABILITY_TEST_HORIZON = 100


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def import_pyac(ac_root: Path):
    sys.path.insert(0, str(ac_root / "pyac" / "src"))
    from pyac.tasks.dfa.dfa_protocol import (  # type: ignore
        _clear_activations,
        _decode_state,
        _stimulus,
        build_dfa_network,
        evaluate_dfa_sequence,
    )

    return _clear_activations, _decode_state, _stimulus, build_dfa_network, evaluate_dfa_sequence


def full_normalise(network) -> None:
    """Apply Chapter 3's single incoming budget across all fibres to an area."""
    for target in network.area_names:
        keys = [key for key in network.weights if key[1] == target]
        if not keys:
            continue
        total = sum(np.asarray(network.weights[key].sum(axis=0)).ravel() for key in keys)
        total[total == 0.0] = 1.0
        for key in keys:
            matrix = network.weights[key]
            matrix.data = matrix.data / total[matrix.indices]


def train_dfa_supervised(network, task, rounds: int, rng, helpers) -> None:
    clear, _, stimulus, _, _ = helpers
    sym = task.area_map["sym"]
    cur = task.area_map["cur"]
    hidden = task.area_map["hidden"]
    dst = task.area_map["dst"]
    sizes = {name: network.areas_by_name[name].n for name in (sym, hidden, dst)}

    for _ in range(rounds):
        pairs = list(task.delta)
        rng.shuffle(pairs)
        for state, symbol in pairs:
            next_state = task.delta[(state, symbol)]
            clear(network)
            # The source state must precede the ordered sweep so hidden receives it.
            network.activations[cur] = task.state_assemblies[state].indices.copy()
            network.step(
                external_stimuli={
                    sym: stimulus(sizes[sym], task.sym_assemblies[symbol].indices),
                    hidden: stimulus(sizes[hidden], task.hidden_assemblies[(state, symbol)].indices),
                    dst: stimulus(sizes[dst], task.state_assemblies[next_state].indices),
                },
                plasticity_on=True,
            )

    full_normalise(network)
    relay = sparse.lil_matrix(network.weights[(dst, cur)].shape, dtype=np.float64)
    for state in range(task.n_states):
        indices = task.state_assemblies[state].indices
        relay[np.ix_(indices, indices)] = 1.0
    network.weights[(dst, cur)] = relay.tocsr()
    full_normalise(network)


def build_trained(
    seed: int,
    helpers,
    *,
    assembly_size: int,
    density: float,
    n_symbols: int = 2,
    identity_symbol: int | None = None,
):
    _, _, _, build, _ = helpers
    rng = np.random.default_rng(seed)
    network, task = build(
        n_states=5,
        n_symbols=n_symbols,
        assembly_size=assembly_size,
        density=density,
        plasticity=0.25,
        rng=rng,
    )
    if identity_symbol is not None:
        if not 0 <= identity_symbol < n_symbols:
            raise ValueError("identity_symbol must index an available symbol")
        for state in range(task.n_states):
            task.delta[(state, identity_symbol)] = state
    full_normalise(network)
    train_dfa_supervised(network, task, rounds=12, rng=rng, helpers=helpers)
    return network, task, rng


def run_primary(helpers) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    example: dict[str, object] | None = None
    for seed in PRIMARY_SEEDS:
        network, task, rng = build_trained(seed, helpers, assembly_size=24, density=1.0)
        for length in LENGTHS:
            for sample in range(20):
                start = int(rng.integers(0, task.n_states))
                sequence = [int(rng.integers(0, task.n_symbols)) for _ in range(length)]
                for dwell in DWELL_TIMES:
                    result = evaluate_dfa_dwell(
                        network, task, sequence, start_state=start, dwell=dwell, helpers=helpers
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "sample": sample,
                            "L": length,
                            "c": dwell,
                            "path_accuracy": result["path_accuracy"],
                            "path_correct": result["first_error_index"] is None,
                            "final_correct": result["correct"],
                            "first_error_index": result["first_error_index"],
                            "true_trajectory": json.dumps(result["true_trajectory"]),
                            "trajectory": json.dumps(result["trajectory"]),
                        }
                    )
        if seed == 43:
            # Select only for symbolic variety, before observing neural activity.
            trace_rng = np.random.default_rng(2026)
            candidates = []
            for candidate_index in range(1000):
                start = int(trace_rng.integers(0, task.n_states))
                sequence = [int(trace_rng.integers(0, task.n_symbols)) for _ in range(12)]
                states = [start]
                current = start
                for symbol in sequence:
                    current = task.delta[(current, symbol)]
                    states.append(current)
                score = (len(set(states)), sum(a != b for a, b in zip(states, states[1:])))
                candidates.append((score, -candidate_index, start, sequence))
            _, _, start, sequence = max(candidates)
            example = trace_sequence(network, task, sequence, start=start, helpers=helpers)
    assert example is not None
    return pd.DataFrame(rows), example


def evaluate_dfa_dwell(network, task, sequence: list[int], start_state: int, dwell: int, helpers):
    clear, decode, stimulus, _, _ = helpers
    sym, cur = task.area_map["sym"], task.area_map["cur"]
    clear(network)
    network.activations[cur] = task.state_assemblies[start_state].indices.copy()
    true_state = start_state
    true_path = [start_state]
    decoded_path = [start_state]
    for symbol in sequence:
        true_state = task.delta[(true_state, symbol)]
        true_path.append(true_state)
        symbol_stimulus = stimulus(
            network.areas_by_name[sym].n, task.sym_assemblies[symbol].indices
        )
        for _ in range(dwell):
            network.step(external_stimuli={sym: symbol_stimulus}, plasticity_on=False)
        decoded_path.append(decode(task, network.activations[cur]))

    first_error = next(
        (index for index, (predicted, true) in enumerate(zip(decoded_path, true_path))
         if predicted != true),
        None,
    )
    return {
        "correct": decoded_path[-1] == true_path[-1],
        "path_accuracy": sum(
            predicted == true for predicted, true in zip(decoded_path, true_path)
        ) / len(true_path),
        "first_error_index": first_error,
        "true_trajectory": true_path,
        "trajectory": decoded_path,
    }


def trace_sequence(network, task, sequence: list[int], start: int, helpers) -> dict[str, object]:
    clear, decode, stimulus, _, _ = helpers
    sym, cur = task.area_map["sym"], task.area_map["cur"]
    clear(network)
    network.activations[cur] = task.state_assemblies[start].indices.copy()
    true_states = [start]
    decoded_states = [start]
    overlaps = [[1.0 if state == start else 0.0 for state in range(task.n_states)]]
    current = start
    for symbol in sequence:
        current = task.delta[(current, symbol)]
        true_states.append(current)
        network.step(
            external_stimuli={sym: stimulus(network.areas_by_name[sym].n, task.sym_assemblies[symbol].indices)},
            plasticity_on=False,
        )
        active = network.activations[cur]
        decoded_states.append(decode(task, active))
        active_set = set(int(index) for index in active)
        overlaps.append(
            [
                len(active_set.intersection(int(index) for index in task.state_assemblies[state].indices))
                / task.assembly_size
                for state in range(task.n_states)
            ]
        )
    return {
        "sequence": sequence,
        "true_states": true_states,
        "decoded_states": decoded_states,
        "overlaps": overlaps,
    }


def target_margin(task, active_indices: np.ndarray, target_state: int) -> float:
    active = set(int(index) for index in active_indices)
    scores = []
    for state in range(task.n_states):
        assembly = task.state_assemblies[state].indices
        scores.append(len(active.intersection(int(index) for index in assembly)) / task.assembly_size)
    return float(scores[target_state] - max(score for state, score in enumerate(scores) if state != target_state))


def evaluate_dfa_holding(
    network,
    task,
    sequence: list[int],
    start_state: int,
    post_symbol: int,
    helpers,
) -> dict[str, object]:
    clear, decode, stimulus, _, _ = helpers
    sym, cur = task.area_map["sym"], task.area_map["cur"]
    clear(network)
    network.activations[cur] = task.state_assemblies[start_state].indices.copy()

    true_state = start_state
    prefix_correct = True
    for symbol in sequence:
        true_state = task.delta[(true_state, symbol)]
        network.step(
            external_stimuli={
                sym: stimulus(network.areas_by_name[sym].n, task.sym_assemblies[symbol].indices)
            },
            plasticity_on=False,
        )
        prefix_correct = prefix_correct and decode(task, network.activations[cur]) == true_state

    held = []
    margins = []
    decoded = []
    for lag in range(HOLD_MAX_LAG + 1):
        prediction = decode(task, network.activations[cur])
        decoded.append(prediction)
        held.append(prediction == true_state)
        margins.append(target_margin(task, network.activations[cur], true_state))
        if lag < HOLD_MAX_LAG:
            network.step(
                external_stimuli={
                    sym: stimulus(
                        network.areas_by_name[sym].n,
                        task.sym_assemblies[post_symbol].indices,
                    )
                },
                plasticity_on=False,
            )

    return {
        "target": true_state,
        "prefix_correct": prefix_correct,
        "held": held,
        "margins": margins,
        "decoded": decoded,
    }


def build_holding_pair(seed: int, helpers):
    _, _, _, build, _ = helpers
    rng = np.random.default_rng(seed)
    network, task = build(
        n_states=5,
        n_symbols=3,
        assembly_size=24,
        density=1.0,
        plasticity=0.25,
        rng=rng,
    )
    hold_transitions = {(state, HOLD_SYMBOL): state for state in range(task.n_states)}
    task.delta.update(hold_transitions)
    for transition in hold_transitions:
        task.delta.pop(transition)

    full_normalise(network)
    train_dfa_supervised(network, task, rounds=12, rng=rng, helpers=helpers)
    task.delta.update(hold_transitions)

    untrained = copy.deepcopy(network)
    learned = copy.deepcopy(network)
    hold_task = copy.deepcopy(task)
    hold_task.delta = hold_transitions
    train_dfa_supervised(learned, hold_task, rounds=12, rng=learned.rng, helpers=helpers)
    return learned, untrained, task


def run_holding(helpers) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in HOLD_SEEDS:
        learned, untrained, task = build_holding_pair(seed, helpers)
        evaluation_rng = np.random.default_rng(np.random.SeedSequence([seed, 20260718]))
        for sample in range(HOLD_SEQUENCES_PER_SEED):
            start = int(evaluation_rng.integers(0, task.n_states))
            sequence = [int(evaluation_rng.integers(0, 2)) for _ in range(HOLD_SEQUENCE_LENGTH)]
            conditions = (
                ("learned_hold", learned, HOLD_SYMBOL),
                ("untrained_hold", untrained, HOLD_SYMBOL),
                ("repeat_final_symbol", learned, sequence[-1]),
            )
            for condition, network, post_symbol in conditions:
                result = evaluate_dfa_holding(
                    network,
                    task,
                    sequence,
                    start_state=start,
                    post_symbol=post_symbol,
                    helpers=helpers,
                )
                for lag, (held, margin, decoded) in enumerate(
                    zip(result["held"], result["margins"], result["decoded"])
                ):
                    rows.append(
                        {
                            "seed": seed,
                            "sample": sample,
                            "L": HOLD_SEQUENCE_LENGTH,
                            "condition": condition,
                            "d": lag,
                            "target": result["target"],
                            "prediction": decoded,
                            "held": held,
                            "margin": margin,
                            "prefix_correct": result["prefix_correct"],
                            "post_symbol": post_symbol,
                        }
                    )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["condition", "seed", "sample", "d"]).reset_index(drop=True)
    frame["continuously_held"] = frame.groupby(
        ["condition", "seed", "sample"], sort=False
    )["held"].cummin()
    completed = frame[frame["d"] == 0]
    assert bool(completed["prefix_correct"].all())
    assert bool(completed["held"].all())
    return frame


def exact_reference_cap(task, active_indices: np.ndarray, state: int) -> bool:
    return bool(
        np.array_equal(
            np.sort(np.asarray(active_indices, dtype=np.int64)),
            np.sort(np.asarray(task.state_assemblies[state].indices, dtype=np.int64)),
        )
    )


def evaluate_reliability_path(
    network,
    task,
    sequence: list[int],
    start_state: int,
    helpers,
) -> tuple[list[bool], list[dict[str, object]]]:
    clear, decode, stimulus, _, _ = helpers
    sym, cur = task.area_map["sym"], task.area_map["cur"]
    clear(network)
    network.activations[cur] = task.state_assemblies[start_state].indices.copy()
    true_state = start_state
    prefix_correct = True
    survival: list[bool] = []
    events: list[dict[str, object]] = []

    for depth, symbol in enumerate(sequence, start=1):
        if not prefix_correct:
            survival.append(False)
            continue
        source_state = true_state
        source_exact = exact_reference_cap(task, network.activations[cur], source_state)
        true_state = task.delta[(source_state, symbol)]
        network.step(
            external_stimuli={
                sym: stimulus(network.areas_by_name[sym].n, task.sym_assemblies[symbol].indices)
            },
            plasticity_on=False,
        )
        prediction = decode(task, network.activations[cur])
        correct = prediction == true_state
        destination_exact = correct and exact_reference_cap(
            task, network.activations[cur], true_state
        )
        events.append(
            {
                "depth": depth,
                "source_state": source_state,
                "source_exact": int(source_exact),
                "symbol": int(symbol),
                "target_state": true_state,
                "correct": int(correct),
                "destination_exact": int(destination_exact),
            }
        )
        prefix_correct = prefix_correct and correct
        survival.append(prefix_correct)
    return survival, events


def estimate_survival_kernels(task, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    label_kernel = np.zeros((task.n_states, task.n_states), dtype=np.float64)
    representation_kernel = np.zeros((2 * task.n_states, 2 * task.n_states), dtype=np.float64)

    for source_state in range(task.n_states):
        for symbol in range(task.n_symbols):
            target = task.delta[(source_state, symbol)]
            subset = events[
                (events["source_state"] == source_state) & (events["symbol"] == symbol)
            ]
            if not subset.empty:
                label_kernel[source_state, target] += 0.5 * float(subset["correct"].mean())

            for source_exact in (0, 1):
                state_subset = subset[subset["source_exact"] == source_exact]
                if state_subset.empty:
                    continue
                source_index = 2 * source_state + source_exact
                denominator = float(len(state_subset))
                for destination_exact in (0, 1):
                    count = int(
                        (
                            (state_subset["correct"] == 1)
                            & (state_subset["destination_exact"] == destination_exact)
                        ).sum()
                    )
                    target_index = 2 * target + destination_exact
                    representation_kernel[source_index, target_index] += 0.5 * count / denominator

    constant_error = 1.0 - float(events["correct"].mean())
    return label_kernel, representation_kernel, constant_error


def survival_curve(kernel: np.ndarray, initial: np.ndarray, horizon: int) -> np.ndarray:
    distribution = initial.astype(np.float64, copy=True)
    values = []
    for _ in range(horizon):
        distribution = distribution @ kernel
        values.append(float(distribution.sum()))
    return np.asarray(values, dtype=np.float64)


def kernel_diagnostics(kernel: np.ndarray, initial: np.ndarray) -> tuple[float, float]:
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(kernel))))
    if spectral_radius >= 1.0 - 1e-10:
        return spectral_radius, float("inf")
    expected_survival = float(initial @ np.linalg.solve(np.eye(len(kernel)) - kernel, np.ones(len(kernel))))
    return spectral_radius, expected_survival


def run_reliability(helpers) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in RELIABILITY_SEEDS:
        network, task, _ = build_trained(seed, helpers, assembly_size=10, density=0.2)
        seed_sequence = np.random.SeedSequence([seed, 20260718])
        fit_sequence_seed, fit_tie_seed, test_sequence_seed, test_tie_seed = seed_sequence.spawn(4)
        fit_network = copy.deepcopy(network)
        test_network = copy.deepcopy(network)
        fit_network.rng = np.random.default_rng(fit_tie_seed)
        test_network.rng = np.random.default_rng(test_tie_seed)
        fit_rng = np.random.default_rng(fit_sequence_seed)
        test_rng = np.random.default_rng(test_sequence_seed)

        fit_events: list[dict[str, object]] = []
        for _ in range(RELIABILITY_FIT_PATHS):
            sequence = [
                int(fit_rng.integers(0, task.n_symbols))
                for _ in range(RELIABILITY_FIT_HORIZON)
            ]
            _, events = evaluate_reliability_path(
                fit_network,
                task,
                sequence,
                start_state=int(fit_rng.integers(0, task.n_states)),
                helpers=helpers,
            )
            fit_events.extend(events)

        event_frame = pd.DataFrame(fit_events)
        label_kernel, representation_kernel, constant_error = estimate_survival_kernels(
            task, event_frame
        )
        label_initial = np.full(task.n_states, 1.0 / task.n_states)
        representation_initial = np.zeros(2 * task.n_states, dtype=np.float64)
        representation_initial[1::2] = 1.0 / task.n_states
        label_prediction = survival_curve(
            label_kernel, label_initial, RELIABILITY_TEST_HORIZON
        )
        representation_prediction = survival_curve(
            representation_kernel, representation_initial, RELIABILITY_TEST_HORIZON
        )
        constant_prediction = np.power(
            1.0 - constant_error, np.arange(1, RELIABILITY_TEST_HORIZON + 1)
        )
        label_rho, label_expected = kernel_diagnostics(label_kernel, label_initial)
        representation_rho, representation_expected = kernel_diagnostics(
            representation_kernel, representation_initial
        )

        test_survival: list[list[bool]] = []
        for _ in range(RELIABILITY_TEST_PATHS):
            sequence = [
                int(test_rng.integers(0, task.n_symbols))
                for _ in range(RELIABILITY_TEST_HORIZON)
            ]
            survival, _ = evaluate_reliability_path(
                test_network,
                task,
                sequence,
                start_state=int(test_rng.integers(0, task.n_states)),
                helpers=helpers,
            )
            test_survival.append(survival)
        observed = np.asarray(test_survival, dtype=np.float64).mean(axis=0)

        for depth in range(1, RELIABILITY_TEST_HORIZON + 1):
            rows.append(
                {
                    "seed": seed,
                    "L": depth,
                    "held_out_survival": observed[depth - 1],
                    "decoded_state_prediction": label_prediction[depth - 1],
                    "representation_prediction": representation_prediction[depth - 1],
                    "constant_hazard_prediction": constant_prediction[depth - 1],
                    "constant_error": constant_error,
                    "decoded_state_spectral_radius": label_rho,
                    "representation_spectral_radius": representation_rho,
                    "decoded_state_expected_survival": label_expected,
                    "representation_expected_survival": representation_expected,
                    "fit_transition_count": len(event_frame),
                }
            )
    return pd.DataFrame(rows)


def mean_interval(values: pd.Series) -> tuple[float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return mean, 0.0
    half = float(student_t.ppf(0.975, len(data) - 1)) * float(data.std(ddof=1)) / np.sqrt(len(data))
    return mean, half


def plot_trace(example: dict[str, object], output: Path) -> None:
    overlaps = np.asarray(example["overlaps"], dtype=float)
    sequence = list(example["sequence"])
    true_states = list(example["true_states"])
    fig, axis = plt.subplots(figsize=(7.1, 2.6))
    image = axis.imshow(overlaps.T, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axis.set_xlabel("Logical checkpoint")
    axis.set_ylabel("State assembly")
    axis.set_xticks(range(overlaps.shape[0]))
    axis.set_yticks(range(overlaps.shape[1]))
    axis.set_xticklabels([str(index) for index in range(overlaps.shape[0])])
    axis.set_yticklabels([f"$S_{{A,{state}}}$" for state in range(overlaps.shape[1])])
    for checkpoint, state in enumerate(true_states):
        axis.text(checkpoint, state, str(state), ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    top = axis.secondary_xaxis("top")
    top.set_xticks(range(1, len(sequence) + 1))
    top.set_xticklabels([str(symbol) for symbol in sequence])
    top.set_xlabel("Input symbol")
    colourbar = fig.colorbar(image, ax=axis, pad=0.02)
    colourbar.set_label("Assembly overlap")
    fig.tight_layout()
    save(fig, output, "dfa_overlap_trajectory")


def plot_primary(frame: pd.DataFrame, output: Path) -> None:
    grouped = frame.groupby(["c", "L"], as_index=False).agg(path_accuracy=("path_accuracy", "mean"))
    matrix = np.full((len(DWELL_TIMES), len(LENGTHS)), np.nan)
    for row_index, dwell in enumerate(DWELL_TIMES):
        for column_index, length in enumerate(LENGTHS):
            values = grouped.loc[
                (grouped["c"] == dwell) & (grouped["L"] == length), "path_accuracy"
            ]
            matrix[row_index, column_index] = float(values.iloc[0])

    fig, axis = plt.subplots(figsize=(5.8, 3.15))
    image = axis.imshow(matrix, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    axis.set_xticks(range(len(LENGTHS)), labels=[str(length) for length in LENGTHS])
    axis.set_yticks(range(len(DWELL_TIMES)), labels=[str(dwell) for dwell in DWELL_TIMES])
    axis.set(xlabel="Input length $L$", ylabel="Held updates per symbol $c$")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            colour = "black" if value > 0.72 else "white"
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center",
                      color=colour, fontsize=8)
    colourbar = fig.colorbar(image, ax=axis, pad=0.02)
    colourbar.set_label("Mean checkpoint accuracy")
    fig.tight_layout()
    save(fig, output, "dfa_path_accuracy")


def plot_reliability(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25))
    styles = [
        ("held_out_survival", "Held-out survival", "#0077BB", "-"),
        ("decoded_state_prediction", "Decoded-state kernel", "#EE7733", "--"),
        ("representation_prediction", "Representation-aware kernel", "#009988", "-."),
        ("constant_hazard_prediction", "Constant hazard", "#CC3311", ":"),
    ]
    for metric, label, colour, style in styles:
        per_depth = frame.groupby("L")[metric].mean()
        axes[0].plot(per_depth.index, per_depth.values, style, lw=1.8, color=colour, label=label)
        if metric == "held_out_survival":
            intervals = frame.groupby("L")[metric].apply(lambda values: mean_interval(values)[1])
            axes[0].fill_between(
                per_depth.index,
                np.clip(per_depth.values - intervals.values, 0, 1),
                np.clip(per_depth.values + intervals.values, 0, 1),
                color=colour,
                alpha=0.13,
                linewidth=0,
            )
    axes[0].axvline(RELIABILITY_FIT_HORIZON, color="0.45", lw=0.8, ls=":")
    axes[0].text(
        RELIABILITY_FIT_HORIZON + 2,
        0.96,
        "Beyond fit horizon",
        fontsize=7.5,
        color="0.35",
        va="top",
    )
    axes[0].set(
        xlabel="Logical depth $L$",
        ylabel="Complete-path survival",
        ylim=(-0.02, 1.02),
    )
    axes[0].legend(frameon=False, loc="upper right", fontsize=7.2)

    endpoint = frame[frame["L"] == RELIABILITY_TEST_HORIZON]
    axes[1].scatter(
        endpoint["representation_prediction"],
        endpoint["held_out_survival"],
        s=28,
        color="#009988",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].plot([0, 1], [0, 1], color="0.45", lw=0.9, ls=":")
    axes[1].set(
        xlabel=f"Kernel prediction at $L={RELIABILITY_TEST_HORIZON}$",
        ylabel="Held-out survival",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    axes[0].text(-0.16, 1.05, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.16, 1.05, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout()
    save(fig, output, "dfa_reliability_kernel_prediction")


def plot_holding(frame: pd.DataFrame, output: Path) -> None:
    conditional = frame[frame["prefix_correct"]].copy()
    per_seed = conditional.groupby(["condition", "seed", "d"], as_index=False).agg(
        holding_probability=("continuously_held", "mean")
    )
    styles = (
        ("learned_hold", "Learned HOLD operation", "#0077BB"),
        ("untrained_hold", "Untrained HOLD input", "#CC3311"),
        ("repeat_final_symbol", "Repeat final task symbol", "#EE7733"),
    )

    fig, axis = plt.subplots(figsize=(5.8, 3.25))
    for condition, label, colour in styles:
        subset = per_seed[per_seed["condition"] == condition]
        means = subset.groupby("d")["holding_probability"].mean()
        intervals = subset.groupby("d")["holding_probability"].apply(
            lambda values: mean_interval(values)[1]
        )
        axis.plot(means.index, means.values, color=colour, lw=1.9, label=label)
        axis.fill_between(
            means.index,
            np.clip(means.values - intervals.values, 0, 1),
            np.clip(means.values + intervals.values, 0, 1),
            color=colour,
            alpha=0.14,
            linewidth=0,
        )
    axis.set(
        xlabel="Updates after completion $d$",
        ylabel="Continuous holding probability",
        xlim=(0, HOLD_MAX_LAG),
        ylim=(-0.02, 1.02),
    )
    axis.legend(frameon=False, loc="center right")
    fig.tight_layout()
    save(fig, output, "dfa_holding_ablation")


def plot_pointer(pointer_raw: Path, output: Path) -> None:
    frame = pd.read_csv(pointer_raw)
    grouped = frame.groupby(["L", "t"], as_index=False).agg(
        accuracy=("accuracy", "mean"), path_accuracy=("path_accuracy", "mean")
    )
    lengths = sorted(grouped["L"].astype(int).unique())
    budgets = sorted(grouped["t"].astype(int).unique())
    matrix = np.full((len(lengths), len(budgets)), np.nan)
    for row_index, length in enumerate(lengths):
        for column_index, budget in enumerate(budgets):
            values = grouped.loc[(grouped["L"] == length) & (grouped["t"] == budget), "accuracy"]
            if not values.empty:
                matrix[row_index, column_index] = float(values.iloc[0])

    fig, axis = plt.subplots(figsize=(6.3, 3.35))
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="cividis", vmin=0, vmax=1)
    axis.set_xticks(range(len(budgets)), labels=[str(value) for value in budgets])
    axis.set_yticks(range(len(lengths)), labels=[str(value) for value in lengths])
    axis.set(xlabel="Internal update budget $t$", ylabel="Pointer depth $L$")
    for length in lengths:
        if length in budgets:
            axis.plot(budgets.index(length), lengths.index(length), marker="s", ms=5,
                      markerfacecolor="none", markeredgecolor="white", markeredgewidth=0.8)
    colourbar = fig.colorbar(image, ax=axis, pad=0.02)
    colourbar.set_label("Completion-checkpoint accuracy")
    fig.tight_layout()
    save(fig, output, "pointer_execution_boundary_clean")

    fig, axis = plt.subplots(figsize=(5.8, 3.25))
    selected = (0, 2, 4, 6, 8)
    for colour, budget in zip(COLOURS, selected):
        subset = grouped[grouped["t"] == budget].sort_values("L")
        axis.plot(subset["L"], subset["path_accuracy"], marker="o", ms=3.4,
                  lw=1.5, color=colour, label=f"$t={budget}$")
    axis.set(xlabel="Pointer depth $L$", ylabel="Mean path progress", ylim=(-0.02, 1.02))
    axis.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    save(fig, output, "pointer_path_progress_clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--pointer-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    helpers = import_pyac(args.ac_root)
    primary, example = run_primary(helpers)
    reliability = run_reliability(helpers)
    holding = run_holding(helpers)
    primary.to_csv(args.output / "dfa_learned_raw.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    reliability.to_csv(args.output / "dfa_reliability_kernel_raw.csv", index=False)
    holding.to_csv(args.output / "dfa_holding_ablation_raw.csv", index=False)
    (args.output / "dfa_trace.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    metadata = {
        "primary_seeds": list(PRIMARY_SEEDS),
        "states": 5,
        "symbols": 2,
        "plasticity": 0.25,
        "training_rounds": 12,
        "normalisation": "one incoming budget across all fibres to each area",
        "primary": {
            "assembly_size": 24,
            "density": 1.0,
            "lengths": list(LENGTHS),
            "held_updates_per_symbol": list(DWELL_TIMES),
            "sequences_per_seed_cell": 20,
            "symbol_protocol": "same symbol held throughout each dwell interval",
        },
        "reliability": {
            "seeds": list(RELIABILITY_SEEDS),
            "assembly_size": 10,
            "density": 0.2,
            "fit_horizon": RELIABILITY_FIT_HORIZON,
            "test_horizon": RELIABILITY_TEST_HORIZON,
            "fit_sequences_per_seed": RELIABILITY_FIT_PATHS,
            "test_sequences_per_seed": RELIABILITY_TEST_PATHS,
            "pre_specified_models": [
                "decoded-state substochastic kernel",
                "decoded-state plus exact-reference-cap substochastic kernel",
                "constant correct-prefix hazard",
            ],
        },
        "holding": {
            "seeds": list(HOLD_SEEDS),
            "assembly_size": 24,
            "density": 1.0,
            "binary_task_symbols": 2,
            "total_symbols": 3,
            "identity_hold_symbol": HOLD_SYMBOL,
            "sequence_length": HOLD_SEQUENCE_LENGTH,
            "post_completion_lag": list(range(HOLD_MAX_LAG + 1)),
            "sequences_per_seed": HOLD_SEQUENCES_PER_SEED,
            "conditions": ["learned_hold", "untrained_hold", "repeat_final_symbol"],
            "paired_protocol": "binary-trained network cloned before HOLD-only training",
        },
        "plasticity_during_evaluation": False,
    }
    (args.output / "dfa_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_trace(example, args.output)
    plot_primary(primary, args.output)
    plot_reliability(reliability, args.output)
    plot_holding(holding, args.output)
    plot_pointer(args.pointer_raw, args.output)


if __name__ == "__main__":
    main()
