from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = tuple(range(42, 62))
NODES = 50
MAX_DEPTH = 40
ASSEMBLY_SIZE = 16
DENSITY = 0.35
PLASTICITY = 0.25
BLUE = "#1769AA"
ORANGE = "#D97706"
TEAL = "#07877B"
GRID = "#D9E0E6"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.8,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def full_normalise(network, area_name: str | None = None) -> None:
    targets = [area_name] if area_name is not None else network.area_names
    for target in targets:
        keys = [key for key in network.weights if key[1] == target]
        if not keys:
            continue
        total = sum(np.asarray(network.weights[key].sum(axis=0)).ravel() for key in keys)
        total[total == 0.0] = 1.0
        for key in keys:
            matrix = network.weights[key]
            matrix.data = matrix.data / total[matrix.indices]


class ExactUnrolledPointerMLP:
    """An explicit untied stack of exact table-specific ReLU layers."""

    def __init__(self, pointer: np.ndarray, depth: int):
        self.pointer = np.asarray(pointer, dtype=np.int64)
        self.nodes = int(len(self.pointer))
        one_hop = np.zeros((self.nodes, self.nodes), dtype=np.float64)
        one_hop[np.arange(self.nodes), self.pointer] = 1.0
        self.weights = [one_hop.copy() for _ in range(depth)]
        self.biases = [np.zeros(self.nodes, dtype=np.float64) for _ in range(depth)]
        assert all(
            not np.shares_memory(left, right)
            for index, left in enumerate(self.weights)
            for right in self.weights[index + 1 :]
        )

    @property
    def dense_parameter_slots_per_hop(self) -> int:
        return self.nodes * self.nodes + self.nodes

    @property
    def nonzero_coefficients_per_hop(self) -> int:
        return int(np.count_nonzero(self.weights[0]))

    def apply(self, states: np.ndarray) -> np.ndarray:
        inputs = np.eye(self.nodes, dtype=np.float64)[states]
        for weight, bias in zip(self.weights, self.biases):
            inputs = np.maximum(inputs @ weight + bias, 0.0)
        return inputs.argmax(axis=1).astype(np.int64)

    @property
    def dense_parameter_slots(self) -> int:
        return len(self.weights) * self.dense_parameter_slots_per_hop

    @property
    def nonzero_coefficients(self) -> int:
        return sum(int(np.count_nonzero(weight)) for weight in self.weights)


def git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "scripts/thesis_c", "pyac/src/pyac"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    return revision, bool(status.strip())


def import_pointer_model(ac_root: Path):
    sys.path.insert(0, str(ac_root / "pyac" / "src"))
    from pyac.tasks.pointer import (  # type: ignore
        build_pointer_network,
        generate_full_cycle,
        rollout_seen_pointer_sequence,
        train_node_assemblies,
        train_seen_transitions,
    )

    return (
        build_pointer_network,
        generate_full_cycle,
        rollout_seen_pointer_sequence,
        train_node_assemblies,
        train_seen_transitions,
    )


def run(ac_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    build, generate_cycle, rollout, train_nodes, train_transitions = import_pointer_model(ac_root)
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        list_seed, network_seed = np.random.SeedSequence(seed).spawn(2)
        pointer = generate_cycle(NODES, np.random.default_rng(list_seed))
        network, task = build(
            num_lists=1,
            list_length=NODES,
            assembly_size=ASSEMBLY_SIZE,
            density=DENSITY,
            plasticity=PLASTICITY,
            rng=np.random.default_rng(network_seed),
        )
        full_normalise(network)
        network.normalize = types.MethodType(
            lambda self, target=None: full_normalise(self, target), network
        )
        train_nodes(network, task, presentation_rounds=4, settle_steps=1)
        train_transitions(
            network,
            task,
            [pointer],
            transition_rounds=12,
            association_steps=3,
        )

        starts = np.arange(NODES, dtype=np.int64)
        mlp_prefix = np.ones(NODES, dtype=bool)
        ac_sequences = {
            start: rollout(
                network,
                task,
                pointer,
                list_idx=0,
                start_node=int(start),
                internal_steps=MAX_DEPTH,
                settle_steps=1,
            )
            for start in starts
        }
        targets = starts.copy()

        for depth in range(1, MAX_DEPTH + 1):
            targets = pointer[targets]
            mlp = ExactUnrolledPointerMLP(pointer, depth)
            mlp_states = mlp.apply(starts)
            mlp_correct = mlp_states == targets
            mlp_prefix &= mlp_correct
            for index, start in enumerate(starts):
                ac_prediction = int(ac_sequences[int(start)][depth])
                ac_true_path = [int(start)]
                current = int(start)
                for _ in range(depth):
                    current = int(pointer[current])
                    ac_true_path.append(current)
                ac_path = ac_sequences[int(start)][: depth + 1]
                ac_prefix_correct = ac_path == ac_true_path
                rows.append(
                    {
                        "seed": seed,
                        "start_node": int(start),
                        "L": depth,
                        "target": int(targets[index]),
                        "mlp_prediction": int(mlp_states[index]),
                        "mlp_correct": int(mlp_correct[index]),
                        "mlp_path_correct": int(mlp_prefix[index]),
                        "mlp_blocks": depth,
                        "mlp_dense_parameter_slots": mlp.dense_parameter_slots,
                        "mlp_nonzero_coefficients": mlp.nonzero_coefficients,
                        "ac_prediction": ac_prediction,
                        "ac_correct": int(ac_prediction == targets[index]),
                        "ac_path_correct": int(ac_prefix_correct),
                        "ac_updates": depth,
                        "pointer": json.dumps(pointer.tolist()),
                    }
                )

    frame = pd.DataFrame(rows)
    if not bool((frame["mlp_path_correct"] == 1).all()):
        raise AssertionError("Exact unrolled MLP failed a matched seen-table path")
    if not bool((frame["ac_path_correct"] == 1).all()):
        failures = frame[frame["ac_path_correct"] == 0]
        raise AssertionError(
            f"Seen-map AC failed {len(failures)} matched rows; tune only from training evidence"
        )

    revision, relevant_paths_dirty = git_state(ac_root)
    metadata: dict[str, object] = {
        "protocol": "matched seen-table one-hop composition",
        "seeds": list(SEEDS),
        "nodes": NODES,
        "starts_per_table": NODES,
        "depths": list(range(1, MAX_DEPTH + 1)),
        "ac": {
            "assembly_size": ASSEMBLY_SIZE,
            "density": DENSITY,
            "plasticity": PLASTICITY,
            "node_training_rounds": 4,
            "transition_training_rounds": 12,
            "association_steps": 3,
            "updates_per_hop": 1,
            "recurrent_self_synapses": False,
            "expected_structural_synapses": 447720,
        },
        "mlp": {
            "construction": "table-specific N-wide ReLU transition layer, untied by hop",
            "execution": "a fresh depth-L model with L independently allocated weight and bias arrays",
            "dense_parameter_slots_per_hop": NODES * NODES + NODES,
            "nonzero_coefficients_per_hop": NODES,
            "precomputed_power_shortcuts": False,
        },
        "plasticity_during_evaluation": False,
        "software_revision": revision,
        "software_relevant_paths_dirty": relevant_paths_dirty,
    }
    return frame, metadata


def plot(frame: pd.DataFrame, output: Path) -> None:
    summary = frame.groupby("L", as_index=False).agg(
        mlp_accuracy=("mlp_path_correct", "mean"),
        ac_accuracy=("ac_path_correct", "mean"),
        mlp_parameters=("mlp_dense_parameter_slots", "first"),
        mlp_nonzero=("mlp_nonzero_coefficients", "first"),
        ac_updates=("ac_updates", "first"),
    )
    selected = summary[summary["L"].isin((1, 5, 10, 20, 30, 40))]

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))
    axes[0].plot(
        summary["mlp_parameters"] / 1000,
        summary["L"],
        color=ORANGE,
        lw=2.0,
    )
    axes[0].scatter(
        selected["mlp_parameters"] / 1000,
        selected["L"],
        color=ORANGE,
        s=24,
    )
    axes[0].set(
        xlabel="Dense MLP parameter slots (thousands)",
        ylabel="Supported pointer depth $L$",
        title="Feedforward depth uses layers",
    )

    axes[1].plot(summary["ac_updates"], summary["L"], color=BLUE, lw=2.0)
    axes[1].scatter(selected["ac_updates"], selected["L"], color=BLUE, s=24)
    axes[1].set(
        xlabel="AC internal updates $t$",
        title="One recurrent map is reused",
    )

    axes[2].plot(
        summary["ac_updates"],
        summary["mlp_parameters"] / 1000,
        color=TEAL,
        lw=2.0,
    )
    axes[2].scatter(
        selected["ac_updates"],
        selected["mlp_parameters"] / 1000,
        color=TEAL,
        s=24,
    )
    for _, row in selected.iterrows():
        axes[2].annotate(
            f"$L={int(row['L'])}$",
            (row["ac_updates"], row["mlp_parameters"] / 1000),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7,
        )
    axes[2].set(
        xlabel="AC internal updates $t$",
        ylabel="Dense MLP slots (thousands)",
        title="Matched successful depths",
    )
    for index, axis in enumerate(axes):
        axis.text(-0.18, 1.06, f"({chr(ord('a') + index)})", transform=axis.transAxes,
                  fontweight="bold")
    fig.tight_layout(w_pad=1.7)
    save(fig, output, "seen_time_size_three_panel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    frame, metadata = run(args.ac_root)
    frame.to_csv(args.output / "seen_time_size_raw.csv", index=False)
    (args.output / "seen_time_size_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot(frame, args.output)


if __name__ == "__main__":
    main()
