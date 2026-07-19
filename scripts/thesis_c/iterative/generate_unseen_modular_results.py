from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44, 45, 46)
CONTROLLER_FIBRES = (
    ("current", "control"),
    ("control", "source"),
    ("destination", "control"),
    ("control", "current"),
)


@dataclass
class ModularPointerTask:
    nodes: int
    assembly_size: int
    node_assemblies: dict[str, dict[int, np.ndarray]]
    query_assemblies: dict[int, np.ndarray]
    writeback_assemblies: dict[int, np.ndarray]
    memory_baseline: object


def stimulus(size: int, indices: np.ndarray, strength: float = 10.0) -> np.ndarray:
    values = np.zeros(size, dtype=np.float64)
    values[np.asarray(indices, dtype=np.int64)] = strength
    return values


def clear(network, *areas_to_keep: str) -> None:
    keep = set(areas_to_keep)
    for area in network.area_names:
        if area not in keep:
            network.activations[area] = np.array([], dtype=np.int64)


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


def blocks(area: str, count: int, assembly_size: int, offset: int = 0) -> dict[int, np.ndarray]:
    del area
    return {
        index: np.arange(
            offset + index * assembly_size,
            offset + (index + 1) * assembly_size,
            dtype=np.int64,
        )
        for index in range(count)
    }


def build_network(
    pyac_root: Path,
    *,
    nodes: int,
    assembly_size: int,
    controller_density: float,
    memory_density: float,
    plasticity: float,
    rng: np.random.Generator,
):
    sys.path.insert(0, str(pyac_root / "pyac" / "src"))
    from pyac.core.network import Network  # type: ignore
    from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec  # type: ignore

    node_n = nodes * assembly_size
    control_n = 2 * node_n
    spec = NetworkSpec(
        areas=[
            AreaSpec(name="control", n=control_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="source", n=node_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="destination", n=node_n, k=assembly_size, dynamics_type="feedforward"),
            AreaSpec(name="current", n=node_n, k=assembly_size, dynamics_type="feedforward"),
        ],
        fibers=[
            FiberSpec(src="current", dst="control", p_fiber=controller_density),
            FiberSpec(src="control", dst="source", p_fiber=controller_density),
            FiberSpec(src="source", dst="destination", p_fiber=memory_density),
            FiberSpec(src="destination", dst="control", p_fiber=controller_density),
            FiberSpec(src="control", dst="current", p_fiber=controller_density),
        ],
        beta=plasticity,
        step_order="sequential",
    )
    network = Network(spec, rng)
    network.normalize = types.MethodType(
        lambda self, target=None: full_normalise(self, target), network
    )
    full_normalise(network)

    node_assemblies = {
        area: blocks(area, nodes, assembly_size)
        for area in ("current", "source", "destination")
    }
    query = blocks("control", nodes, assembly_size)
    writeback = blocks("control", nodes, assembly_size, offset=node_n)
    task = ModularPointerTask(
        nodes=nodes,
        assembly_size=assembly_size,
        node_assemblies=node_assemblies,
        query_assemblies=query,
        writeback_assemblies=writeback,
        memory_baseline=network.weights[("source", "destination")].copy(),
    )
    return network, task


def train_controller(network, task: ModularPointerTask, rounds: int) -> None:
    sizes = {name: network.areas_by_name[name].n for name in network.area_names}
    for _ in range(rounds):
        for node in range(task.nodes):
            clear(network)
            network.activations["current"] = task.node_assemblies["current"][node].copy()
            network.inhibit("current")
            network.inhibit("destination")
            network.step(
                external_stimuli={
                    "control": stimulus(sizes["control"], task.query_assemblies[node]),
                    "source": stimulus(
                        sizes["source"], task.node_assemblies["source"][node]
                    ),
                },
                plasticity_on=True,
            )
            network.disinhibit("current")
            network.disinhibit("destination")

            clear(network)
            network.activations["destination"] = (
                task.node_assemblies["destination"][node].copy()
            )
            network.inhibit("source")
            network.inhibit("destination")
            network.step(
                external_stimuli={
                    "control": stimulus(sizes["control"], task.writeback_assemblies[node]),
                    "current": stimulus(
                        sizes["current"], task.node_assemblies["current"][node]
                    ),
                },
                plasticity_on=True,
            )
            network.disinhibit("source")
            network.disinhibit("destination")

        full_normalise(network)

    full_normalise(network)
    clear(network)
    task.memory_baseline = network.weights[("source", "destination")].copy()


def reset_memory(network, task: ModularPointerTask) -> None:
    network.weights[("source", "destination")] = copy.deepcopy(task.memory_baseline)
    clear(network)


def controller_snapshot(network) -> dict[tuple[str, str], object]:
    return {fibre: network.weights[fibre].copy() for fibre in CONTROLLER_FIBRES}


def assert_controller_unchanged(network, snapshot: dict[tuple[str, str], object]) -> None:
    for fibre, expected in snapshot.items():
        difference = network.weights[fibre] != expected
        assert difference.nnz == 0, f"controller fibre changed during evaluation: {fibre}"


def write_memory(
    network,
    task: ModularPointerTask,
    pointer: np.ndarray,
    *,
    rounds: int,
) -> int:
    reset_memory(network, task)
    sizes = {name: network.areas_by_name[name].n for name in network.area_names}
    network.inhibit("control")
    network.inhibit("current")
    steps = 0
    for _ in range(rounds):
        for source, destination in enumerate(pointer.tolist()):
            clear(network)
            network.step(
                external_stimuli={
                    "source": stimulus(
                        sizes["source"], task.node_assemblies["source"][source]
                    ),
                    "destination": stimulus(
                        sizes["destination"],
                        task.node_assemblies["destination"][int(destination)],
                    ),
                },
                plasticity_on=True,
            )
            steps += 1
        full_normalise(network, "destination")
    network.disinhibit("control")
    network.disinhibit("current")
    full_normalise(network, "destination")
    clear(network)
    return steps


def decode(indices: np.ndarray, assemblies: dict[int, np.ndarray]) -> tuple[int, float]:
    active = set(int(index) for index in indices)
    scores = {
        node: len(active.intersection(int(index) for index in prototype)) / len(prototype)
        for node, prototype in assemblies.items()
    }
    winner = max(scores, key=lambda node: (scores[node], -node))
    return int(winner), float(scores[winner])


def query_phase(network, task: ModularPointerTask) -> dict[str, int | float]:
    clear(network, "current")
    network.inhibit("current")
    network.step(plasticity_on=False)
    network.disinhibit("current")
    query_node, query_overlap = decode(network.activations["control"], task.query_assemblies)
    source_node, source_overlap = decode(
        network.activations["source"], task.node_assemblies["source"]
    )
    destination_node, destination_overlap = decode(
        network.activations["destination"], task.node_assemblies["destination"]
    )
    return {
        "query_node": query_node,
        "query_overlap": query_overlap,
        "source_node": source_node,
        "source_overlap": source_overlap,
        "destination_node": destination_node,
        "destination_overlap": destination_overlap,
    }


def writeback_phase(network, task: ModularPointerTask) -> tuple[int, float]:
    clear(network, "destination")
    network.inhibit("source")
    network.inhibit("destination")
    network.step(plasticity_on=False)
    network.disinhibit("source")
    network.disinhibit("destination")
    return decode(network.activations["current"], task.node_assemblies["current"])


def rollout(
    network,
    task: ModularPointerTask,
    pointer: np.ndarray,
    *,
    start: int,
    hops: int,
) -> dict[str, object]:
    clear(network)
    network.activations["current"] = task.node_assemblies["current"][start].copy()
    true_path = [int(start)]
    decoded_path = [int(start)]
    phase_rows: list[dict[str, int | float]] = []
    current = int(start)
    for hop in range(1, hops + 1):
        current = int(pointer[current])
        true_path.append(current)
        query = query_phase(network, task)
        predicted, overlap = writeback_phase(network, task)
        decoded_path.append(predicted)
        phase_rows.append(
            {
                "hop": hop,
                **query,
                "current_node": predicted,
                "current_overlap": overlap,
            }
        )
    first_error = next(
        (
            index
            for index, (prediction, target) in enumerate(
                zip(decoded_path[1:], true_path[1:]), start=1
            )
            if prediction != target
        ),
        None,
    )
    return {
        "true_path": true_path,
        "decoded_path": decoded_path,
        "phase_rows": phase_rows,
        "path_correct": first_error is None,
        "path_accuracy": float(
            np.mean(np.asarray(decoded_path, dtype=int) == np.asarray(true_path, dtype=int))
        ),
        "first_error_index": first_error,
        "final_correct": decoded_path[-1] == true_path[-1],
    }


def cycle_table(rng: np.random.Generator, nodes: int) -> np.ndarray:
    order = rng.permutation(nodes)
    pointer = np.empty(nodes, dtype=np.int64)
    pointer[order] = np.roll(order, -1)
    return pointer


def software_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--assembly-size", type=int, default=16)
    parser.add_argument("--controller-density", type=float, default=1.0)
    parser.add_argument("--memory-density", type=float, default=1.0)
    parser.add_argument("--plasticity", type=float, default=0.3)
    parser.add_argument("--controller-rounds", type=int, default=12)
    parser.add_argument(
        "--write-rounds-values", type=int, nargs="+", default=[1, 2, 4, 10]
    )
    parser.add_argument("--tables-per-seed", type=int, default=20)
    parser.add_argument(
        "--starts-per-table", type=int, default=None,
        help="Number of distinct start nodes sampled per table; defaults to every node.",
    )
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    trace: dict[str, object] | None = None
    for seed in args.seeds:
        seed_sequence = np.random.SeedSequence(seed)
        network_seed, table_seed = seed_sequence.spawn(2)
        network, task = build_network(
            args.ac_root,
            nodes=args.nodes,
            assembly_size=args.assembly_size,
            controller_density=args.controller_density,
            memory_density=args.memory_density,
            plasticity=args.plasticity,
            rng=np.random.default_rng(network_seed),
        )
        train_controller(network, task, args.controller_rounds)
        frozen_controller = controller_snapshot(network)
        table_rng = np.random.default_rng(table_seed)
        table_keys: set[tuple[int, ...]] = set()
        for table_index in range(args.tables_per_seed):
            pointer = cycle_table(table_rng, args.nodes)
            pointer_key = tuple(int(value) for value in pointer)
            assert pointer_key not in table_keys
            table_keys.add(pointer_key)
            if args.starts_per_table is None or args.starts_per_table >= args.nodes:
                starts = np.arange(args.nodes, dtype=np.int64)
            else:
                starts = table_rng.choice(
                    args.nodes, size=args.starts_per_table, replace=False
                )
            for write_rounds in args.write_rounds_values:
                write_steps = write_memory(
                    network, task, pointer, rounds=write_rounds
                )
                for start_value in starts:
                    start = int(start_value)
                    result = rollout(
                        network,
                        task,
                        pointer,
                        start=start,
                        hops=args.max_depth,
                    )
                    true_path = result["true_path"]
                    decoded_path = result["decoded_path"]
                    for depth in range(1, args.max_depth + 1):
                        prefix_correct = all(
                            prediction == target
                            for prediction, target in zip(
                                decoded_path[1 : depth + 1], true_path[1 : depth + 1]
                            )
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "table_index": table_index,
                                "start_node": start,
                                "write_rounds": write_rounds,
                                "L": depth,
                                "t": 2 * depth,
                                "target": true_path[depth],
                                "prediction": decoded_path[depth],
                                "accuracy": float(decoded_path[depth] == true_path[depth]),
                                "path_correct": bool(prefix_correct),
                                "path_accuracy": float(
                                    np.mean(
                                        np.asarray(decoded_path[: depth + 1])
                                        == np.asarray(true_path[: depth + 1])
                                    )
                                ),
                                "first_error_index": result["first_error_index"],
                                "pointer": json.dumps(pointer.tolist()),
                                "write_steps": write_steps,
                                "rollout_plasticity_on": False,
                            }
                        )
                    if (
                        trace is None
                        and seed == 42
                        and table_index == 0
                        and start == int(starts[0])
                        and write_rounds == max(args.write_rounds_values)
                    ):
                        trace = {
                            "seed": seed,
                            "table_index": table_index,
                            "write_rounds": write_rounds,
                            "pointer": pointer.tolist(),
                            **result,
                        }
                assert_controller_unchanged(network, frozen_controller)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "pointer_unseen_modular_raw.csv", index=False)
    assert trace is not None
    (args.output / "pointer_unseen_modular_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )
    revision, dirty = software_state(args.ac_root)
    metadata = {
        "seeds": args.seeds,
        "nodes": args.nodes,
        "assembly_size": args.assembly_size,
        "controller_density": args.controller_density,
        "memory_density": args.memory_density,
        "plasticity": args.plasticity,
        "controller_rounds": args.controller_rounds,
        "write_rounds_values": args.write_rounds_values,
        "tables_per_seed": args.tables_per_seed,
        "starts_per_table": args.nodes if args.starts_per_table is None else args.starts_per_table,
        "maximum_depth": args.max_depth,
        "controller_training_tables": 0,
        "evaluation_tables": "unique new full-cycle permutations written after controller training",
        "memory_fibre": "source -> destination",
        "controller_fibres": [
            "current -> control",
            "control -> source",
            "destination -> control",
            "control -> current",
        ],
        "updates_per_hop": 2,
        "phase_schedule": "fixed query/writeback alternation with inactive workspaces reset between phases",
        "external_state_cues_during_rollout": 0,
        "controller_weight_audit": "bitwise unchanged after every table/write-strength evaluation",
        "memory_write_plasticity_on": True,
        "plasticity_during_rollout": False,
        "normalisation": "one incoming budget across all fibres to each area after every complete training or write round",
        "software_revision": revision,
        "software_worktree_dirty": dirty,
    }
    (args.output / "pointer_unseen_modular_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    summary = frame.groupby(["write_rounds", "L"]).agg(
        final_accuracy=("accuracy", "mean"),
        complete_path_accuracy=("path_correct", "mean"),
        mean_path_accuracy=("path_accuracy", "mean"),
        rows=("accuracy", "size"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
