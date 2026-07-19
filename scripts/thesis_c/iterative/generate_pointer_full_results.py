from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44, 45, 46)
BUDGETS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))

    from pyac.tasks.pointer import (  # type: ignore
        build_pointer_network,
        evaluate_seen_per_instance,
        generate_unique_lists,
        train_node_assemblies,
        train_seen_transitions,
    )

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        seed_sequence = np.random.SeedSequence(seed)
        list_seed, network_seed, evaluation_seed = seed_sequence.spawn(3)
        list_rng = np.random.default_rng(list_seed)
        network_rng = np.random.default_rng(network_seed)
        evaluation_rng = np.random.default_rng(evaluation_seed)
        pointers = generate_unique_lists(10, 10, list_rng)
        network, task = build_pointer_network(
            num_lists=10,
            list_length=10,
            assembly_size=16,
            density=0.35,
            plasticity=0.25,
            rng=network_rng,
        )
        full_normalise(network)
        network.normalize = types.MethodType(
            lambda self, target=None: full_normalise(self, target), network
        )
        train_node_assemblies(network, task, presentation_rounds=4, settle_steps=1)
        train_seen_transitions(
            network,
            task,
            pointers,
            transition_rounds=12,
            association_steps=3,
        )
        for depth in range(1, 9):
            for budget in BUDGETS:
                result_rows = evaluate_seen_per_instance(
                    network,
                    task,
                    pointers,
                    hops=depth,
                    time_budget=budget,
                    samples_per_list=32,
                    rng=evaluation_rng,
                    theta_id=f"{seed}-pointer-full-normalisation",
                    settle_steps=1,
                )
                for row in result_rows:
                    rows.append(
                        {
                            "seed": seed,
                            "L": row["L"],
                            "t": row["t"],
                            "list_idx": row["list_idx"],
                            "sample_idx": row["sample_idx"],
                            "start_node": row["start_node"],
                            "target": row["target"],
                            "prediction": row["prediction"],
                            "accuracy": 1.0 if row["correct"] else 0.0,
                            "path_accuracy": row["path_accuracy"],
                            "first_error_index": row["first_error_index"],
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "pointer_full_normalisation_raw.csv", index=False)
    metadata = {
        "seeds": list(SEEDS),
        "num_pointer_tables": 10,
        "nodes_per_table": 10,
        "assembly_size": 16,
        "density": 0.35,
        "plasticity": 0.25,
        "presentation_rounds": 4,
        "transition_rounds": 12,
        "association_steps": 3,
        "samples_per_table": 32,
        "time_budgets": list(BUDGETS),
        "normalisation": "one incoming budget across all fibres to each area",
        "plasticity_during_evaluation": False,
        "readout": "completion checkpoint min(t, L)",
    }
    (args.output / "pointer_full_normalisation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
