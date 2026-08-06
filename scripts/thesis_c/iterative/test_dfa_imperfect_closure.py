from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_iterative_module(ac_root: Path):
    path = ac_root / "scripts" / "thesis_c" / "iterative" / "generate_iterative_results.py"
    spec = importlib.util.spec_from_file_location("iterative_results", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(data) - 1) * stats.sem(data))
    return mean, mean - half, mean + half


def corrupted_cap(
    task,
    state: int,
    area_size: int,
    replacements: int,
    rng: np.random.Generator,
) -> np.ndarray:
    reference = np.asarray(task.state_assemblies[state].indices, dtype=np.int64)
    if replacements == 0:
        return reference.copy()
    retained = rng.choice(reference, size=len(reference) - replacements, replace=False)
    outside = np.setdiff1d(np.arange(area_size, dtype=np.int64), reference, assume_unique=True)
    distractors = rng.choice(outside, size=replacements, replace=False)
    return np.sort(np.concatenate([retained, distractors])).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_ints, default=(42, 43, 44, 45, 46))
    parser.add_argument("--replacements", type=parse_ints, default=(0, 4, 8, 12, 14, 16, 18))
    parser.add_argument("--paths", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=80)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    iterative, source_script = load_iterative_module(args.ac_root)
    helpers = iterative.import_pyac(args.ac_root)
    clear, decode, stimulus, _, _ = helpers
    curve_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        print(f"Controlled DFA seed={seed}", flush=True)
        base_network, task, _ = iterative.build_trained(
            seed, helpers, assembly_size=24, density=1.0
        )
        task_rng = np.random.default_rng(np.random.SeedSequence([seed, 20260806]))
        paths = [
            (
                int(task_rng.integers(0, task.n_states)),
                [int(task_rng.integers(0, task.n_symbols)) for _ in range(args.horizon)],
            )
            for _ in range(args.paths)
        ]
        sym, cur = task.area_map["sym"], task.area_map["cur"]
        current_n = base_network.areas_by_name[cur].n
        k = len(task.state_assemblies[0].indices)

        for replacements in args.replacements:
            if not 0 <= replacements < k:
                raise ValueError("replacements must lie in [0, k)")
            network = copy.deepcopy(base_network)
            network.rng = np.random.default_rng(
                np.random.SeedSequence([seed, replacements, 20260806, 1])
            )
            corruption_rng = np.random.default_rng(
                np.random.SeedSequence([seed, replacements, 20260806, 2])
            )
            decoded_counts = np.zeros(args.horizon, dtype=np.int64)
            exact_counts = np.zeros(args.horizon, dtype=np.int64)
            transitions = 0
            correct_transitions = 0
            exact_destinations = 0

            for start_state, sequence in paths:
                clear(network)
                true_state = start_state
                decoded_alive = True
                exact_alive = True
                for depth, symbol in enumerate(sequence, start=1):
                    if not decoded_alive:
                        break
                    for _ in range(1000):
                        source_cap = corrupted_cap(
                            task,
                            true_state,
                            current_n,
                            replacements,
                            corruption_rng,
                        )
                        if decode(task, source_cap) == true_state:
                            break
                    else:
                        raise RuntimeError("could not sample a correctly decoded corrupted cap")
                    network.activations[cur] = source_cap
                    true_state = task.delta[(true_state, symbol)]
                    network.step(
                        external_stimuli={
                            sym: stimulus(
                                network.areas_by_name[sym].n,
                                task.sym_assemblies[symbol].indices,
                            )
                        },
                        plasticity_on=False,
                    )
                    prediction = decode(task, network.activations[cur])
                    correct = prediction == true_state
                    destination_exact = correct and iterative.exact_reference_cap(
                        task, network.activations[cur], true_state
                    )
                    transitions += 1
                    correct_transitions += int(correct)
                    exact_destinations += int(destination_exact)
                    decoded_alive = decoded_alive and correct
                    exact_alive = exact_alive and destination_exact
                    decoded_counts[depth - 1] += int(decoded_alive)
                    exact_counts[depth - 1] += int(exact_alive)

            for depth in range(1, args.horizon + 1):
                curve_rows.append(
                    {
                        "seed": seed,
                        "replacements": replacements,
                        "source_overlap": (k - replacements) / k,
                        "L": depth,
                        "decoded_path_survival": decoded_counts[depth - 1] / args.paths,
                        "exact_destination_path_survival": exact_counts[depth - 1] / args.paths,
                    }
                )
            transition_rows.append(
                {
                    "seed": seed,
                    "replacements": replacements,
                    "source_overlap": (k - replacements) / k,
                    "evaluated_transitions": transitions,
                    "decoded_transition_accuracy": correct_transitions / transitions,
                    "exact_destination_rate": exact_destinations / transitions,
                }
            )

    curves = pd.DataFrame(curve_rows)
    transitions = pd.DataFrame(transition_rows)
    curve_summary_rows: list[dict[str, object]] = []
    for (replacements, overlap, depth), group in curves.groupby(
        ["replacements", "source_overlap", "L"], sort=True
    ):
        row: dict[str, object] = {
            "replacements": int(replacements),
            "source_overlap": float(overlap),
            "L": int(depth),
        }
        for column in ("decoded_path_survival", "exact_destination_path_survival"):
            mean, low, high = mean_ci(group[column])
            row[column] = mean
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        curve_summary_rows.append(row)
    curve_summary = pd.DataFrame(curve_summary_rows)

    transition_summary_rows: list[dict[str, object]] = []
    for (replacements, overlap), group in transitions.groupby(
        ["replacements", "source_overlap"], sort=True
    ):
        row: dict[str, object] = {
            "replacements": int(replacements),
            "source_overlap": float(overlap),
            "evaluated_transitions": int(group["evaluated_transitions"].sum()),
        }
        for column in ("decoded_transition_accuracy", "exact_destination_rate"):
            mean, low, high = mean_ci(group[column])
            row[column] = mean
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        transition_summary_rows.append(row)
    transition_summary = pd.DataFrame(transition_summary_rows)

    curves.to_csv(args.output / "dfa_corruption_per_seed_curve.csv", index=False)
    curve_summary.to_csv(args.output / "dfa_corruption_curve_summary.csv", index=False)
    transitions.to_csv(args.output / "dfa_corruption_transition_per_seed.csv", index=False)
    transition_summary.to_csv(args.output / "dfa_corruption_transition_summary.csv", index=False)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.ac_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=args.ac_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    metadata = {
        "protocol": {
            "seeds": list(args.seeds),
            "paths_per_seed_condition": args.paths,
            "horizon": args.horizon,
            "replacements": list(args.replacements),
            "states": 5,
            "symbols": 2,
            "assembly_size": 24,
            "density": 1.0,
            "plasticity": 0.25,
            "training_rounds": 12,
            "evaluation_plasticity": False,
            "intervention": "before every transition, replace m canonical current-state neurons with m neurons sampled without replacement from other state assemblies; reject and resample any cap that does not still decode as the required source state",
            "paired_sequences": True,
        },
        "source": {
            "repository_revision": revision,
            "repository_dirty": bool(dirty),
            "source_script": str(source_script),
            "source_script_sha256": hashlib.sha256(source_script.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    (args.output / "dfa_corruption_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nTRANSITION SUMMARY")
    print(transition_summary.to_string(index=False))
    print("\nPATH SURVIVAL")
    print(
        curve_summary[curve_summary["L"].isin([1, 5, 10, 20, 40, 80])].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
