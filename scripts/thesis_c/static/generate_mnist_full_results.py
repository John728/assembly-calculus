from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44, 45, 46)
EXPOSURES = (1, 10, 100)
CUE_DURATIONS = (1, 2)
RETENTION_LAGS = (0, 1, 2, 4, 8, 10, 20, 50)
TIME_VALUES = (0, 1, 2, 4, 8, 10, 20, 40, 60, 80, 100)


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


def compact_retention(row: dict[str, object], seed: int, exposure: int) -> dict[str, object]:
    return {
        "seed": seed,
        "presentation_rounds": exposure,
        "cue_duration_s": row["cue_duration_s"],
        "retention_ell": row["retention_ell"],
        "instance_id": row["instance_id"],
        "target": row["target"],
        "prediction": row["prediction"],
        "accuracy": 1.0 if row["correct"] else 0.0,
        "correct_overlap": row["correct_overlap"],
        "strongest_wrong_overlap": row["strongest_wrong_overlap"],
        "margin": row["margin"],
        "correct_at_t1": row["correct_at_t1"],
        "stayed_correct": row["stayed_correct"],
        "became_correct_later": row["became_correct_later"],
        "first_error_index": row["first_error_index"],
        "retention_time": row["retention_time"],
        "retained_full_horizon": row["retained_full_horizon"],
        "normalisation": "full_incoming",
        "plasticity_on": False,
    }


def compact_time(row: dict[str, object], seed: int, mode: str) -> dict[str, object]:
    return {
        "seed": seed,
        "t": row["t"],
        "instance_id": row["instance_id"],
        "stimulus_mode": mode,
        "target": row["target"],
        "prediction": row["prediction"],
        "accuracy": 1.0 if row["correct"] else 0.0,
        "correct_overlap": row["correct_overlap"],
        "strongest_wrong_overlap": row["strongest_wrong_overlap"],
        "margin": row["margin"],
        "normalisation": "full_incoming",
        "plasticity_on": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))

    from pyac.tasks.mnist import (  # type: ignore
        RawPixelEncoder,
        build_mnist_network,
        evaluate_mnist_retention_sweep,
        evaluate_mnist_t_sweep,
        load_mnist_split,
        train_mnist_assemblies,
    )

    data_dir = args.ac_root / "data" / "mnist"
    train = load_mnist_split(data_dir, "train")
    test = load_mnist_split(data_dir, "test")
    train_images, train_labels = train.images[:500], train.labels[:500]
    test_images, test_labels = test.images[:50], test.labels[:50]

    retention_rows: list[dict[str, object]] = []
    held_rows: list[dict[str, object]] = []
    trace_candidates: list[dict[str, object]] = []

    for seed in SEEDS:
        for exposure in EXPOSURES:
            rng = np.random.default_rng(seed)
            encoder = RawPixelEncoder(k=200, area_name="X")
            network, task = build_mnist_network(
                n=2000, k=200, p=0.1, beta=0.5, rng=rng, encoder=encoder
            )
            task.seed = seed
            full_normalise(network)
            network.normalize = types.MethodType(
                lambda self, target=None: full_normalise(self, target), network
            )
            train_mnist_assemblies(
                network,
                task,
                train_images,
                train_labels,
                presentation_rounds=exposure,
                settle_steps=1,
                class_organized=True,
                normalization_on=True,
            )

            raw_retention = evaluate_mnist_retention_sweep(
                network,
                task,
                test_images,
                test_labels,
                cue_duration_values=list(CUE_DURATIONS),
                retention_ell_values=list(RETENTION_LAGS),
                instance_ids=list(range(len(test_images))),
            )
            retention_rows.extend(
                compact_retention(row, seed, exposure) for row in raw_retention
            )
            if seed == 42 and exposure == 100:
                trace_candidates.extend(raw_retention)

            if exposure == 1:
                for mode in ("held", "transient"):
                    raw_time = evaluate_mnist_t_sweep(
                        network,
                        task,
                        test_images,
                        test_labels,
                        t_values=list(TIME_VALUES),
                        instance_ids=list(range(len(test_images))),
                        stimulus_mode=mode,
                    )
                    held_rows.extend(compact_time(row, seed, mode) for row in raw_time)

    retention = pd.DataFrame(retention_rows)
    held = pd.DataFrame(held_rows)
    retention.to_csv(args.output / "mnist_full_retention_raw.csv", index=False)
    held.to_csv(args.output / "mnist_full_held_removed_raw.csv", index=False)

    removal = [
        row for row in trace_candidates
        if int(row["cue_duration_s"]) == 2 and int(row["retention_ell"]) == 0
        and float(row["margin"]) > 0
    ]
    median_margin = float(np.median([float(row["margin"]) for row in removal]))
    chosen = min(
        removal,
        key=lambda row: (abs(float(row["margin"]) - median_margin), int(row["instance_id"])),
    )
    trace_row = next(
        row for row in trace_candidates
        if int(row["cue_duration_s"]) == 2 and int(row["retention_ell"]) == 50
        and int(row["instance_id"]) == int(chosen["instance_id"])
    )
    trace = {
        "selection_rule": "seed 42, R=100, s=2; correct at removal; nearest median removal margin; instance-id tie break",
        "seed": 42,
        "presentation_rounds": 100,
        "cue_duration_s": 2,
        "instance_id": int(trace_row["instance_id"]),
        "target": int(trace_row["target"]),
        "removal_margin": float(chosen["margin"]),
        "overlap_trajectory": trace_row["overlap_trajectory"],
    }
    (args.output / "mnist_full_representative_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )

    metadata = {
        "seeds": list(SEEDS),
        "train_images": 500,
        "test_images": 50,
        "n": 2000,
        "k": 200,
        "p": 0.1,
        "beta": 0.5,
        "raw_input_k": 200,
        "class_organised": True,
        "presentation_rounds": list(EXPOSURES),
        "cue_durations": list(CUE_DURATIONS),
        "retention_lags": list(RETENTION_LAGS),
        "normalisation": "one incoming budget across sensory and recurrent fibres",
        "plasticity_during_evaluation": False,
        "held_removed_model": "R=1",
        "software_revision": "43500a3d3437dd0d67387a8eaa766a11ff4074e0",
        "software_worktree_dirty": True,
    }
    (args.output / "mnist_full_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
