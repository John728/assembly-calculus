from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44, 45, 46)
EXPOSURES = (1, 10, 100)
HORIZON = 100
TRAIN_IMAGES = 500
TEST_IMAGES = 50


def git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "scripts/thesis_c", "pyac/src/pyac"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, bool(status.strip())


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


def weight_structure(network, task) -> dict[str, float]:
    coding_area = task.area_map["coding"]
    recurrent = network.weights[(coding_area, coding_area)].tocsr()
    assemblies = [task.class_assemblies[digit].indices for digit in range(10)]
    all_indices = np.concatenate(assemblies)
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("class assemblies must be disjoint for the block-weight audit")
    if len(all_indices) != recurrent.shape[0]:
        raise ValueError("class assemblies must partition the coding area")

    within_mass = 0.0
    within_possible = 0
    between_mass = 0.0
    between_possible = 0
    for source_class, source in enumerate(assemblies):
        for target_class, target in enumerate(assemblies):
            mass = float(recurrent[source][:, target].sum())
            if source_class == target_class:
                within_mass += mass
                within_possible += len(source) * (len(target) - 1)
            else:
                between_mass += mass
                between_possible += len(source) * len(target)

    total_mass = within_mass + between_mass
    within_mean = within_mass / within_possible
    between_mean = between_mass / between_possible
    return {
        "within_mean_weight": within_mean,
        "between_mean_weight": between_mean,
        "within_weight_fraction": within_mass / total_mass,
        "mean_weight_contrast": within_mean - between_mean,
    }


def settling_readout(predictions: list[int]) -> int:
    changes = [
        index + 1
        for index in range(1, len(predictions))
        if predictions[index] != predictions[index - 1]
    ]
    return max(changes, default=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))

    from pyac.tasks.mnist import (  # type: ignore
        RawPixelEncoder,
        build_mnist_network,
        evaluate_mnist_example,
        load_mnist_split,
        train_mnist_assemblies,
    )

    data_dir = args.ac_root / "data" / "mnist"
    train = load_mnist_split(data_dir, "train")
    test = load_mnist_split(data_dir, "test")
    train_images, train_labels = train.images[:TRAIN_IMAGES], train.labels[:TRAIN_IMAGES]
    test_images, test_labels = test.images[:TEST_IMAGES], test.labels[:TEST_IMAGES]
    available_per_class = {
        str(digit): int(np.sum(train_labels == digit)) for digit in range(10)
    }

    trajectory_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    seed_summaries: list[dict[str, object]] = []

    for seed in SEEDS:
        for exposure in EXPOSURES:
            rng = np.random.default_rng(seed)
            encoder = RawPixelEncoder(k=200, area_name="X")
            network, task = build_mnist_network(
                n=2000,
                k=200,
                p=0.1,
                beta=0.5,
                rng=rng,
                encoder=encoder,
            )
            task.seed = seed
            coding_area = task.area_map["coding"]
            recurrent = network.weights[(coding_area, coding_area)]
            recurrent.setdiag(0.0)
            recurrent.eliminate_zeros()
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

            structure = weight_structure(network, task)
            weight_rows.append({"seed": seed, "presentation_rounds": exposure, **structure})
            example_summaries: list[dict[str, object]] = []

            for instance_id, (image, target) in enumerate(zip(test_images, test_labels)):
                result = evaluate_mnist_example(
                    network,
                    task,
                    image,
                    int(target),
                    instance_id=instance_id,
                    t=HORIZON - 1,
                    stimulus_mode="held",
                )
                predictions = [int(value) for value in result["trajectory"]]
                overlaps = np.asarray(result["overlap_trajectory"], dtype=float)
                if len(predictions) != HORIZON or overlaps.shape != (HORIZON, 10):
                    raise ValueError("the fixed-input trajectory does not match the requested horizon")

                tau = settling_readout(predictions)
                initial_prediction = predictions[0]
                final_prediction = predictions[-1]
                example_summaries.append(
                    {
                        "initial_correct": initial_prediction == int(target),
                        "final_correct": final_prediction == int(target),
                        "settling_readout": tau,
                    }
                )

                for index, prediction in enumerate(predictions):
                    readout = index + 1
                    rivals = [digit for digit in range(10) if digit != initial_prediction]
                    initial_overlap = float(overlaps[index, initial_prediction])
                    strongest_other = float(np.max(overlaps[index, rivals]))
                    trajectory_rows.append(
                        {
                            "seed": seed,
                            "presentation_rounds": exposure,
                            "instance_id": instance_id,
                            "readout_r": readout,
                            "target": int(target),
                            "prediction": prediction,
                            "correct": prediction == int(target),
                            "switched": index > 0 and prediction != predictions[index - 1],
                            "initial_prediction": initial_prediction,
                            "final_prediction": final_prediction,
                            "settling_readout": tau,
                            "settled_by_readout": readout >= tau,
                            "initial_winner_overlap": initial_overlap,
                            "strongest_other_overlap": strongest_other,
                            "initial_winner_margin": initial_overlap - strongest_other,
                        }
                    )

            initial_accuracy = float(np.mean([row["initial_correct"] for row in example_summaries]))
            final_accuracy = float(np.mean([row["final_correct"] for row in example_summaries]))
            seed_summaries.append(
                {
                    "seed": seed,
                    "presentation_rounds": exposure,
                    "initial_accuracy": initial_accuracy,
                    "final_accuracy": final_accuracy,
                    "accuracy_change": final_accuracy - initial_accuracy,
                    "maximum_settling_readout": int(
                        max(row["settling_readout"] for row in example_summaries)
                    ),
                    "fraction_settled_at_first_readout": float(
                        np.mean([row["settling_readout"] == 1 for row in example_summaries])
                    ),
                    **structure,
                }
            )

    trajectories = pd.DataFrame(trajectory_rows)
    weights = pd.DataFrame(weight_rows)
    trajectories.to_csv(args.output / "mnist_static_settling_raw.csv", index=False)
    weights.to_csv(args.output / "mnist_static_weight_structure.csv", index=False)

    revision, relevant_paths_dirty = git_state(args.ac_root)
    summary = {
        "protocol": {
            "seeds": list(SEEDS),
            "training_images": TRAIN_IMAGES,
            "test_images_per_network": TEST_IMAGES,
            "presentation_rounds": list(EXPOSURES),
            "presentation_rounds_semantics": "maximum distinct images used per class",
            "available_train_images_per_class": available_per_class,
            "horizon": HORIZON,
            "readout_indexing": "r=1 is the state after exactly one network update",
            "stimulus": "the same encoded image is applied at every update",
            "plasticity_during_evaluation": False,
            "normalisation": "one incoming budget across sensory and recurrent fibres",
            "recurrent_self_synapses": False,
        },
        "per_seed": seed_summaries,
        "software_revision": revision,
        "software_relevant_paths_dirty": relevant_paths_dirty,
    }
    (args.output / "mnist_static_settling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
