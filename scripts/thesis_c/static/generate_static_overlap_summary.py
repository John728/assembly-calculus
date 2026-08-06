from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEEDS = tuple(range(42, 52))


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_experiment(path: Path):
    spec = importlib.util.spec_from_file_location(
        "static_training_dynamics",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def evaluate_seed(
    *,
    seed: int,
    horizon: int,
    pyac,
    base,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scheduled_images, scheduled_labels = base.build_training_schedule(
        train_images,
        train_labels,
        "distinct_50",
    )
    network, task, _ = base.train_model(
        seed=seed,
        images=scheduled_images,
        labels=scheduled_labels,
        pyac=pyac,
    )

    correct_overlaps: list[np.ndarray] = []
    strongest_rivals: list[np.ndarray] = []
    initial_winner_overlaps: list[np.ndarray] = []
    initial_alternative_overlaps: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[int] = []
    for instance_id, (image, target_value) in enumerate(
        zip(test_images, test_labels)
    ):
        target = int(target_value)
        result = pyac.evaluate_mnist_example(
            network,
            task,
            image,
            target,
            instance_id=instance_id,
            t=horizon - 1,
            stimulus_mode="held",
        )
        overlaps = np.asarray(result["overlap_trajectory"], dtype=float)
        trajectory = np.asarray(result["trajectory"], dtype=int)
        if overlaps.shape != (horizon, 10):
            raise ValueError(f"unexpected overlap shape: {overlaps.shape}")
        if trajectory.shape != (horizon,):
            raise ValueError(f"unexpected prediction shape: {trajectory.shape}")

        competitors = overlaps.copy()
        competitors[:, target] = -np.inf
        initial_winner = int(trajectory[0])
        initial_alternatives = overlaps[0].copy()
        initial_alternatives[initial_winner] = -np.inf
        correct_overlaps.append(overlaps[:, target])
        strongest_rivals.append(np.max(competitors, axis=1))
        initial_winner_overlaps.append(overlaps[:, initial_winner])
        initial_alternative_overlaps.append(
            float(np.max(initial_alternatives))
        )
        predictions.append(trajectory)
        targets.append(target)

    correct_array = np.asarray(correct_overlaps)
    rival_array = np.asarray(strongest_rivals)
    initial_winner_array = np.asarray(initial_winner_overlaps)
    initial_alternative_array = np.asarray(initial_alternative_overlaps)
    prediction_array = np.asarray(predictions)
    target_array = np.asarray(targets)
    final_correct = prediction_array[:, -1] == target_array
    corrected = (
        (prediction_array[:, 0] != target_array)
        & final_correct
    )
    gain_by_class = base.class_within_gains(network, task)
    cohorts = {
        "all": np.ones(len(target_array), dtype=bool),
        "final_correct": final_correct,
        "corrected": corrected,
    }

    rows: list[dict[str, object]] = []
    for cohort, mask in cohorts.items():
        if not np.any(mask):
            raise ValueError(f"seed {seed} has no examples in cohort {cohort}")
        for index in range(horizon):
            rows.append(
                {
                    "seed": seed,
                    "condition": "distinct_50",
                    "cohort": cohort,
                    "readout_r": index + 1,
                    "examples": int(np.sum(mask)),
                    "correct_class_overlap": float(
                        np.mean(correct_array[mask, index])
                    ),
                    "strongest_rival_overlap": float(
                        np.mean(rival_array[mask, index])
                    ),
                    "initial_winner_overlap": float(
                        np.mean(initial_winner_array[mask, index])
                    ),
                    "accuracy": float(
                        np.mean(
                            prediction_array[:, index] == target_array
                        )
                    ),
                }
            )
    trajectory_rows: list[dict[str, object]] = []
    for instance_id, target in enumerate(target_array):
        predictions_for_example = prediction_array[instance_id]
        initial_prediction = int(predictions_for_example[0])
        final_prediction = int(predictions_for_example[-1])
        initial_target_gap = float(
            initial_winner_array[instance_id, 0]
            - correct_array[instance_id, 0]
        )
        gain_advantage = float(
            gain_by_class[int(target)] - gain_by_class[initial_prediction]
        )
        switch_count = int(
            np.sum(
                predictions_for_example[1:]
                != predictions_for_example[:-1]
            )
        )
        trajectory_rows.append(
            {
                "seed": seed,
                "condition": "distinct_50",
                "instance_id": instance_id,
                "target": int(target),
                "initial_prediction": initial_prediction,
                "final_prediction": final_prediction,
                "transition_type": base.classify_transition(
                    target=int(target),
                    initial=initial_prediction,
                    final=final_prediction,
                ),
                "switch_count": switch_count,
                "settling_readout": base.last_change_readout(
                    predictions_for_example.tolist()
                ),
                "initial_winner_overlap": float(
                    initial_winner_array[instance_id, 0]
                ),
                "initial_target_overlap": float(
                    correct_array[instance_id, 0]
                ),
                "initial_target_gap": initial_target_gap,
                "initial_strongest_rival_overlap": float(
                    rival_array[instance_id, 0]
                ),
                "initial_strongest_alternative_overlap": float(
                    initial_alternative_array[instance_id]
                ),
                "target_is_initial_strongest_alternative": bool(
                    np.isclose(
                        correct_array[instance_id, 0],
                        initial_alternative_array[instance_id],
                    )
                ),
                "initial_winner_gain": float(
                    gain_by_class[initial_prediction]
                ),
                "target_gain": float(gain_by_class[int(target)]),
                "gain_advantage": gain_advantage,
                "overtaking_difficulty": (
                    initial_target_gap / gain_advantage
                    if gain_advantage > 0.0
                    else np.nan
                ),
            }
        )
    return rows, trajectory_rows


def validate_accuracy(
    frame: pd.DataFrame,
    reference_path: Path,
) -> None:
    reference = pd.read_csv(reference_path)
    reference = reference[
        reference["condition"].eq("distinct_50")
        & reference["seed"].isin(frame["seed"].unique())
        & reference["readout_r"].isin(frame["readout_r"].unique())
    ][["seed", "readout_r", "accuracy"]]
    observed = frame[frame["cohort"].eq("all")][
        ["seed", "readout_r", "accuracy"]
    ]
    merged = observed.merge(
        reference,
        on=["seed", "readout_r"],
        suffixes=("_observed", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(observed):
        raise ValueError("reference accuracy does not cover every result")
    if not np.allclose(
        merged["accuracy_observed"],
        merged["accuracy_reference"],
        atol=1e-12,
        rtol=0.0,
    ):
        maximum_error = float(
            np.max(
                np.abs(
                    merged["accuracy_observed"]
                    - merged["accuracy_reference"]
                )
            )
        )
        raise ValueError(
            f"focused evaluation disagrees with reference by {maximum_error}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-script", type=Path)
    parser.add_argument("--reference-time-series", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--trajectory-csv", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--core-revision")
    args = parser.parse_args()

    if args.test_per_class <= 0 or args.horizon <= 0:
        raise ValueError("test-per-class and horizon must be positive")
    base_script = (
        args.base_script
        if args.base_script is not None
        else Path(__file__).with_name(
            "generate_static_training_dynamics.py"
        )
    )
    base = load_base_experiment(base_script.resolve())
    pyac = base.load_pyac(args.ac_root.resolve())
    train = pyac.load_mnist_split(args.data_dir, "train")
    test = pyac.load_mnist_split(args.data_dir, "test")
    train_images, train_labels, train_ids = base.balanced_subset(
        train.images,
        train.labels,
        base.TRAIN_PER_CLASS,
    )
    test_images, test_labels, test_ids = base.balanced_subset(
        test.images,
        test.labels,
        args.test_per_class,
    )

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        seed_rows, seed_trajectories = evaluate_seed(
            seed=seed,
            horizon=args.horizon,
            pyac=pyac,
            base=base,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
        )
        rows.extend(seed_rows)
        trajectory_rows.extend(seed_trajectories)
    frame = pd.DataFrame(rows)
    validate_accuracy(frame, args.reference_time_series)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    trajectory_frame = pd.DataFrame(trajectory_rows)
    if args.trajectory_csv is not None:
        args.trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
        trajectory_frame.to_csv(args.trajectory_csv, index=False)
    metadata = {
        "protocol": {
            "seeds": list(args.seeds),
            "condition": "distinct_50",
            "train_images_per_class": base.TRAIN_PER_CLASS,
            "test_images_per_class": args.test_per_class,
            "horizon": args.horizon,
            "stimulus_during_evaluation": "held",
            "plasticity_during_evaluation": False,
            "correct_overlap": "overlap with the labelled class assembly",
            "strongest_rival_overlap": (
                "maximum overlap over all non-labelled class assemblies "
                "at each update"
            ),
            "initial_winner_overlap": (
                "overlap with the fixed assembly predicted at update 1"
            ),
            "cohorts": {
                "all": "all balanced test examples",
                "final_correct": (
                    "fixed subset classified correctly at the final readout"
                ),
                "corrected": (
                    "fixed subset wrong at update 1 and correct at the "
                    "final readout"
                ),
            },
            "balanced_train_source_indices": train_ids.tolist(),
            "balanced_test_source_indices": test_ids.tolist(),
        },
        "validation": {
            "reference_time_series": str(
                args.reference_time_series.resolve()
            ),
            "reference_sha256": sha256(args.reference_time_series),
            "accuracy_exactly_reproduced": True,
        },
        "software": {
            "experiment_script": str(Path(__file__).resolve()),
            "experiment_script_sha256": sha256(Path(__file__).resolve()),
            "base_experiment_script": str(base_script.resolve()),
            "base_experiment_script_sha256": sha256(base_script),
            "ac_root": str(args.ac_root.resolve()),
            "core_revision": (
                args.core_revision
                if args.core_revision is not None
                else git_revision(args.ac_root)
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "output": {
            "csv": str(args.output_csv.resolve()),
            "sha256": sha256(args.output_csv),
            "rows": int(len(frame)),
            "trajectory_csv": (
                str(args.trajectory_csv.resolve())
                if args.trajectory_csv is not None
                else None
            ),
            "trajectory_sha256": (
                sha256(args.trajectory_csv)
                if args.trajectory_csv is not None
                else None
            ),
            "trajectory_rows": int(len(trajectory_frame)),
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
