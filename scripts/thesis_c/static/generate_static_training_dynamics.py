from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEEDS = tuple(range(42, 52))
TRAIN_PER_CLASS = 50
TEST_PER_CLASS = 20
HORIZON = 100
N = 2000
K = 200
P = 0.1
BETA = 0.5
RAW_INPUT_K = 200

TRAINING_CONDITIONS = (
    "distinct_1",
    "distinct_10",
    "distinct_50",
    "repeat_1x50",
    "blocked_10x5",
    "interleaved_10x5",
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relevant_core_dirty(root: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "pyac/src/pyac"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return bool(status.strip())


def full_normalise(network, area_name: str | None = None) -> None:
    targets = [area_name] if area_name is not None else network.area_names
    for target in targets:
        keys = [key for key in network.weights if key[1] == target]
        if not keys:
            continue
        total = sum(
            np.asarray(network.weights[key].sum(axis=0)).ravel() for key in keys
        )
        total[total == 0.0] = 1.0
        for key in keys:
            matrix = network.weights[key]
            matrix.data = matrix.data / total[matrix.indices]


def balanced_indices(labels: np.ndarray, per_class: int) -> np.ndarray:
    selected: list[np.ndarray] = []
    for digit in range(10):
        matches = np.flatnonzero(np.asarray(labels) == digit)
        if len(matches) < per_class:
            raise ValueError(
                f"digit {digit} has {len(matches)} examples, fewer than {per_class}"
            )
        selected.append(matches[:per_class])
    return np.concatenate(selected)


def balanced_subset(
    images: np.ndarray,
    labels: np.ndarray,
    per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = balanced_indices(labels, per_class)
    return images[indices], np.asarray(labels)[indices], indices


def build_training_schedule(
    pool_images: np.ndarray,
    pool_labels: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    if condition not in TRAINING_CONDITIONS:
        raise ValueError(f"unknown training condition: {condition}")

    scheduled_images: list[np.ndarray] = []
    scheduled_labels: list[np.ndarray] = []
    for digit in range(10):
        digit_images = pool_images[np.asarray(pool_labels) == digit]
        if len(digit_images) != TRAIN_PER_CLASS:
            raise ValueError("the balanced training pool must contain 50 images per digit")

        if condition == "distinct_1":
            schedule = digit_images[:1]
        elif condition == "distinct_10":
            schedule = digit_images[:10]
        elif condition == "distinct_50":
            schedule = digit_images
        elif condition == "repeat_1x50":
            schedule = np.repeat(digit_images[:1], TRAIN_PER_CLASS, axis=0)
        elif condition == "blocked_10x5":
            schedule = np.repeat(digit_images[:10], 5, axis=0)
        else:
            schedule = np.concatenate([digit_images[:10]] * 5, axis=0)

        scheduled_images.append(schedule)
        scheduled_labels.append(
            np.full(len(schedule), digit, dtype=np.int64)
        )

    return np.concatenate(scheduled_images), np.concatenate(scheduled_labels)


def condition_metadata(condition: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "condition": condition,
        "exposure_level": np.nan,
        "schedule": "",
        "intervention": "",
    }
    if condition.startswith("distinct_"):
        exposure = int(condition.rsplit("_", 1)[1])
        metadata["exposure_level"] = exposure
        if exposure == 50:
            metadata["schedule"] = "50 distinct"
            metadata["intervention"] = "baseline"
    elif condition == "repeat_1x50":
        metadata["schedule"] = "1 repeated"
    elif condition == "blocked_10x5":
        metadata["schedule"] = "10 blocked"
    elif condition == "interleaved_10x5":
        metadata["schedule"] = "10 interleaved"
    elif condition == "gain_balanced":
        metadata["intervention"] = "gain balanced"
    elif condition == "gain_reversed":
        metadata["intervention"] = "gain reversed"
    return metadata


def last_change_readout(predictions: list[int]) -> int:
    return max(
        (
            index + 1
            for index in range(1, len(predictions))
            if predictions[index] != predictions[index - 1]
        ),
        default=1,
    )


def train_model(
    *,
    seed: int,
    images: np.ndarray,
    labels: np.ndarray,
    pyac,
) -> tuple[object, object, dict[int, float]]:
    rng = np.random.default_rng(seed)
    encoder = pyac.RawPixelEncoder(k=RAW_INPUT_K, area_name="X")
    network, task = pyac.build_mnist_network(
        n=N,
        k=K,
        p=P,
        beta=BETA,
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

    calls_per_digit = {
        digit: int(np.sum(np.asarray(labels) == digit)) for digit in range(10)
    }
    if len(set(calls_per_digit.values())) != 1:
        raise ValueError("every training condition must be balanced across digits")
    presentation_rounds = next(iter(calls_per_digit.values()))
    ordered_digits = [
        digit for digit in range(10) for _ in range(calls_per_digit[digit])
    ]
    transitions: dict[int, list[float]] = {digit: [] for digit in range(10)}
    original_step = network.step
    call_index = 0

    def recorded_step(*args, **kwargs):
        nonlocal call_index
        digit = ordered_digits[call_index]
        previous = network.activations[coding_area].copy()
        result = original_step(*args, **kwargs)
        current = network.activations[coding_area]
        if previous.size:
            overlap = (
                np.intersect1d(previous, current, assume_unique=True).size / K
            )
            transitions[digit].append(float(overlap))
        call_index += 1
        return result

    network.step = recorded_step
    pyac.train_mnist_assemblies(
        network,
        task,
        images,
        labels,
        presentation_rounds=presentation_rounds,
        settle_steps=1,
        class_organized=True,
        normalization_on=True,
    )
    network.step = original_step

    persistence = {
        digit: (
            float(np.mean(transitions[digit]))
            if transitions[digit]
            else np.nan
        )
        for digit in range(10)
    }
    return network, task, persistence


def recurrent_selectivity(network, task) -> float:
    coding = task.area_map["coding"]
    recurrent = network.weights[(coding, coding)]
    assemblies = [task.class_assemblies[digit].indices for digit in range(10)]
    within = sum(
        float(recurrent[indices][:, indices].sum()) for indices in assemblies
    )
    total = float(recurrent.sum())
    return within / total


def class_budget_metrics(
    *,
    network,
    task,
    training_images: np.ndarray,
    training_labels: np.ndarray,
    persistence: dict[int, float],
    seed: int,
    condition: str,
) -> list[dict[str, object]]:
    sensory = task.area_map["sensory"]
    coding = task.area_map["coding"]
    sensory_weights = network.weights[(sensory, coding)]
    recurrent_weights = network.weights[(coding, coding)]
    selectivity = recurrent_selectivity(network, task)
    condition_fields = condition_metadata(condition)
    rows: list[dict[str, object]] = []

    for digit in range(10):
        indices = task.class_assemblies[digit].indices
        recurrent_total = (
            float(recurrent_weights[:, indices].sum()) / len(indices)
        )
        sensory_total = float(sensory_weights[:, indices].sum()) / len(indices)
        within_gain = (
            float(recurrent_weights[indices][:, indices].sum()) / len(indices)
        )

        digit_images = training_images[np.asarray(training_labels) == digit]
        sensory_counts = np.zeros(sensory_weights.shape[0], dtype=np.int64)
        for image in digit_images:
            sensory_counts[task.encoder.encode(image).indices] += 1
        presentations = len(digit_images)
        if presentations > 1:
            pair_overlap = float(
                np.sum(sensory_counts * (sensory_counts - 1))
                / (presentations * (presentations - 1) * RAW_INPUT_K)
            )
        else:
            pair_overlap = 1.0

        recurrent_pressure = (K - 1) * (1.0 + BETA) ** max(
            presentations - 1, 0
        )
        recurrent_background = N - K
        sensory_pressure = float(np.sum((1.0 + BETA) ** sensory_counts))
        predicted_gain = recurrent_pressure / (
            recurrent_pressure + recurrent_background + sensory_pressure
        )

        rows.append(
            {
                "seed": seed,
                **condition_fields,
                "digit": digit,
                "presentations": presentations,
                "sensory_pair_overlap": pair_overlap,
                "sensory_core_count": int(
                    np.sum(sensory_counts == presentations)
                ),
                "training_cap_persistence": persistence[digit],
                "sensory_budget": sensory_total,
                "recurrent_budget": recurrent_total,
                "within_class_gain": within_gain,
                "cross_class_recurrent_budget": recurrent_total - within_gain,
                "predicted_within_gain": predicted_gain,
                "recurrent_selectivity": selectivity,
                "coding_bias_mean": float(
                    np.mean(task.coding_bias[indices])
                ),
            }
        )
    return rows


def class_within_gains(network, task) -> dict[int, float]:
    coding = task.area_map["coding"]
    recurrent = network.weights[(coding, coding)]
    return {
        digit: float(
            recurrent[task.class_assemblies[digit].indices][
                :, task.class_assemblies[digit].indices
            ].sum()
        ) / K
        for digit in range(10)
    }


def assign_recurrent_gains(
    network,
    task,
    target_by_class: dict[int, float],
) -> None:
    coding = task.area_map["coding"]
    recurrent_weights = network.weights[(coding, coding)]
    recurrent_scales = np.ones(recurrent_weights.shape[1], dtype=float)
    current_gains = class_within_gains(network, task)

    for digit, target in target_by_class.items():
        if target <= 0.0:
            raise ValueError("target recurrent gains must be positive")
        indices = task.class_assemblies[digit].indices
        if current_gains[digit] == 0.0:
            raise ValueError("cannot rescale a class with zero recurrent gain")
        recurrent_scales[indices] = target / current_gains[digit]

    recurrent_weights.data *= recurrent_scales[recurrent_weights.indices]


def gain_targets(network, task) -> tuple[dict[int, float], dict[int, float]]:
    gains = class_within_gains(network, task)
    mean_gain = float(np.mean(list(gains.values())))
    balanced = {digit: mean_gain for digit in range(10)}

    ordered_classes = sorted(gains, key=gains.get)
    ordered_gains = sorted(gains.values(), reverse=True)
    reversed_targets = {
        digit: target
        for digit, target in zip(ordered_classes, ordered_gains)
    }
    return balanced, reversed_targets


def classify_transition(
    *,
    target: int,
    initial: int,
    final: int,
) -> str:
    if initial == target and final == target:
        return "stable correct"
    if initial != target and final == target:
        return "corrected"
    if initial == target and final != target:
        return "corrupted"
    if initial == final:
        return "stable wrong"
    return "wrong to different wrong"


def evaluate_condition(
    *,
    network,
    task,
    pyac,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    test_ids: np.ndarray,
    seed: int,
    condition: str,
    horizon: int,
    gain_by_class: dict[int, float],
    collect_traces: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    correct = np.zeros(horizon, dtype=float)
    unsettled = np.zeros(horizon, dtype=float)
    changed_from_initial = np.zeros(horizon, dtype=float)
    switched_at = np.zeros(horizon, dtype=float)
    trajectory_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    condition_fields = condition_metadata(condition)

    for image, target_value, instance_id in zip(
        test_images, test_labels, test_ids
    ):
        target = int(target_value)
        result = pyac.evaluate_mnist_example(
            network,
            task,
            image,
            target,
            instance_id=int(instance_id),
            t=horizon - 1,
            stimulus_mode="held",
        )
        predictions = [int(value) for value in result["trajectory"]]
        overlaps = np.asarray(result["overlap_trajectory"], dtype=float)
        tau = last_change_readout(predictions)
        initial = predictions[0]
        final = predictions[-1]
        switch_count = int(
            np.sum(np.asarray(predictions[1:]) != np.asarray(predictions[:-1]))
        )

        for readout, prediction in enumerate(predictions, start=1):
            index = readout - 1
            correct[index] += prediction == target
            unsettled[index] += tau > readout
            changed_from_initial[index] += prediction != initial
            if index > 0:
                switched_at[index] += predictions[index] != predictions[index - 1]

        initial_gain = gain_by_class[initial]
        final_gain = gain_by_class[final]
        trajectory_rows.append(
            {
                "seed": seed,
                **condition_fields,
                "instance_id": int(instance_id),
                "target": target,
                "initial_prediction": initial,
                "final_prediction": final,
                "initial_correct": initial == target,
                "final_correct": final == target,
                "transition_type": classify_transition(
                    target=target, initial=initial, final=final
                ),
                "switch_count": switch_count,
                "settling_readout": tau,
                "right_censored": tau == horizon and switch_count > 0,
                "initial_gain": initial_gain,
                "final_gain": final_gain,
                "gain_change": final_gain - initial_gain,
                "initial_overlap": float(overlaps[0, initial]),
                "initial_strongest_rival_overlap": float(
                    np.max(np.delete(overlaps[0], initial))
                ),
                "final_overlap": float(overlaps[-1, final]),
            }
        )
        if collect_traces:
            trace_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "instance_id": int(instance_id),
                    "target": target,
                    "initial_prediction": initial,
                    "final_prediction": final,
                    "transition_type": classify_transition(
                        target=target, initial=initial, final=final
                    ),
                    "predictions": predictions,
                    "overlaps": overlaps.tolist(),
                }
            )

    denominator = len(test_images)
    time_rows = [
        {
            "seed": seed,
            **condition_fields,
            "readout_r": readout,
            "accuracy": correct[readout - 1] / denominator,
            "unsettled": unsettled[readout - 1] / denominator,
            "changed_from_initial": changed_from_initial[readout - 1]
            / denominator,
            "switch_event_rate": switched_at[readout - 1] / denominator,
        }
        for readout in range(1, horizon + 1)
    ]
    return time_rows, trajectory_rows, trace_rows


def select_representative_traces(
    traces: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    priorities = (
        "stable correct",
        "corrected",
        "corrupted",
        "stable wrong",
    )
    used_targets: set[int] = set()
    for transition_type in priorities:
        candidates = sorted(
            (
                row
                for row in traces
                if row["transition_type"] == transition_type
            ),
            key=lambda row: int(row["instance_id"]),
        )
        distinct = [
            row for row in candidates if int(row["target"]) not in used_targets
        ]
        choice = distinct[0] if distinct else (candidates[0] if candidates else None)
        if choice is not None:
            selected.append(choice)
            used_targets.add(int(choice["target"]))
    if len(selected) < 4:
        remaining = sorted(
            (
                row
                for row in traces
                if int(row["instance_id"])
                not in {int(item["instance_id"]) for item in selected}
            ),
            key=lambda row: int(row["instance_id"]),
        )
        selected.extend(remaining[: 4 - len(selected)])
    return selected[:4]


def seed_summaries(trajectories: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (seed, condition), group in trajectories.groupby(["seed", "condition"]):
        transition_counts = group["transition_type"].value_counts()
        rows.append(
            {
                "seed": int(seed),
                "condition": str(condition),
                "initial_accuracy": float(group["initial_correct"].mean()),
                "final_accuracy": float(group["final_correct"].mean()),
                "accuracy_change": float(
                    group["final_correct"].mean()
                    - group["initial_correct"].mean()
                ),
                "switch_rate": float((group["switch_count"] > 0).mean()),
                "maximum_settling_readout": int(
                    group["settling_readout"].max()
                ),
                "corrected": int(transition_counts.get("corrected", 0)),
                "corrupted": int(transition_counts.get("corrupted", 0)),
                "wrong_to_different_wrong": int(
                    transition_counts.get("wrong to different wrong", 0)
                ),
                "switches_to_higher_gain": int(
                    ((group["switch_count"] > 0) & (group["gain_change"] > 0)).sum()
                ),
                "switches_to_lower_gain": int(
                    ((group["switch_count"] > 0) & (group["gain_change"] < 0)).sum()
                ),
            }
        )
    return rows


def load_pyac(ac_root: Path):
    sys.path.insert(0, str(ac_root / "pyac" / "src"))
    import pyac.tasks.mnist as mnist

    return mnist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated network seeds",
    )
    parser.add_argument("--test-per-class", type=int, default=TEST_PER_CLASS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    if args.test_per_class <= 0 or args.horizon <= 0:
        raise ValueError("test-per-class and horizon must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    pyac = load_pyac(args.ac_root)
    train = pyac.load_mnist_split(args.data_dir, "train")
    test = pyac.load_mnist_split(args.data_dir, "test")
    train_images, train_labels, train_ids = balanced_subset(
        train.images, train.labels, TRAIN_PER_CLASS
    )
    test_images, test_labels, test_ids = balanced_subset(
        test.images, test.labels, args.test_per_class
    )

    time_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    representative_candidates: list[dict[str, object]] = []

    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        for training_condition in TRAINING_CONDITIONS:
            print(f"  {training_condition}", flush=True)
            scheduled_images, scheduled_labels = build_training_schedule(
                train_images, train_labels, training_condition
            )
            network, task, persistence = train_model(
                seed=seed,
                images=scheduled_images,
                labels=scheduled_labels,
                pyac=pyac,
            )

            if training_condition == "distinct_50":
                balanced_targets, reversed_targets = gain_targets(network, task)
                variants = {
                    "distinct_50": (copy.deepcopy(network), copy.deepcopy(task)),
                    "gain_balanced": (copy.deepcopy(network), copy.deepcopy(task)),
                    "gain_reversed": (copy.deepcopy(network), copy.deepcopy(task)),
                }
                assign_recurrent_gains(
                    variants["gain_balanced"][0],
                    variants["gain_balanced"][1],
                    balanced_targets,
                )
                assign_recurrent_gains(
                    variants["gain_reversed"][0],
                    variants["gain_reversed"][1],
                    reversed_targets,
                )
            else:
                variants = {training_condition: (network, task)}

            for condition, (variant_network, variant_task) in variants.items():
                metrics = class_budget_metrics(
                    network=variant_network,
                    task=variant_task,
                    training_images=scheduled_images,
                    training_labels=scheduled_labels,
                    persistence=persistence,
                    seed=seed,
                    condition=condition,
                )
                class_rows.extend(metrics)
                gain_by_class = {
                    int(row["digit"]): float(row["within_class_gain"])
                    for row in metrics
                }
                evaluated_time, evaluated_trajectories, traces = evaluate_condition(
                    network=variant_network,
                    task=variant_task,
                    pyac=pyac,
                    test_images=test_images,
                    test_labels=test_labels,
                    test_ids=test_ids,
                    seed=seed,
                    condition=condition,
                    horizon=args.horizon,
                    gain_by_class=gain_by_class,
                    collect_traces=seed == args.seeds[0]
                    and condition == "distinct_50",
                )
                time_rows.extend(evaluated_time)
                trajectory_rows.extend(evaluated_trajectories)
                representative_candidates.extend(traces)

    time_frame = pd.DataFrame(time_rows)
    trajectory_frame = pd.DataFrame(trajectory_rows)
    class_frame = pd.DataFrame(class_rows)
    time_frame.to_csv(args.output / "mnist_static_time_series.csv", index=False)
    trajectory_frame.to_csv(
        args.output / "mnist_static_trajectory_summary.csv", index=False
    )
    class_frame.to_csv(
        args.output / "mnist_static_class_metrics.csv", index=False
    )

    selected_traces = select_representative_traces(representative_candidates)
    (args.output / "mnist_static_representative_traces.json").write_text(
        json.dumps(selected_traces, indent=2) + "\n", encoding="utf-8"
    )

    script_path = Path(__file__).resolve()
    summary = {
        "protocol": {
            "seeds": list(args.seeds),
            "train_images_per_class": TRAIN_PER_CLASS,
            "test_images_per_class": args.test_per_class,
            "horizon": args.horizon,
            "training_conditions": list(TRAINING_CONDITIONS),
            "gain_interventions": ["gain_balanced", "gain_reversed"],
            "n": N,
            "k": K,
            "p": P,
            "beta": BETA,
            "raw_input_k": RAW_INPUT_K,
            "settle_steps_per_training_image": 1,
            "class_organised": True,
            "coding_state_carried_within_class": True,
            "stimulus_during_evaluation": "held",
            "plasticity_during_evaluation": False,
            "normalisation": "one incoming budget across sensory and recurrent fibres",
            "recurrent_self_synapses": False,
            "readout_indexing": "r=1 is the state after one network update",
            "balanced_train_source_indices": train_ids.tolist(),
            "balanced_test_source_indices": test_ids.tolist(),
        },
        "per_seed": seed_summaries(trajectory_frame),
        "software": {
            "base_revision": git_revision(args.ac_root),
            "pyac_core_dirty": relevant_core_dirty(args.ac_root),
            "experiment_script_sha256": file_sha256(script_path),
            "executed_script": str(script_path),
        },
        "data": {
            path.name: file_sha256(path)
            for path in sorted(args.data_dir.glob("*.gz"))
        },
    }
    (args.output / "mnist_static_training_dynamics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
