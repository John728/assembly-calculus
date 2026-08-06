from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HORIZON = 20
PLOT_HORIZON = 15
TRAIN_PER_CLASS = 50
TEST_PER_CLASS = 20
CONDITIONS = (
    "learned recurrence",
    "mean-balanced recurrence",
    "no recurrence",
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def first_cap_hash(cap: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(cap, dtype=np.int64).tobytes()).hexdigest()


def class_partition(task, n: int) -> tuple[list[np.ndarray], np.ndarray]:
    assemblies = [
        np.asarray(task.class_assemblies[digit].indices, dtype=np.int64)
        for digit in range(10)
    ]
    membership = np.full(n, -1, dtype=np.int64)
    for digit, indices in enumerate(assemblies):
        if np.any(membership[indices] != -1):
            raise ValueError("class assemblies overlap")
        membership[indices] = digit
    if np.any(membership == -1):
        raise ValueError("class assemblies do not partition the coding area")
    return assemblies, membership


def class_gains(network, task) -> np.ndarray:
    coding = task.area_map["coding"]
    recurrent = network.weights[(coding, coding)]
    return np.asarray(
        [
            float(recurrent[indices][:, indices].sum()) / len(indices)
            for indices in (
                task.class_assemblies[digit].indices for digit in range(10)
            )
        ],
        dtype=float,
    )


def balance_within_class_blocks(network, task) -> dict[str, object]:
    coding = task.area_map["coding"]
    recurrent = network.weights[(coding, coding)]
    _, membership = class_partition(task, recurrent.shape[0])
    rows = np.repeat(np.arange(recurrent.shape[0]), np.diff(recurrent.indptr))
    columns = recurrent.indices
    original_data = recurrent.data.copy()
    original_gains = class_gains(network, task)
    target = float(original_gains.mean())

    for digit, gain in enumerate(original_gains):
        if gain <= 0.0:
            raise ValueError(f"class {digit} has non-positive recurrent gain")
        mask = (membership[rows] == digit) & (membership[columns] == digit)
        recurrent.data[mask] *= target / gain

    balanced_gains = class_gains(network, task)
    off_diagonal = membership[rows] != membership[columns]
    off_diagonal_change = (
        float(np.max(np.abs(recurrent.data[off_diagonal] - original_data[off_diagonal])))
        if np.any(off_diagonal)
        else 0.0
    )
    return {
        "target_gain": target,
        "original_gains": original_gains,
        "balanced_gains": balanced_gains,
        "maximum_gain_error": float(np.max(np.abs(balanced_gains - target))),
        "maximum_off_diagonal_weight_change": off_diagonal_change,
    }


def last_change_readout(predictions: list[int]) -> int:
    return max(
        (
            index + 1
            for index in range(1, len(predictions))
            if predictions[index] != predictions[index - 1]
        ),
        default=1,
    )


def evaluate_example(network, task, mnist, image, target: int, instance_id: int):
    sensory = task.area_map["sensory"]
    coding = task.area_map["coding"]
    sensory_n = network.areas_by_name[sensory].n
    sensory_assembly = task.encoder.encode(image)
    stimulus = np.zeros(sensory_n, dtype=float)
    stimulus[sensory_assembly.indices] = 1.0

    network.activations[sensory] = np.array([], dtype=np.int64)
    network.activations[coding] = np.array([], dtype=np.int64)
    network.step_count = 0

    predictions: list[int] = []
    caps: list[np.ndarray] = []
    for _ in range(HORIZON):
        network.step(
            external_stimuli={sensory: stimulus},
            plasticity_on=False,
            biases={coding: task.coding_bias},
        )
        cap = network.activations[coding].copy()
        caps.append(cap)
        predictions.append(mnist.decode_mnist_class(network.get_assembly(coding), task))

    return {
        "instance_id": instance_id,
        "target": target,
        "predictions": predictions,
        "first_cap_hash": first_cap_hash(caps[0]),
        "settling_readout": last_change_readout(predictions),
        "switch_count": int(
            np.sum(np.asarray(predictions[1:]) != np.asarray(predictions[:-1]))
        ),
    }


def evaluate_condition(seed, condition, network, task, mnist, images, labels, ids):
    trajectories: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    for image, target_value, instance_id_value in zip(images, labels, ids):
        target = int(target_value)
        instance_id = int(instance_id_value)
        result = evaluate_example(network, task, mnist, image, target, instance_id)
        predictions = result["predictions"]
        initial = int(predictions[0])
        final = int(predictions[-1])
        revised = initial != final
        trajectories.append(
            {
                "seed": seed,
                "condition": condition,
                "instance_id": instance_id,
                "target": target,
                "first_cap_hash": result["first_cap_hash"],
                "initial_prediction": initial,
                "final_prediction": final,
                "initial_correct": initial == target,
                "final_correct": final == target,
                "switch_count": result["switch_count"],
                "settling_readout": result["settling_readout"],
                "final_revision": revised,
                "correction": revised and initial != target and final == target,
                "corruption": revised and initial == target and final != target,
                "wrong_to_wrong": revised and initial != target and final != target,
            }
        )
        for readout, prediction in enumerate(predictions, start=1):
            time_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "instance_id": instance_id,
                    "readout_r": readout,
                    "prediction": int(prediction),
                    "correct": int(prediction) == target,
                    "changed_from_first": int(prediction) != initial,
                }
            )
    return trajectories, time_rows


def mean_interval(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) < 2 or np.all(array == array[0]):
        return mean, mean, mean
    half_width = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    return mean, mean - half_width, mean + half_width


def seed_summaries(trajectories: pd.DataFrame, times: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory_seed = (
        trajectories.groupby(["seed", "condition"], as_index=False)
        .agg(
            initial_accuracy=("initial_correct", "mean"),
            final_accuracy=("final_correct", "mean"),
            any_revision=("switch_count", lambda values: np.mean(values > 0)),
            final_revision=("final_revision", "mean"),
            corrections=("correction", "mean"),
            corruptions=("corruption", "mean"),
            wrong_to_wrong=("wrong_to_wrong", "mean"),
            maximum_settling_readout=("settling_readout", "max"),
        )
        .assign(accuracy_change=lambda frame: frame.final_accuracy - frame.initial_accuracy)
    )
    time_seed = (
        times.groupby(["seed", "condition", "readout_r"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            changed_from_first=("changed_from_first", "mean"),
        )
    )
    return trajectory_seed, time_seed


def aggregate_time_series(seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition, readout), group in seed_frame.groupby(["condition", "readout_r"]):
        row: dict[str, object] = {"condition": condition, "readout_r": int(readout)}
        for metric in ("accuracy", "changed_from_first"):
            mean, low, high = mean_interval(group[metric])
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def condition_statistics(seed_frame: pd.DataFrame, trajectories: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, group in seed_frame.groupby("condition"):
        row: dict[str, object] = {"condition": condition}
        for metric in (
            "initial_accuracy",
            "final_accuracy",
            "accuracy_change",
            "any_revision",
            "final_revision",
            "corrections",
            "corruptions",
            "wrong_to_wrong",
        ):
            mean, low, high = mean_interval(group[metric])
            if metric != "accuracy_change":
                low = max(0.0, low)
                high = min(1.0, high)
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        condition_rows = trajectories[trajectories.condition == condition]
        row["trajectories"] = int(len(condition_rows))
        row["revised_trajectories"] = int((condition_rows.switch_count > 0).sum())
        row["final_revisions"] = int(condition_rows.final_revision.sum())
        row["correction_count"] = int(condition_rows.correction.sum())
        row["corruption_count"] = int(condition_rows.corruption.sum())
        row["wrong_to_wrong_count"] = int(condition_rows.wrong_to_wrong.sum())
        row["maximum_settling_readout"] = int(condition_rows.settling_readout.max())
        rows.append(row)
    return rows


def paired_differences(seed_frame: pd.DataFrame) -> dict[str, object]:
    learned = seed_frame[seed_frame.condition == "learned recurrence"].set_index("seed")
    balanced = seed_frame[seed_frame.condition == "mean-balanced recurrence"].set_index("seed")
    result: dict[str, object] = {}
    for metric in (
        "accuracy_change",
        "any_revision",
        "final_revision",
        "corrections",
        "corruptions",
        "wrong_to_wrong",
    ):
        difference = balanced[metric] - learned[metric]
        mean, low, high = mean_interval(difference)
        result[f"balanced_minus_learned_{metric}"] = {
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
        }
    return result


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#202020",
            "axes.linewidth": 0.8,
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "savefig.dpi": 300,
        }
    )


def make_figure(time_summary: pd.DataFrame, seed_summary: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    colours = {
        "learned recurrence": "#009E8E",
        "mean-balanced recurrence": "#7A5AA6",
        "no recurrence": "#4D4D4D",
    }
    labels = {
        "learned recurrence": "Learned recurrence",
        "mean-balanced recurrence": "Mean-balanced recurrence",
        "no recurrence": "No recurrence",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))

    for zorder, condition in enumerate(CONDITIONS, start=2):
        subset = time_summary[
            (time_summary.condition == condition)
            & (time_summary.readout_r <= PLOT_HORIZON)
        ]
        x = subset.readout_r.to_numpy()
        mean = subset.changed_from_first.to_numpy()
        low = np.clip(subset.changed_from_first_ci_low.to_numpy(), 0.0, 1.0)
        high = np.clip(subset.changed_from_first_ci_high.to_numpy(), 0.0, 1.0)
        axes[0].plot(
            x,
            mean,
            color=colours[condition],
            linewidth=1.9,
            marker="o",
            markersize=3.5,
            label=labels[condition],
            zorder=zorder,
        )
        axes[0].fill_between(x, low, high, color=colours[condition], alpha=0.13, linewidth=0)

    axes[0].set_title("(a) Re-ranking after the first readout", loc="left", fontweight="bold")
    axes[0].set_ylabel(r"$\Pr[\widehat y(r)\ne\widehat y(1)]$")
    axes[0].set_xlabel("Readout $r$")
    axes[0].set_xlim(0.8, PLOT_HORIZON + 0.2)
    axes[0].set_ylim(-0.004, 0.105)
    axes[0].set_xticks([1, 3, 5, 7, 10, 15])
    axes[0].legend(frameon=False, fontsize=7.6, loc="upper left")

    outcomes = ("corrections", "corruptions", "wrong_to_wrong")
    outcome_labels = ("Corrections", "Corruptions", "Wrong to wrong")
    learned = seed_summary[seed_summary.condition == "learned recurrence"].set_index("seed")
    balanced = seed_summary[seed_summary.condition == "mean-balanced recurrence"].set_index("seed")
    offsets = {"learned recurrence": -0.12, "mean-balanced recurrence": 0.12}
    rng = np.random.default_rng(2)
    for index, outcome in enumerate(outcomes):
        for seed in learned.index:
            axes[1].plot(
                [index + offsets["learned recurrence"], index + offsets["mean-balanced recurrence"]],
                [learned.loc[seed, outcome], balanced.loc[seed, outcome]],
                color="#C8C8C8",
                linewidth=0.7,
                zorder=1,
            )
        for condition, frame, marker in (
            ("learned recurrence", learned, "o"),
            ("mean-balanced recurrence", balanced, "D"),
        ):
            jitter = rng.uniform(-0.025, 0.025, size=len(frame))
            x = index + offsets[condition] + jitter
            axes[1].scatter(
                x,
                frame[outcome],
                s=18,
                marker=marker,
                color=colours[condition],
                alpha=0.62,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            axes[1].scatter(
                index + offsets[condition],
                frame[outcome].mean(),
                s=48,
                marker=marker,
                color=colours[condition],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
                label=labels[condition] if index == 0 else None,
            )

    axes[1].set_title("(b) Final outcome of re-ranking", loc="left", fontweight="bold")
    axes[1].set_ylabel("Fraction of test trajectories")
    axes[1].set_xticks(range(len(outcomes)), outcome_labels)
    axes[1].set_xlim(-0.48, 2.48)
    axes[1].set_ylim(-0.002, 0.07)

    for axis in axes:
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.88, wspace=0.32)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=tuple(range(42, 52)))
    parser.add_argument("--base-revision", default="unrecorded")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))
    import pyac.tasks.mnist as mnist

    helpers = load_module("class_level_helpers", args.training_helper)
    train = mnist.load_mnist_split(args.data_dir, "train")
    test = mnist.load_mnist_split(args.data_dir, "test")
    train_images, train_labels, _ = helpers.balanced_subset(
        train.images, train.labels, TRAIN_PER_CLASS
    )
    test_images, test_labels, test_ids = helpers.balanced_subset(
        test.images, test.labels, TEST_PER_CLASS
    )

    trajectory_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    first_cap_checks: list[bool] = []
    first_label_checks: list[bool] = []
    off_diagonal_checks: list[float] = []
    gain_error_checks: list[float] = []

    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        trained_network, trained_task, _ = helpers.train_model(
            seed, train_images, train_labels, mnist
        )
        variants = {
            condition: (copy.deepcopy(trained_network), copy.deepcopy(trained_task))
            for condition in CONDITIONS
        }
        balanced_network, balanced_task = variants["mean-balanced recurrence"]
        audit = balance_within_class_blocks(balanced_network, balanced_task)
        no_network, no_task = variants["no recurrence"]
        coding = no_task.area_map["coding"]
        no_network.weights[(coding, coding)].data.fill(0.0)
        no_network.weights[(coding, coding)].eliminate_zeros()

        for digit, (original, balanced) in enumerate(
            zip(audit["original_gains"], audit["balanced_gains"])
        ):
            gain_rows.append(
                {
                    "seed": seed,
                    "digit": digit,
                    "original_gain": float(original),
                    "target_mean_gain": float(audit["target_gain"]),
                    "balanced_gain": float(balanced),
                }
            )
        off_diagonal_checks.append(float(audit["maximum_off_diagonal_weight_change"]))
        gain_error_checks.append(float(audit["maximum_gain_error"]))

        condition_trajectories: dict[str, pd.DataFrame] = {}
        for condition, (network, task) in variants.items():
            trajectories, times = evaluate_condition(
                seed, condition, network, task, mnist,
                test_images, test_labels, test_ids,
            )
            trajectory_rows.extend(trajectories)
            time_rows.extend(times)
            condition_trajectories[condition] = pd.DataFrame(trajectories).sort_values("instance_id")

        learned = condition_trajectories["learned recurrence"]
        for condition in CONDITIONS[1:]:
            comparison = condition_trajectories[condition]
            first_cap_checks.append(
                bool(np.array_equal(learned.first_cap_hash.to_numpy(), comparison.first_cap_hash.to_numpy()))
            )
            first_label_checks.append(
                bool(np.array_equal(learned.initial_prediction.to_numpy(), comparison.initial_prediction.to_numpy()))
            )

    trajectories = pd.DataFrame(trajectory_rows)
    times = pd.DataFrame(time_rows)
    gains = pd.DataFrame(gain_rows)
    trajectory_seed, time_seed = seed_summaries(trajectories, times)
    time_summary = aggregate_time_series(time_seed)
    condition_stats = condition_statistics(trajectory_seed, trajectories)

    trajectories.to_csv(args.output / "equal_support_trajectories.csv", index=False)
    times.to_csv(args.output / "equal_support_raw.csv", index=False)
    gains.to_csv(args.output / "equal_support_gain_audit.csv", index=False)
    trajectory_seed.to_csv(args.output / "equal_support_per_seed.csv", index=False)
    time_summary.to_csv(args.output / "equal_support_time_series.csv", index=False)
    make_figure(time_summary, trajectory_seed, args.output / "equal_recurrent_support")

    result = {
        "protocol": {
            "seeds": list(args.seeds),
            "train_per_class": TRAIN_PER_CLASS,
            "test_per_class": TEST_PER_CLASS,
            "horizon": HORIZON,
            "training_condition": "50 distinct images per class, class organised",
            "input": "held",
            "plasticity_during_evaluation": False,
            "balanced_intervention": "only within-class recurrent blocks scaled to the seed-level mean class gain; no renormalisation",
        },
        "intervention_audit": {
            "all_first_caps_identical": all(first_cap_checks),
            "all_first_predictions_identical": all(first_label_checks),
            "maximum_balanced_gain_error": max(gain_error_checks),
            "maximum_off_diagonal_weight_change": max(off_diagonal_checks),
        },
        "conditions": condition_stats,
        "paired_seed_differences": paired_differences(trajectory_seed),
        "software": {
            "base_revision": args.base_revision,
            "pyac_source_tree_sha256": source_tree_sha256(args.ac_root / "pyac" / "src" / "pyac"),
            "experiment_script_sha256": file_sha256(Path(__file__)),
            "training_helper_sha256": file_sha256(args.training_helper),
        },
        "data": {
            path.name: file_sha256(path)
            for path in sorted(args.data_dir.glob("*.gz"))
        },
    }
    (args.output / "equal_support_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
