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


CONDITIONS = (
    "learned recurrence",
    "mean-balanced recurrence",
    "reversed recurrence",
)
PLOT_HORIZON = 15


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rescale_within_class_blocks(network, task, target_gains: np.ndarray, helper):
    coding = task.area_map["coding"]
    recurrent = network.weights[(coding, coding)]
    _, membership = helper.class_partition(task, recurrent.shape[0])
    rows = np.repeat(np.arange(recurrent.shape[0]), np.diff(recurrent.indptr))
    columns = recurrent.indices
    original_data = recurrent.data.copy()
    original_gains = helper.class_gains(network, task)

    for digit, (current, target) in enumerate(zip(original_gains, target_gains)):
        if current <= 0.0 or target <= 0.0:
            raise ValueError("recurrent gains must be positive")
        mask = (membership[rows] == digit) & (membership[columns] == digit)
        recurrent.data[mask] *= target / current

    achieved = helper.class_gains(network, task)
    off_diagonal = membership[rows] != membership[columns]
    off_diagonal_change = (
        float(np.max(np.abs(recurrent.data[off_diagonal] - original_data[off_diagonal])))
        if np.any(off_diagonal)
        else 0.0
    )
    return {
        "original_gains": original_gains,
        "target_gains": np.asarray(target_gains, dtype=float),
        "achieved_gains": achieved,
        "maximum_gain_error": float(np.max(np.abs(achieved - target_gains))),
        "maximum_off_diagonal_weight_change": off_diagonal_change,
    }


def reversed_gain_targets(gains: np.ndarray) -> np.ndarray:
    ordered_classes = np.argsort(gains, kind="stable")
    descending_gains = np.sort(gains)[::-1]
    targets = np.empty_like(gains)
    targets[ordered_classes] = descending_gains
    return targets


def paired_differences(seed_frame: pd.DataFrame, helper) -> dict[str, object]:
    learned = seed_frame[seed_frame.condition == "learned recurrence"].set_index("seed")
    result: dict[str, object] = {}
    for condition in ("mean-balanced recurrence", "reversed recurrence"):
        comparison = seed_frame[seed_frame.condition == condition].set_index("seed")
        condition_result: dict[str, object] = {}
        for metric in (
            "accuracy_change",
            "any_revision",
            "final_revision",
            "corrections",
            "corruptions",
            "wrong_to_wrong",
        ):
            difference = comparison[metric] - learned[metric]
            mean, low, high = helper.mean_interval(difference)
            condition_result[f"{metric}_difference"] = {
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        result[f"{condition}_minus_learned"] = condition_result
    return result


def make_figure(time_summary: pd.DataFrame, seed_summary: pd.DataFrame, output: Path, helper) -> None:
    helper.configure_plotting()
    colours = {
        "learned recurrence": "#009E8E",
        "mean-balanced recurrence": "#7A5AA6",
        "reversed recurrence": "#D0644A",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))

    for zorder, condition in enumerate(CONDITIONS, start=2):
        subset = time_summary[
            (time_summary.condition == condition)
            & (time_summary.readout_r <= PLOT_HORIZON)
        ]
        x = subset.readout_r.to_numpy()
        mean = subset.accuracy.to_numpy()
        low = np.clip(subset.accuracy_ci_low.to_numpy(), 0.0, 1.0)
        high = np.clip(subset.accuracy_ci_high.to_numpy(), 0.0, 1.0)
        axes[0].plot(
            x,
            mean,
            color=colours[condition],
            linewidth=1.9,
            marker="o",
            markersize=3.5,
            zorder=zorder,
        )
        axes[0].fill_between(x, low, high, color=colours[condition], alpha=0.13, linewidth=0)

    axes[0].set_title("(a) Accuracy after the first readout", loc="left", fontweight="bold")
    axes[0].set_xlabel("Readout $r$")
    axes[0].set_ylabel("Classification accuracy")
    axes[0].set_xlim(0.8, 18.4)
    axes[0].set_ylim(0.28, 0.42)
    axes[0].set_xticks([1, 3, 5, 7, 10, 15])
    for condition, short_label in (
        ("learned recurrence", "Learned"),
        ("mean-balanced recurrence", "Balanced"),
        ("reversed recurrence", "Reversed"),
    ):
        final_value = time_summary.loc[
            (time_summary.condition == condition)
            & (time_summary.readout_r == PLOT_HORIZON),
            "accuracy",
        ].iloc[0]
        axes[0].text(
            15.45,
            final_value,
            short_label,
            color=colours[condition],
            fontsize=7.7,
            va="center",
        )

    condition_positions = np.arange(len(CONDITIONS), dtype=float)
    outcome_specs = (
        ("corrections", "Corrections", "#009E8E", -0.12, "o"),
        ("corruptions", "Corruptions", "#D0644A", 0.12, "D"),
    )
    rng = np.random.default_rng(7)
    for metric, outcome_label, colour, offset, marker in outcome_specs:
        for index, condition in enumerate(CONDITIONS):
            values = seed_summary.loc[
                seed_summary.condition == condition, metric
            ].to_numpy(dtype=float)
            jitter = rng.uniform(-0.025, 0.025, size=len(values))
            axes[1].scatter(
                index + offset + jitter,
                values,
                s=18,
                marker=marker,
                color=colour,
                alpha=0.58,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            axes[1].scatter(
                index + offset,
                values.mean(),
                s=48,
                marker=marker,
                color=colour,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
                label=outcome_label if index == 0 else None,
            )

    axes[1].set_title("(b) Final outcome of revisions", loc="left", fontweight="bold")
    axes[1].set_ylabel("Fraction of test trajectories")
    axes[1].set_xticks(condition_positions, ["Learned", "Balanced", "Reversed"])
    axes[1].set_xlim(-0.48, 2.48)
    axes[1].set_ylim(-0.002, 0.075)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper center")

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
    parser.add_argument("--equal-support-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=lambda value: tuple(int(v) for v in value.split(",")), default=tuple(range(42, 52)))
    parser.add_argument("--base-revision", default="unrecorded")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))
    import pyac.tasks.mnist as mnist

    training = load_module("class_level_helpers", args.training_helper)
    helper = load_module("equal_support_helpers", args.equal_support_helper)
    train = mnist.load_mnist_split(args.data_dir, "train")
    test = mnist.load_mnist_split(args.data_dir, "test")
    train_images, train_labels, _ = training.balanced_subset(
        train.images, train.labels, helper.TRAIN_PER_CLASS
    )
    test_images, test_labels, test_ids = training.balanced_subset(
        test.images, test.labels, helper.TEST_PER_CLASS
    )

    trajectory_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    first_cap_checks: list[bool] = []
    first_label_checks: list[bool] = []
    gain_errors: list[float] = []
    off_diagonal_changes: list[float] = []

    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        trained_network, trained_task, _ = training.train_model(
            seed, train_images, train_labels, mnist
        )
        variants = {
            condition: (copy.deepcopy(trained_network), copy.deepcopy(trained_task))
            for condition in CONDITIONS
        }
        original_gains = helper.class_gains(trained_network, trained_task)
        targets = {
            "mean-balanced recurrence": np.full(10, original_gains.mean()),
            "reversed recurrence": reversed_gain_targets(original_gains),
        }
        for condition, target in targets.items():
            network, task = variants[condition]
            audit = rescale_within_class_blocks(network, task, target, helper)
            gain_errors.append(float(audit["maximum_gain_error"]))
            off_diagonal_changes.append(float(audit["maximum_off_diagonal_weight_change"]))
            for digit in range(10):
                gain_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "digit": digit,
                        "original_gain": float(original_gains[digit]),
                        "target_gain": float(target[digit]),
                        "achieved_gain": float(audit["achieved_gains"][digit]),
                    }
                )

        condition_trajectories: dict[str, pd.DataFrame] = {}
        for condition, (network, task) in variants.items():
            trajectories, times = helper.evaluate_condition(
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
    seed_summary, time_seed = helper.seed_summaries(trajectories, times)
    time_summary = helper.aggregate_time_series(time_seed)
    condition_stats = helper.condition_statistics(seed_summary, trajectories)

    trajectories.to_csv(args.output / "reversed_support_trajectories.csv", index=False)
    times.to_csv(args.output / "reversed_support_raw.csv", index=False)
    gains.to_csv(args.output / "reversed_support_gain_audit.csv", index=False)
    seed_summary.to_csv(args.output / "reversed_support_per_seed.csv", index=False)
    time_summary.to_csv(args.output / "reversed_support_time_series.csv", index=False)
    make_figure(time_summary, seed_summary, args.output / "recurrent_support_direction", helper)

    result = {
        "protocol": {
            "seeds": list(args.seeds),
            "train_per_class": helper.TRAIN_PER_CLASS,
            "test_per_class": helper.TEST_PER_CLASS,
            "horizon": helper.HORIZON,
            "training_condition": "50 distinct images per class, class organised",
            "input": "held",
            "plasticity_during_evaluation": False,
            "interventions": "only within-class recurrent blocks rescaled; no renormalisation or retraining",
        },
        "intervention_audit": {
            "all_first_caps_identical": all(first_cap_checks),
            "all_first_predictions_identical": all(first_label_checks),
            "maximum_gain_error": max(gain_errors),
            "maximum_off_diagonal_weight_change": max(off_diagonal_changes),
        },
        "conditions": condition_stats,
        "paired_seed_differences": paired_differences(seed_summary, helper),
        "software": {
            "base_revision": args.base_revision,
            "pyac_source_tree_sha256": helper.source_tree_sha256(
                args.ac_root / "pyac" / "src" / "pyac"
            ),
            "experiment_script_sha256": file_sha256(Path(__file__)),
            "training_helper_sha256": file_sha256(args.training_helper),
            "equal_support_helper_sha256": file_sha256(args.equal_support_helper),
        },
        "data": {
            path.name: file_sha256(path)
            for path in sorted(args.data_dir.glob("*.gz"))
        },
    }
    (args.output / "reversed_support_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
