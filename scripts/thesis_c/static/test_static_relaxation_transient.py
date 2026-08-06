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


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_training_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("class_level_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        active = network.activations[coding].copy()
        caps.append(active)
        predictions.append(
            mnist.decode_mnist_class(network.get_assembly(coding), task)
        )

    tau = last_change_readout(predictions)
    first_cap = caps[0]
    cap_equal_first = [np.array_equal(cap, first_cap) for cap in caps]
    consecutive_overlap = [
        1.0
        if index == 0
        else np.intersect1d(caps[index - 1], caps[index], assume_unique=True).size
        / task.k
        for index in range(HORIZON)
    ]
    return {
        "instance_id": instance_id,
        "target": target,
        "predictions": predictions,
        "settling_readout": tau,
        "switch_count": int(
            np.sum(np.asarray(predictions[1:]) != np.asarray(predictions[:-1]))
        ),
        "cap_equal_first": cap_equal_first,
        "consecutive_cap_overlap": consecutive_overlap,
    }


def evaluate_condition(seed, condition, network, task, mnist, images, labels, ids):
    trajectories = []
    time_rows = []
    for image, target_value, instance_id in zip(images, labels, ids):
        target = int(target_value)
        result = evaluate_example(
            network,
            task,
            mnist,
            image,
            target,
            int(instance_id),
        )
        predictions = result["predictions"]
        tau = int(result["settling_readout"])
        cap_equal_first = result["cap_equal_first"]
        cap_overlaps = result["consecutive_cap_overlap"]
        trajectories.append(
            {
                "seed": seed,
                "condition": condition,
                "instance_id": int(instance_id),
                "target": target,
                "initial_prediction": predictions[0],
                "final_prediction": predictions[-1],
                "initial_correct": predictions[0] == target,
                "final_correct": predictions[-1] == target,
                "switch_count": int(result["switch_count"]),
                "settling_readout": tau,
                "cap_changed": not all(cap_equal_first),
                "minimum_consecutive_cap_overlap": float(min(cap_overlaps)),
            }
        )
        for readout, (prediction, cap_same, cap_overlap) in enumerate(
            zip(predictions, cap_equal_first, cap_overlaps), start=1
        ):
            time_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "instance_id": int(instance_id),
                    "readout_r": readout,
                    "correct": prediction == target,
                    "prediction": prediction,
                    "unsettled": tau > readout,
                    "changed_from_first": prediction != predictions[0],
                    "cap_equal_first": cap_same,
                    "consecutive_cap_overlap": cap_overlap,
                }
            )
    return trajectories, time_rows


def seed_time_series(time_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        time_frame.groupby(["seed", "condition", "readout_r"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            unsettled=("unsettled", "mean"),
            changed_from_first=("changed_from_first", "mean"),
            cap_equal_first=("cap_equal_first", "mean"),
            consecutive_cap_overlap=("consecutive_cap_overlap", "mean"),
        )
    )


def mean_interval(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, mean, mean
    half_width = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    return mean, mean - half_width, mean + half_width


def aggregate_time_series(seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "accuracy",
        "unsettled",
        "changed_from_first",
        "cap_equal_first",
        "consecutive_cap_overlap",
    ]
    for (condition, readout), group in seed_frame.groupby(
        ["condition", "readout_r"]
    ):
        row: dict[str, object] = {
            "condition": condition,
            "readout_r": int(readout),
        }
        for metric in metrics:
            mean, low, high = mean_interval(group[metric])
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def condition_summary(trajectory_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for condition, group in trajectory_frame.groupby("condition"):
        per_seed = group.groupby("seed").agg(
            initial_accuracy=("initial_correct", "mean"),
            final_accuracy=("final_correct", "mean"),
            revision_rate=("switch_count", lambda values: np.mean(values > 0)),
            cap_revision_rate=("cap_changed", "mean"),
            maximum_settling_readout=("settling_readout", "max"),
        )
        final_difference_by_seed = group.groupby("seed").apply(
            lambda seed_group: np.mean(
                seed_group["initial_prediction"] != seed_group["final_prediction"]
            ),
            include_groups=False,
        )
        row: dict[str, object] = {"condition": condition}
        for metric in (
            "initial_accuracy",
            "final_accuracy",
            "revision_rate",
            "cap_revision_rate",
        ):
            mean, low, high = mean_interval(per_seed[metric])
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        mean, low, high = mean_interval(final_difference_by_seed)
        row["final_difference_rate"] = mean
        row["final_difference_rate_ci_low"] = low
        row["final_difference_rate_ci_high"] = high
        row["maximum_settling_readout"] = int(group["settling_readout"].max())
        row["trajectories"] = int(len(group))
        row["revised_trajectories"] = int((group["switch_count"] > 0).sum())
        row["cap_changed_trajectories"] = int(group["cap_changed"].sum())
        row["minimum_consecutive_cap_overlap"] = float(
            group["minimum_consecutive_cap_overlap"].min()
        )
        rows.append(row)
    return rows


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


def make_figure(
    summary: pd.DataFrame,
    output: Path,
    maximum_settling_readout: int,
) -> None:
    configure_plotting()
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.15))
    styles = {
        "learned recurrence": ("#009E8E", "Learned recurrence"),
        "no recurrence": ("#4D4D4D", "No recurrence"),
    }
    for condition, (colour, label) in styles.items():
        subset = summary[
            (summary["condition"] == condition)
            & (summary["readout_r"] <= PLOT_HORIZON)
        ]
        x = subset["readout_r"].to_numpy()
        for axis, metric in zip(axes, ("accuracy", "unsettled")):
            mean = subset[metric].to_numpy()
            low = subset[f"{metric}_ci_low"].to_numpy()
            high = subset[f"{metric}_ci_high"].to_numpy()
            if metric == "unsettled":
                low = np.clip(low, 0.0, 1.0)
                high = np.clip(high, 0.0, 1.0)
            axis.plot(
                x,
                mean,
                color=colour,
                linewidth=1.9,
                marker="o",
                markersize=3.6,
                label=label,
                zorder=3 if condition == "learned recurrence" else 2,
            )
            axis.fill_between(
                x, low, high, color=colour, alpha=0.14, linewidth=0, zorder=1
            )

    axes[0].set_title("(a) Test accuracy", loc="left", fontweight="bold")
    axes[0].set_ylabel("Classification accuracy")
    axes[0].set_ylim(0.28, 0.42)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    axes[1].set_title("(b) Trajectories yet to settle", loc="left", fontweight="bold")
    axes[1].set_ylabel(r"$\Pr[\tau_{20}>r]$")
    axes[1].set_ylim(-0.005, 0.105)
    axes[1].text(
        14.7,
        0.095,
        f"No decoded changes\nafter readout {maximum_settling_readout}",
        ha="right",
        va="top",
        fontsize=8,
        color="#505050",
    )

    for axis in axes:
        axis.set_xlim(0.8, PLOT_HORIZON + 0.2)
        axis.set_xticks([1, 3, 5, 7, 10, 15])
        axis.set_xlabel("Readout $r$")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.31)
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

    helpers = load_training_helpers(args.training_helper)
    train = mnist.load_mnist_split(args.data_dir, "train")
    test = mnist.load_mnist_split(args.data_dir, "test")
    train_images, train_labels, _ = helpers.balanced_subset(
        train.images, train.labels, TRAIN_PER_CLASS
    )
    test_images, test_labels, test_ids = helpers.balanced_subset(
        test.images, test.labels, TEST_PER_CLASS
    )

    trajectory_rows = []
    time_rows = []
    first_prediction_matches = []
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        trained_network, trained_task, _ = helpers.train_model(
            seed, train_images, train_labels, mnist
        )
        variants = {
            "learned recurrence": (
                copy.deepcopy(trained_network),
                copy.deepcopy(trained_task),
            ),
            "no recurrence": (
                copy.deepcopy(trained_network),
                copy.deepcopy(trained_task),
            ),
        }
        no_recurrence_network, no_recurrence_task = variants["no recurrence"]
        coding = no_recurrence_task.area_map["coding"]
        recurrent = no_recurrence_network.weights[(coding, coding)]
        recurrent.data.fill(0.0)
        recurrent.eliminate_zeros()

        condition_trajectories = {}
        for condition, (network, task) in variants.items():
            trajectories, times = evaluate_condition(
                seed,
                condition,
                network,
                task,
                mnist,
                test_images,
                test_labels,
                test_ids,
            )
            trajectory_rows.extend(trajectories)
            time_rows.extend(times)
            condition_trajectories[condition] = trajectories

        learned = pd.DataFrame(condition_trajectories["learned recurrence"])
        control = pd.DataFrame(condition_trajectories["no recurrence"])
        first_prediction_matches.append(
            bool(
                np.array_equal(
                    learned["initial_prediction"].to_numpy(),
                    control["initial_prediction"].to_numpy(),
                )
            )
        )

    trajectory_frame = pd.DataFrame(trajectory_rows)
    time_frame = pd.DataFrame(time_rows)
    seed_frame = seed_time_series(time_frame)
    aggregate_frame = aggregate_time_series(seed_frame)
    summary_rows = condition_summary(trajectory_frame)

    trajectory_frame.to_csv(args.output / "static_relaxation_trajectories.csv", index=False)
    time_frame.to_csv(args.output / "static_relaxation_raw.csv", index=False)
    seed_frame.to_csv(args.output / "static_relaxation_per_seed.csv", index=False)
    aggregate_frame.to_csv(args.output / "static_relaxation_time_series.csv", index=False)
    learned_summary = next(
        row for row in summary_rows if row["condition"] == "learned recurrence"
    )
    learned_late_cap_changes = time_frame[
        (time_frame["condition"] == "learned recurrence")
        & (time_frame["readout_r"] >= 16)
        & (time_frame["consecutive_cap_overlap"] < 1.0)
    ]
    make_figure(
        aggregate_frame,
        args.output / "static_relaxation_held_input",
        int(learned_summary["maximum_settling_readout"]),
    )

    result = {
        "protocol": {
            "seeds": list(args.seeds),
            "train_per_class": TRAIN_PER_CLASS,
            "test_per_class": TEST_PER_CLASS,
            "horizon": HORIZON,
            "training_condition": "50 distinct images per class, class organised",
            "input": "held",
            "plasticity_during_evaluation": False,
            "control": "coding-area recurrent weights set to zero after training",
            "first_predictions_paired_identically": all(first_prediction_matches),
        },
        "conditions": summary_rows,
        "trajectories_with_decoded_changes_at_readouts_16_to_20": int(
            (
                (trajectory_frame["condition"] == "learned recurrence")
                & (trajectory_frame["settling_readout"] > 15)
            ).sum()
        ),
        "microscopic_cap_change_events_at_readouts_16_to_20": int(
            len(learned_late_cap_changes)
        ),
        "trajectories_with_microscopic_cap_changes_at_readouts_16_to_20": int(
            learned_late_cap_changes[["seed", "instance_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "software": {
            "base_revision": args.base_revision,
            "pyac_source_tree_sha256": source_tree_sha256(
                args.ac_root / "pyac" / "src" / "pyac"
            ),
            "experiment_script_sha256": file_sha256(Path(__file__)),
            "training_helper_sha256": file_sha256(args.training_helper),
        },
        "data": {
            path.name: file_sha256(path)
            for path in sorted(args.data_dir.glob("*.gz"))
        },
    }
    (args.output / "static_relaxation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
