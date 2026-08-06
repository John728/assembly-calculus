from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import t as student_t


DEFAULT_SEEDS = tuple(range(42, 52))
DEFAULT_BETAS = (0.025, 0.05, 0.1, 0.2, 0.35, 0.5)
FORMATION_PER_CLASS = 50
CORRECTION_PER_CLASS = 50
VALIDATION_PER_CLASS = 20
TEST_PER_CLASS = 20
HORIZON = 10

TEAL = "#008F7A"
INDIGO = "#3D4FA3"
CHARCOAL = "#202A33"
GRID = "#DDE4E8"
WHITE = "#FFFFFF"


def parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def parse_float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("positive correction strengths are required")
    return parsed


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


def load_static_generator(path: Path):
    spec = importlib.util.spec_from_file_location("static_training_dynamics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import baseline generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def balanced_slice(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    start_per_class: int,
    count_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected: list[np.ndarray] = []
    labels_array = np.asarray(labels)
    stop = start_per_class + count_per_class
    for digit in range(10):
        matches = np.flatnonzero(labels_array == digit)
        if len(matches) < stop:
            raise ValueError(
                f"digit {digit} has {len(matches)} examples, fewer than {stop}"
            )
        selected.append(matches[start_per_class:stop])
    indices = np.concatenate(selected)
    return images[indices], labels_array[indices], indices


def seeded_rng(model_seed: int, stream: int, instance_id: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([model_seed, stream, int(instance_id)])
    return np.random.default_rng(sequence)


def natural_first_cap(
    network,
    task,
    image: np.ndarray,
    *,
    model_seed: int,
    instance_id: int,
    pyac,
) -> np.ndarray:
    sensory = task.area_map["sensory"]
    coding = task.area_map["coding"]
    network.activations[sensory] = np.array([], dtype=np.int64)
    network.activations[coding] = np.array([], dtype=np.int64)
    network.step_count = 0
    network.rng = seeded_rng(model_seed, 101, instance_id)
    stimulus = pyac.protocol._mnist_stimulus(network, task, image)
    network.step(
        external_stimuli={sensory: stimulus},
        plasticity_on=False,
        biases={coding: task.coding_bias},
    )
    return network.activations[coding].copy()


def decode_cap(active: np.ndarray, task, pyac) -> int:
    from pyac.core.types import Assembly

    return int(
        pyac.decode_mnist_class(
            Assembly(area_name=task.area_map["coding"], indices=active),
            task,
        )
    )


def restore_column_budgets(matrix, target_sums: np.ndarray) -> None:
    current = np.asarray(matrix.sum(axis=0)).ravel()
    scales = np.ones_like(current)
    nonzero = current > 0.0
    scales[nonzero] = target_sums[nonzero] / current[nonzero]
    matrix.data *= scales[matrix.indices]


def sparse_identical(left, right) -> bool:
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def recurrent_teacher_fine_tune(
    network,
    task,
    images: np.ndarray,
    labels: np.ndarray,
    source_ids: np.ndarray,
    *,
    beta: float,
    model_seed: int,
    pyac,
) -> dict[str, object]:
    """Teacher-force wrong first caps towards their labelled assemblies.

    The sensory matrix is frozen. For every misclassified correction example,
    the natural first cap is treated as the presynaptic state and the labelled
    assembly as a clamped postsynaptic state. Only existing recurrent synapses
    are multiplied, after which every target neuron's original recurrent budget
    is restored.
    """

    from pyac.core.plasticity import hebbian_update

    sensory = task.area_map["sensory"]
    coding = task.area_map["coding"]
    sensory_weights = network.weights[(sensory, coding)]
    recurrent_weights = network.weights[(coding, coding)]
    sensory_before = sensory_weights.copy()
    recurrent_budgets = np.asarray(recurrent_weights.sum(axis=0)).ravel().copy()

    proposals = [
        natural_first_cap(
            network,
            task,
            image,
            model_seed=model_seed,
            instance_id=int(instance_id),
            pyac=pyac,
        )
        for image, instance_id in zip(images, source_ids)
    ]
    predictions = np.asarray(
        [decode_cap(cap, task, pyac) for cap in proposals],
        dtype=np.int64,
    )
    targets = np.asarray(labels, dtype=np.int64)
    wrong_indices = np.flatnonzero(predictions != targets)
    update_order = seeded_rng(model_seed, 102, 0).permutation(wrong_indices)

    for index in update_order:
        target = int(targets[index])
        hebbian_update(
            weights=recurrent_weights,
            pre_firing=proposals[index],
            post_firing=task.class_assemblies[target].indices,
            beta=beta,
        )
        restore_column_budgets(recurrent_weights, recurrent_budgets)

    final_budgets = np.asarray(recurrent_weights.sum(axis=0)).ravel()
    if not sparse_identical(sensory_weights, sensory_before):
        raise AssertionError("teacher forcing modified the frozen sensory weights")
    budget_error = float(np.max(np.abs(final_budgets - recurrent_budgets)))
    if budget_error > 1e-10:
        raise AssertionError(
            f"recurrent column budget changed by {budget_error:.3e}"
        )

    return {
        "correction_beta": beta,
        "correction_examples": len(images),
        "teacher_forced_examples": int(len(wrong_indices)),
        "correction_first_readout_accuracy": float(
            np.mean(predictions == targets)
        ),
        "sensory_weights_unchanged": True,
        "maximum_recurrent_budget_error": budget_error,
    }


def evaluate_trajectory(
    network,
    task,
    images: np.ndarray,
    labels: np.ndarray,
    source_ids: np.ndarray,
    *,
    model_seed: int,
    stream: int,
    horizon: int,
    condition: str,
    split: str,
    pyac,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    correct = np.zeros(horizon, dtype=np.int64)
    trajectories: list[dict[str, object]] = []

    for image, target_value, source_id in zip(images, labels, source_ids):
        target = int(target_value)
        network.rng = seeded_rng(model_seed, stream, int(source_id))
        result = pyac.evaluate_mnist_example(
            network,
            task,
            image,
            target,
            instance_id=int(source_id),
            t=horizon - 1,
            stimulus_mode="held",
        )
        predictions = [int(value) for value in result["trajectory"]]
        if len(predictions) != horizon:
            raise AssertionError("evaluation trajectory has the wrong length")
        correct += np.asarray(predictions) == target
        trajectories.append(
            {
                "seed": model_seed,
                "split": split,
                "condition": condition,
                "instance_id": int(source_id),
                "target": target,
                "initial_prediction": predictions[0],
                "final_prediction": predictions[-1],
                "initial_correct": predictions[0] == target,
                "final_correct": predictions[-1] == target,
                "trajectory": json.dumps(predictions),
            }
        )

    time_rows = [
        {
            "seed": model_seed,
            "split": split,
            "condition": condition,
            "readout_r": readout,
            "accuracy": correct[readout - 1] / len(images),
        }
        for readout in range(1, horizon + 1)
    ]
    return time_rows, trajectories


def select_intervention(selection: pd.DataFrame) -> tuple[float, int]:
    candidates = (
        selection[selection["readout_r"] >= 2]
        .groupby(["correction_beta", "readout_r"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "mean_validation_accuracy"})
        .sort_values(
            ["mean_validation_accuracy", "correction_beta", "readout_r"],
            ascending=[False, True, True],
            kind="stable",
        )
    )
    if candidates.empty:
        raise ValueError("no validation candidates are available")
    best = candidates.iloc[0]
    return float(best["correction_beta"]), int(best["readout_r"])


def transition_counts(
    trajectories: pd.DataFrame,
    *,
    readout: int,
) -> dict[str, int]:
    corrected = 0
    corrupted = 0
    stable_correct = 0
    stable_wrong = 0
    wrong_to_wrong = 0
    for row in trajectories.itertuples(index=False):
        predictions = json.loads(row.trajectory)
        selected = int(predictions[readout - 1])
        initial_correct = int(predictions[0]) == int(row.target)
        selected_correct = selected == int(row.target)
        if not initial_correct and selected_correct:
            corrected += 1
        elif initial_correct and not selected_correct:
            corrupted += 1
        elif initial_correct:
            stable_correct += 1
        elif selected == int(predictions[0]):
            stable_wrong += 1
        else:
            wrong_to_wrong += 1
    return {
        "corrected": corrected,
        "corrupted": corrupted,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "wrong_to_different_wrong": wrong_to_wrong,
    }


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, mean, mean
    half = (
        float(student_t.ppf(0.975, len(data) - 1))
        * float(np.std(data, ddof=1))
        / np.sqrt(len(data))
    )
    return mean, mean - half, mean + half


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.9,
            "grid.linewidth": 0.8,
        }
    )


def plot_accuracy(time_frame: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    styles = {
        "baseline": ("Original recurrent dynamics", TEAL),
        "correction_trained": ("Correction-trained recurrence", INDIGO),
    }
    fig, axis = plt.subplots(figsize=(6.4, 6.4))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.93)

    for condition, (label, colour) in styles.items():
        subset = time_frame[time_frame["condition"] == condition]
        readouts = np.sort(subset["readout_r"].unique())
        means: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for readout in readouts:
            values = subset.loc[
                subset["readout_r"] == readout, "accuracy"
            ].to_numpy()
            mean, low, high = mean_interval(values)
            means.append(mean)
            lower.append(low)
            upper.append(high)
        axis.fill_between(
            readouts,
            np.clip(lower, 0.0, 1.0),
            np.clip(upper, 0.0, 1.0),
            color=colour,
            alpha=0.13,
            linewidth=0,
        )
        axis.plot(
            readouts,
            means,
            color=colour,
            linewidth=3.0,
            solid_capstyle="round",
            marker="o",
            markersize=5.0,
            markeredgecolor=WHITE,
            markeredgewidth=0.9,
            label=label,
        )

    axis.set(
        xlabel="Internal update",
        ylabel="Classification accuracy",
        xlim=(1, int(time_frame["readout_r"].max())),
        ylim=(0.0, 0.72),
        xticks=np.arange(1, int(time_frame["readout_r"].max()) + 1),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(frameon=False, loc="upper left")

    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 400} if extension == "png" else {}
        fig.savefig(
            output / f"mnist_temporal_correction_accuracy.{extension}",
            facecolor=WHITE,
            metadata={
                "Title": "MNIST temporal correction accuracy",
                "Creator": "Matplotlib",
            },
            **kwargs,
        )
    plt.close(fig)


def condition_summary(
    time_frame: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    readout: int,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for condition in ("baseline", "correction_trained"):
        subset = time_frame[time_frame["condition"] == condition]
        initial = subset[subset["readout_r"] == 1]["accuracy"].to_numpy()
        selected = subset[subset["readout_r"] == readout]["accuracy"].to_numpy()
        final_r = int(subset["readout_r"].max())
        final = subset[subset["readout_r"] == final_r]["accuracy"].to_numpy()
        condition_trajectories = trajectories[
            trajectories["condition"] == condition
        ]
        summary[condition] = {
            "mean_initial_accuracy": float(np.mean(initial)),
            "mean_selected_readout_accuracy": float(np.mean(selected)),
            "mean_final_accuracy": float(np.mean(final)),
            "transitions_at_selected_readout": transition_counts(
                condition_trajectories,
                readout=readout,
            ),
        }
    return summary


def write_readme(output: Path) -> None:
    text = """# Temporal correction experiment

This experiment starts from the thesis MNIST baseline, freezes the sensory
weights, and teacher-forces only recurrent transitions from misclassified first
caps to their labelled assemblies. Each target neuron's original recurrent
weight budget is restored after every update.

The correction strength and reported readout are selected using a disjoint
validation subset. The plotted test subset is not used during selection.

Files:

- `mnist_temporal_correction_accuracy.{png,pdf,svg}`: mean test accuracy over
  network seeds, with 95% t intervals across seeds.
- `mnist_temporal_correction_time_series.csv`: per-seed test accuracy.
- `mnist_temporal_correction_trajectories.csv`: per-example test trajectories.
- `mnist_temporal_correction_validation.csv`: validation sweep used for
  selection.
- `mnist_temporal_correction_summary.json`: protocol, controls, selected
  configuration, and headline results.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=parse_int_tuple,
        default=DEFAULT_SEEDS,
        help="comma-separated network seeds",
    )
    parser.add_argument(
        "--candidate-betas",
        type=parse_float_tuple,
        default=DEFAULT_BETAS,
        help="comma-separated recurrent correction strengths",
    )
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    if args.horizon < 2:
        raise ValueError("horizon must be at least 2")

    args.output.mkdir(parents=True, exist_ok=True)
    baseline_script = (
        args.ac_root
        / "scripts/thesis_c/static/generate_static_training_dynamics.py"
    )
    baseline_generator = load_static_generator(baseline_script)
    pyac = baseline_generator.load_pyac(args.ac_root)

    train = pyac.load_mnist_split(args.data_dir, "train")
    test = pyac.load_mnist_split(args.data_dir, "test")
    formation_images, formation_labels, formation_ids = balanced_slice(
        train.images,
        train.labels,
        start_per_class=0,
        count_per_class=FORMATION_PER_CLASS,
    )
    correction_images, correction_labels, correction_ids = balanced_slice(
        train.images,
        train.labels,
        start_per_class=FORMATION_PER_CLASS,
        count_per_class=CORRECTION_PER_CLASS,
    )
    validation_images, validation_labels, validation_ids = balanced_slice(
        train.images,
        train.labels,
        start_per_class=FORMATION_PER_CLASS + CORRECTION_PER_CLASS,
        count_per_class=VALIDATION_PER_CLASS,
    )
    test_images, test_labels, test_ids = balanced_slice(
        test.images,
        test.labels,
        start_per_class=0,
        count_per_class=TEST_PER_CLASS,
    )
    if (
        set(formation_ids) & set(correction_ids)
        or set(formation_ids) & set(validation_ids)
        or set(correction_ids) & set(validation_ids)
    ):
        raise AssertionError("training subsets are not disjoint")

    formation_schedule, formation_schedule_labels = (
        baseline_generator.build_training_schedule(
            formation_images,
            formation_labels,
            "distinct_50",
        )
    )

    base_models: list[tuple[int, object, object]] = []
    validation_rows: list[dict[str, object]] = []
    selection_diagnostics: list[dict[str, object]] = []

    for seed in args.seeds:
        print(f"selection seed {seed}", flush=True)
        network, task, _ = baseline_generator.train_model(
            seed=seed,
            images=formation_schedule,
            labels=formation_schedule_labels,
            pyac=pyac,
        )
        base_models.append((seed, network, task))

        for beta in args.candidate_betas:
            candidate_network = copy.deepcopy(network)
            candidate_task = copy.deepcopy(task)
            diagnostics = recurrent_teacher_fine_tune(
                candidate_network,
                candidate_task,
                correction_images,
                correction_labels,
                correction_ids,
                beta=beta,
                model_seed=seed,
                pyac=pyac,
            )
            time_rows, _ = evaluate_trajectory(
                candidate_network,
                candidate_task,
                validation_images,
                validation_labels,
                validation_ids,
                model_seed=seed,
                stream=201,
                horizon=args.horizon,
                condition="correction_candidate",
                split="validation",
                pyac=pyac,
            )
            for row in time_rows:
                validation_rows.append(
                    {**row, "correction_beta": beta}
                )
            selection_diagnostics.append({"seed": seed, **diagnostics})

    validation_frame = pd.DataFrame(validation_rows)
    selected_beta, selected_readout = select_intervention(validation_frame)
    print(
        f"selected beta={selected_beta:g}, readout={selected_readout}",
        flush=True,
    )

    test_time_rows: list[dict[str, object]] = []
    test_trajectory_rows: list[dict[str, object]] = []
    final_diagnostics: list[dict[str, object]] = []

    for seed, network, task in base_models:
        print(f"test seed {seed}", flush=True)
        baseline_network = copy.deepcopy(network)
        baseline_task = copy.deepcopy(task)
        baseline_time, baseline_trajectories = evaluate_trajectory(
            baseline_network,
            baseline_task,
            test_images,
            test_labels,
            test_ids,
            model_seed=seed,
            stream=301,
            horizon=args.horizon,
            condition="baseline",
            split="test",
            pyac=pyac,
        )

        corrected_network = copy.deepcopy(network)
        corrected_task = copy.deepcopy(task)
        diagnostics = recurrent_teacher_fine_tune(
            corrected_network,
            corrected_task,
            correction_images,
            correction_labels,
            correction_ids,
            beta=selected_beta,
            model_seed=seed,
            pyac=pyac,
        )
        corrected_time, corrected_trajectories = evaluate_trajectory(
            corrected_network,
            corrected_task,
            test_images,
            test_labels,
            test_ids,
            model_seed=seed,
            stream=301,
            horizon=args.horizon,
            condition="correction_trained",
            split="test",
            pyac=pyac,
        )

        baseline_initial = [
            row["initial_prediction"] for row in baseline_trajectories
        ]
        corrected_initial = [
            row["initial_prediction"] for row in corrected_trajectories
        ]
        if baseline_initial != corrected_initial:
            raise AssertionError(
                f"seed {seed}: recurrent fine-tuning changed the first predictions"
            )

        test_time_rows.extend(baseline_time)
        test_time_rows.extend(corrected_time)
        test_trajectory_rows.extend(baseline_trajectories)
        test_trajectory_rows.extend(corrected_trajectories)
        final_diagnostics.append({"seed": seed, **diagnostics})

    time_frame = pd.DataFrame(test_time_rows)
    trajectory_frame = pd.DataFrame(test_trajectory_rows)
    diagnostics_frame = pd.DataFrame(selection_diagnostics)
    final_diagnostics_frame = pd.DataFrame(final_diagnostics)

    first_readout = time_frame[time_frame["readout_r"] == 1].pivot(
        index="seed",
        columns="condition",
        values="accuracy",
    )
    if not np.array_equal(
        first_readout["baseline"].to_numpy(),
        first_readout["correction_trained"].to_numpy(),
    ):
        raise AssertionError("condition-level first-readout accuracies differ")

    validation_frame.to_csv(
        args.output / "mnist_temporal_correction_validation.csv",
        index=False,
    )
    time_frame.to_csv(
        args.output / "mnist_temporal_correction_time_series.csv",
        index=False,
    )
    trajectory_frame.to_csv(
        args.output / "mnist_temporal_correction_trajectories.csv",
        index=False,
    )

    summary = {
        "protocol": {
            "seeds": list(args.seeds),
            "formation_images_per_class": FORMATION_PER_CLASS,
            "correction_images_per_class": CORRECTION_PER_CLASS,
            "validation_images_per_class": VALIDATION_PER_CLASS,
            "test_images_per_class": TEST_PER_CLASS,
            "candidate_correction_betas": list(args.candidate_betas),
            "horizon": args.horizon,
            "formation_condition": "distinct_50",
            "correction_examples": "misclassified natural first caps only",
            "correction_rule": (
                "multiply existing recurrent synapses from natural first cap "
                "to labelled assembly"
            ),
            "sensory_weights_during_correction": "frozen",
            "recurrent_budget_during_correction": (
                "original per-target-neuron column sum preserved"
            ),
            "plasticity_during_evaluation": False,
            "stimulus_during_evaluation": "held",
            "readout_indexing": "r=1 is the state after one network update",
            "formation_source_indices": formation_ids.tolist(),
            "correction_source_indices": correction_ids.tolist(),
            "validation_source_indices": validation_ids.tolist(),
            "test_source_indices": test_ids.tolist(),
        },
        "selection": {
            "selected_correction_beta": selected_beta,
            "selected_readout": selected_readout,
            "criterion": (
                "maximum mean validation accuracy over seeds and readouts r>=2"
            ),
            "teacher_forced_examples_per_seed": {
                str(int(row.seed)): int(row.teacher_forced_examples)
                for row in final_diagnostics_frame.itertuples(index=False)
            },
        },
        "test": condition_summary(
            time_frame,
            trajectory_frame,
            readout=selected_readout,
        ),
        "controls": {
            "first_predictions_identical": True,
            "sensory_weights_unchanged": bool(
                final_diagnostics_frame["sensory_weights_unchanged"].all()
            ),
            "maximum_recurrent_budget_error": float(
                final_diagnostics_frame[
                    "maximum_recurrent_budget_error"
                ].max()
            ),
        },
        "software": {
            "base_revision": git_revision(args.ac_root),
            "pyac_core_dirty": relevant_core_dirty(args.ac_root),
            "experiment_script": str(Path(__file__).resolve()),
            "experiment_script_sha256": file_sha256(Path(__file__).resolve()),
            "baseline_script": str(baseline_script.resolve()),
            "baseline_script_sha256": file_sha256(baseline_script),
        },
        "data": {
            path.name: file_sha256(path)
            for path in sorted(args.data_dir.glob("*.gz"))
        },
    }
    (args.output / "mnist_temporal_correction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(args.output)
    plot_accuracy(time_frame, args.output)

    baseline = summary["test"]["baseline"]
    corrected = summary["test"]["correction_trained"]
    print(
        "test mean accuracy: "
        f"r=1 {baseline['mean_initial_accuracy']:.3f}; "
        f"baseline r={selected_readout} "
        f"{baseline['mean_selected_readout_accuracy']:.3f}; "
        f"correction-trained r={selected_readout} "
        f"{corrected['mean_selected_readout_accuracy']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
