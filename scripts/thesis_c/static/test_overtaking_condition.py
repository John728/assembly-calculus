from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HORIZON = 20
EARLY_TRANSITIONS = 6
TRAIN_PER_CLASS = 50
TEST_PER_CLASS = 20
BOOTSTRAP_SAMPLES = 5000


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


def support_matrix(network, task) -> np.ndarray:
    coding = task.area_map["coding"]
    recurrent = network.weights[(coding, coding)]
    assemblies, _ = class_partition(task, recurrent.shape[0])
    matrix = np.zeros((10, 10), dtype=float)
    for target, target_indices in enumerate(assemblies):
        for source, source_indices in enumerate(assemblies):
            matrix[target, source] = (
                float(recurrent[source_indices][:, target_indices].sum())
                / len(target_indices)
            )
    return matrix


def strongest_rival(overlaps: np.ndarray, winner: int) -> int:
    masked = overlaps.copy()
    masked[winner] = -np.inf
    return int(np.argmax(masked))


def evaluate_model(seed, network, task, mnist, images, labels, ids):
    sensory = task.area_map["sensory"]
    coding = task.area_map["coding"]
    sensory_n = network.areas_by_name[sensory].n
    sensory_weights = network.weights[(sensory, coding)]
    recurrent = network.weights[(coding, coding)]
    assemblies, membership = class_partition(task, recurrent.shape[0])
    lambda_matrix = support_matrix(network, task)
    gains = np.diag(lambda_matrix)

    transition_rows: list[dict[str, object]] = []
    ambiguity_rows: list[dict[str, object]] = []
    for image, target_value, instance_id_value in zip(images, labels, ids):
        target = int(target_value)
        instance_id = int(instance_id_value)
        sensory_assembly = task.encoder.encode(image)
        stimulus = np.zeros(sensory_n, dtype=float)
        stimulus[sensory_assembly.indices] = 1.0
        fixed_to_neurons = np.asarray(
            sensory_weights[sensory_assembly.indices].sum(axis=0)
        ).ravel() + np.asarray(task.coding_bias, dtype=float)
        fixed_class_input = np.asarray(
            [fixed_to_neurons[indices].mean() for indices in assemblies],
            dtype=float,
        )

        network.activations[sensory] = np.array([], dtype=np.int64)
        network.activations[coding] = np.array([], dtype=np.int64)
        network.step_count = 0

        predictions: list[int] = []
        overlaps_by_readout: list[np.ndarray] = []
        exact_recurrence_by_readout: list[np.ndarray] = []
        for _ in range(HORIZON):
            network.step(
                external_stimuli={sensory: stimulus},
                plasticity_on=False,
                biases={coding: task.coding_bias},
            )
            active = network.activations[coding]
            overlaps = np.bincount(membership[active], minlength=10) / task.k
            recurrent_to_neurons = np.asarray(recurrent[active].sum(axis=0)).ravel()
            exact_recurrence = np.asarray(
                [recurrent_to_neurons[indices].mean() for indices in assemblies],
                dtype=float,
            )
            overlaps_by_readout.append(overlaps)
            exact_recurrence_by_readout.append(exact_recurrence)
            predictions.append(
                int(mnist.decode_mnist_class(network.get_assembly(coding), task))
            )

        for index in range(EARLY_TRANSITIONS):
            readout = index + 1
            overlaps = overlaps_by_readout[index]
            current = predictions[index]
            challenger = strongest_rival(overlaps, current)
            next_winner = predictions[index + 1]
            fixed_margin = fixed_class_input[current] - fixed_class_input[challenger]
            approximate_recurrence = gains * overlaps
            approximate_advantage = (
                approximate_recurrence[challenger]
                - approximate_recurrence[current]
            )
            exact_recurrence = exact_recurrence_by_readout[index]
            exact_advantage = exact_recurrence[challenger] - exact_recurrence[current]
            transition_rows.append(
                {
                    "seed": seed,
                    "instance_id": instance_id,
                    "target": target,
                    "readout_r": readout,
                    "current_winner": current,
                    "challenger": challenger,
                    "next_winner": next_winner,
                    "challenger_overtakes": next_winner == challenger,
                    "any_next_revision": next_winner != current,
                    "fixed_margin": float(fixed_margin),
                    "approximate_recurrent_advantage": float(approximate_advantage),
                    "exact_recurrent_advantage": float(exact_advantage),
                    "approximate_overtaking_score": float(approximate_advantage - fixed_margin),
                    "exact_overtaking_score": float(exact_advantage - fixed_margin),
                    "gain_advantage": float(gains[challenger] - gains[current]),
                    "current_overlap": float(overlaps[current]),
                    "challenger_overlap": float(overlaps[challenger]),
                    "overlap_margin": float(overlaps[current] - overlaps[challenger]),
                    "approximation_error_difference": float(
                        exact_advantage - approximate_advantage
                    ),
                }
            )

        first_overlaps = overlaps_by_readout[0]
        first_winner = predictions[0]
        first_rival = strongest_rival(first_overlaps, first_winner)
        ambiguity_rows.append(
            {
                "seed": seed,
                "instance_id": instance_id,
                "target": target,
                "first_winner": first_winner,
                "strongest_rival": first_rival,
                "first_overlap_margin": float(
                    first_overlaps[first_winner] - first_overlaps[first_rival]
                ),
                "final_winner": predictions[-1],
                "final_revision": predictions[-1] != first_winner,
                "final_correction": (
                    predictions[0] != target and predictions[-1] == target
                ),
            }
        )

    return transition_rows, ambiguity_rows, lambda_matrix


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(scores, method="average")
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def predictor_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    labels = frame.challenger_overtakes.to_numpy(dtype=bool)
    scores = frame[score_column].to_numpy(dtype=float)
    predicted = scores > 0.0
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    errors = predicted != labels
    distance = np.abs(scores)
    closest_quartile = distance <= np.quantile(distance, 0.25)
    return {
        "observations": int(len(frame)),
        "overtaking_events": int(labels.sum()),
        "above_boundary": int(predicted.sum()),
        "events_above_boundary": tp,
        "events_below_boundary": fn,
        "event_fraction_above_boundary": float(recall),
        "non_event_fraction_below_boundary": float(specificity),
        "event_rate_above_boundary": float(tp / predicted.sum()) if predicted.sum() else float("nan"),
        "event_rate_below_boundary": float(fn / (~predicted).sum()) if (~predicted).sum() else float("nan"),
        "precision": float(precision),
        "balanced_accuracy": float((recall + specificity) / 2),
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "classification_errors": int(errors.sum()),
        "error_fraction_in_closest_score_quartile": float(
            np.sum(errors & closest_quartile) / errors.sum()
        ) if errors.sum() else 0.0,
        "median_absolute_score_for_errors": float(np.median(distance[errors])) if errors.sum() else 0.0,
        "median_absolute_score_for_correct_cases": float(np.median(distance[~errors])) if (~errors).sum() else 0.0,
    }


def score_comparison(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    labels = frame.challenger_overtakes.to_numpy(dtype=bool)
    columns = {
        "fixed_input_ambiguity_alone": "negative_fixed_margin",
        "current_overlap_ambiguity_alone": "negative_overlap_margin",
        "gain_advantage_alone": "gain_advantage",
        "overlap_weighted_recurrent_advantage": "approximate_recurrent_advantage",
        "approximate_full_overtaking_score": "approximate_overtaking_score",
        "exact_full_overtaking_score": "exact_overtaking_score",
    }
    working = frame.assign(
        negative_fixed_margin=-frame.fixed_margin,
        negative_overlap_margin=-frame.overlap_margin,
    )
    return {
        name: {
            "roc_auc": roc_auc(labels, working[column].to_numpy(dtype=float)),
            "average_precision": average_precision(labels, working[column].to_numpy(dtype=float)),
        }
        for name, column in columns.items()
    }


def build_margin_bins(frame: pd.DataFrame, number: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, number + 1)
    edges = np.unique(np.quantile(frame.first_overlap_margin, quantiles))
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    result = frame.copy()
    result["margin_bin"] = pd.cut(
        result.first_overlap_margin,
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(int)
    return result, edges


def bootstrap_margin_curve(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    binned, edges = build_margin_bins(frame)
    seed_bin = (
        binned.groupby(["seed", "margin_bin"], as_index=False)
        .agg(
            revisions=("final_revision", "sum"),
            observations=("final_revision", "size"),
            margin_sum=("first_overlap_margin", "sum"),
        )
    )
    seeds = np.sort(seed_bin.seed.unique())
    bins = np.sort(seed_bin.margin_bin.unique())
    rows: list[dict[str, object]] = []
    for margin_bin in bins:
        subset = seed_bin[seed_bin.margin_bin == margin_bin].set_index("seed")
        revisions = subset.revisions.reindex(seeds, fill_value=0).to_numpy(dtype=float)
        observations = subset.observations.reindex(seeds, fill_value=0).to_numpy(dtype=float)
        margin_sum = subset.margin_sum.reindex(seeds, fill_value=0).to_numpy(dtype=float)
        draws = rng.integers(0, len(seeds), size=(BOOTSTRAP_SAMPLES, len(seeds)))
        boot_revisions = revisions[draws].sum(axis=1)
        boot_observations = observations[draws].sum(axis=1)
        boot_probability = boot_revisions / boot_observations
        rows.append(
            {
                "margin_bin": int(margin_bin),
                "edge_low": float(edges[margin_bin]),
                "edge_high": float(edges[margin_bin + 1]),
                "mean_margin": float(margin_sum.sum() / observations.sum()),
                "revision_probability": float(revisions.sum() / observations.sum()),
                "ci_low": float(np.quantile(boot_probability, 0.025)),
                "ci_high": float(np.quantile(boot_probability, 0.975)),
                "observations": int(observations.sum()),
            }
        )
    return pd.DataFrame(rows)


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


def overtaking_limits(frame: pd.DataFrame, x_column: str, y_column: str) -> tuple[tuple[float, float], tuple[float, float]]:
    events = frame[frame.challenger_overtakes]
    x = frame[x_column].to_numpy(dtype=float)
    y = frame[y_column].to_numpy(dtype=float)
    x_low, x_high = np.quantile(x, [0.005, 0.995])
    y_low, y_high = np.quantile(y, [0.005, 0.995])
    if len(events):
        x_low = min(x_low, float(events[x_column].min()))
        x_high = max(x_high, float(events[x_column].max()))
        y_low = min(y_low, float(events[y_column].min()))
        y_high = max(y_high, float(events[y_column].max()))
    x_padding = 0.04 * (x_high - x_low)
    y_padding = 0.04 * (y_high - y_low)
    return (x_low - x_padding, x_high + x_padding), (y_low - y_padding, y_high + y_padding)


def draw_overtaking_panel(axis, frame: pd.DataFrame, advantage_column: str, title: str) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "overtaking", ["#E7E7E7", "#9BD3CC", "#009E8E"]
    )
    x_limits, y_limits = overtaking_limits(frame, "fixed_margin", advantage_column)
    visible = frame[
        frame.fixed_margin.between(*x_limits)
        & frame[advantage_column].between(*y_limits)
    ]
    collection = axis.hexbin(
        visible.fixed_margin,
        visible[advantage_column],
        C=visible.challenger_overtakes.astype(float),
        reduce_C_function=np.mean,
        gridsize=25,
        mincnt=8,
        cmap=cmap,
        vmin=0.0,
        linewidths=0.25,
        edgecolors="white",
    )
    bin_values = np.ma.filled(collection.get_array(), np.nan)
    positive_bins = bin_values[np.isfinite(bin_values) & (bin_values > 0)]
    collection.set_clim(0.0, max(0.1, float(np.quantile(positive_bins, 0.95))) if len(positive_bins) else 0.1)
    events = visible[visible.challenger_overtakes]
    axis.scatter(
        events.fixed_margin,
        events[advantage_column],
        marker="x",
        s=10,
        linewidths=0.5,
        color="#C44E38",
        alpha=0.52,
        label="Challenger overtakes",
        zorder=3,
    )
    diagonal_low = max(x_limits[0], y_limits[0])
    diagonal_high = min(x_limits[1], y_limits[1])
    axis.plot(
        [diagonal_low, diagonal_high],
        [diagonal_low, diagonal_high],
        linestyle="--",
        linewidth=1.2,
        color="#303030",
        label="Predicted boundary",
        zorder=2,
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(r"Fixed-input margin $M_{c,d}(u)$")
    axis.set_ylabel(r"Recurrent advantage $A_{d,c}(r)$")
    axis.legend(frameon=False, fontsize=7.2, loc="upper right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    return collection


def make_main_figure(transitions: pd.DataFrame, margin_curve: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    figure, axes = plt.subplots(1, 2, figsize=(7.9, 3.25))
    collection = draw_overtaking_panel(
        axes[0],
        transitions,
        "approximate_recurrent_advantage",
        "(a) The overtaking condition",
    )
    colourbar = figure.colorbar(collection, ax=axes[0], fraction=0.047, pad=0.025)
    colourbar.set_label("Empirical overtaking probability", fontsize=8)
    colourbar.ax.tick_params(labelsize=7)

    x = margin_curve.mean_margin.to_numpy()
    y = margin_curve.revision_probability.to_numpy()
    low = margin_curve.ci_low.to_numpy()
    high = margin_curve.ci_high.to_numpy()
    axes[1].plot(x, y, color="#009E8E", linewidth=2.0, marker="o", markersize=4.2)
    axes[1].fill_between(x, low, high, color="#009E8E", alpha=0.16, linewidth=0)
    axes[1].set_title("(b) First-readout ambiguity", loc="left", fontweight="bold")
    axes[1].set_xlabel(r"Initial overlap margin $\Delta_o(u)$")
    axes[1].set_ylabel(r"$\Pr[\widehat y(20)\ne\widehat y(1)]$")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.88, wspace=0.34)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def make_exact_figure(transitions: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    figure, axis = plt.subplots(figsize=(4.0, 3.45))
    collection = draw_overtaking_panel(
        axis,
        transitions,
        "exact_recurrent_advantage",
        "Exact realised recurrence",
    )
    colourbar = figure.colorbar(collection, ax=axis, fraction=0.047, pad=0.025)
    colourbar.set_label("Empirical overtaking probability", fontsize=8)
    colourbar.ax.tick_params(labelsize=7)
    figure.subplots_adjust(left=0.18, right=0.91, bottom=0.16, top=0.89)
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

    transition_rows: list[dict[str, object]] = []
    ambiguity_rows: list[dict[str, object]] = []
    support_matrices: list[np.ndarray] = []
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        network, task, _ = helpers.train_model(seed, train_images, train_labels, mnist)
        transitions, ambiguity, matrix = evaluate_model(
            seed, network, task, mnist, test_images, test_labels, test_ids
        )
        transition_rows.extend(transitions)
        ambiguity_rows.extend(ambiguity)
        support_matrices.append(matrix)

    transition_frame = pd.DataFrame(transition_rows)
    ambiguity_frame = pd.DataFrame(ambiguity_rows)
    margin_curve = bootstrap_margin_curve(ambiguity_frame, np.random.default_rng(9231))
    transition_frame.to_csv(args.output / "overtaking_transitions.csv", index=False)
    ambiguity_frame.to_csv(args.output / "overtaking_first_readout.csv", index=False)
    margin_curve.to_csv(args.output / "overtaking_margin_curve.csv", index=False)
    np.save(args.output / "overtaking_support_matrices.npy", np.stack(support_matrices))

    make_main_figure(transition_frame, margin_curve, args.output / "overtaking_condition")
    make_exact_figure(transition_frame, args.output / "overtaking_exact_recurrence")

    change_events = transition_frame[transition_frame.any_next_revision]
    approximate_metrics = predictor_metrics(
        transition_frame, "approximate_overtaking_score"
    )
    exact_metrics = predictor_metrics(transition_frame, "exact_overtaking_score")
    per_readout = {
        str(int(readout)): {
            "observations": int(len(group)),
            "any_next_revision": int(group.any_next_revision.sum()),
            "challenger_overtakes": int(group.challenger_overtakes.sum()),
            "approximate": predictor_metrics(group, "approximate_overtaking_score"),
            "exact": predictor_metrics(group, "exact_overtaking_score"),
        }
        for readout, group in transition_frame.groupby("readout_r")
    }
    result = {
        "protocol": {
            "seeds": list(args.seeds),
            "train_per_class": TRAIN_PER_CLASS,
            "test_per_class": TEST_PER_CLASS,
            "horizon": HORIZON,
            "analysed_transitions": list(range(1, EARLY_TRANSITIONS + 1)),
            "challenger_selection": "largest current overlap excluding the decoded winner, selected before observing the next readout",
            "input": "held",
            "plasticity_during_evaluation": False,
            "margin_bootstrap": f"{BOOTSTRAP_SAMPLES} cluster resamples of network seed",
        },
        "transition_counts": {
            "observations": int(len(transition_frame)),
            "any_next_revision": int(transition_frame.any_next_revision.sum()),
            "challenger_overtakes": int(transition_frame.challenger_overtakes.sum()),
            "fraction_of_revisions_to_preselected_strongest_rival": float(
                change_events.challenger_overtakes.mean()
            ) if len(change_events) else float("nan"),
        },
        "approximate_boundary": approximate_metrics,
        "exact_boundary": exact_metrics,
        "score_comparison": score_comparison(transition_frame),
        "per_readout": per_readout,
        "first_readout_ambiguity": {
            "trajectories": int(len(ambiguity_frame)),
            "final_revisions": int(ambiguity_frame.final_revision.sum()),
            "roc_auc_using_negative_overlap_margin": roc_auc(
                ambiguity_frame.final_revision.to_numpy(dtype=bool),
                -ambiguity_frame.first_overlap_margin.to_numpy(dtype=float),
            ),
            "spearman_margin_vs_revision": float(
                stats.spearmanr(
                    ambiguity_frame.first_overlap_margin,
                    ambiguity_frame.final_revision.astype(float),
                ).statistic
            ),
            "bins": margin_curve.to_dict(orient="records"),
        },
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
    (args.output / "overtaking_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
