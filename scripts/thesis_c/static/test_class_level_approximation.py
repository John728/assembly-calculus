from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


N = 2000
K = 200
P = 0.1
BETA = 0.5
RAW_INPUT_K = 200
TRAIN_PER_CLASS = 50
TEST_PER_CLASS = 20
READOUTS = 5


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


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


def full_normalise(network, area_name: str | None = None) -> None:
    targets = [area_name] if area_name is not None else network.area_names
    for target in targets:
        keys = [key for key in network.weights if key[1] == target]
        if not keys:
            continue
        total = sum(
            np.asarray(network.weights[key].sum(axis=0)).ravel()
            for key in keys
        )
        total[total == 0.0] = 1.0
        for key in keys:
            matrix = network.weights[key]
            matrix.data = matrix.data / total[matrix.indices]


def balanced_subset(images, labels, per_class: int):
    indices = np.concatenate(
        [np.flatnonzero(np.asarray(labels) == digit)[:per_class] for digit in range(10)]
    )
    return images[indices], np.asarray(labels)[indices], indices


def train_model(seed: int, images, labels, mnist):
    rng = np.random.default_rng(seed)
    encoder = mnist.RawPixelEncoder(k=RAW_INPUT_K, area_name="X")
    network, task = mnist.build_mnist_network(
        n=N, k=K, p=P, beta=BETA, rng=rng, encoder=encoder
    )
    task.seed = seed
    coding = task.area_map["coding"]
    recurrent = network.weights[(coding, coding)]
    recurrent.setdiag(0.0)
    recurrent.eliminate_zeros()
    full_normalise(network)
    network.normalize = types.MethodType(
        lambda self, target=None: full_normalise(self, target), network
    )

    ordered_digits = [digit for digit in range(10) for _ in range(TRAIN_PER_CLASS)]
    transitions: dict[int, list[float]] = {digit: [] for digit in range(10)}
    original_step = network.step
    call_index = 0

    def recorded_step(*args, **kwargs):
        nonlocal call_index
        digit = ordered_digits[call_index]
        previous = network.activations[coding].copy()
        result = original_step(*args, **kwargs)
        current = network.activations[coding]
        if previous.size:
            overlap = np.intersect1d(previous, current, assume_unique=True).size / K
            transitions[digit].append(float(overlap))
        call_index += 1
        return result

    network.step = recorded_step
    mnist.train_mnist_assemblies(
        network,
        task,
        images,
        labels,
        presentation_rounds=TRAIN_PER_CLASS,
        settle_steps=1,
        class_organized=True,
        normalization_on=True,
    )
    network.step = original_step
    return network, task, transitions


def support_matrix(network, task) -> np.ndarray:
    coding = task.area_map["coding"]
    recurrent = network.weights[(coding, coding)]
    assemblies = [task.class_assemblies[digit].indices for digit in range(10)]
    matrix = np.zeros((10, 10), dtype=float)
    for source, source_indices in enumerate(assemblies):
        outgoing = np.asarray(recurrent[source_indices].sum(axis=0)).ravel()
        for target, target_indices in enumerate(assemblies):
            matrix[target, source] = float(outgoing[target_indices].sum()) / K
    return matrix


def pairwise_ordering(exact: np.ndarray, predicted: np.ndarray) -> tuple[int, int]:
    concordant = 0
    comparable = 0
    for first in range(10):
        for second in range(first + 1, 10):
            exact_delta = exact[first] - exact[second]
            predicted_delta = predicted[first] - predicted[second]
            if np.isclose(exact_delta, 0.0, atol=1e-14) or np.isclose(
                predicted_delta, 0.0, atol=1e-14
            ):
                continue
            comparable += 1
            concordant += np.sign(exact_delta) == np.sign(predicted_delta)
    return int(concordant), int(comparable)


def evaluate_model(seed, network, task, images, labels, ids):
    sensory = task.area_map["sensory"]
    coding = task.area_map["coding"]
    sensory_n = network.areas_by_name[sensory].n
    recurrent = network.weights[(coding, coding)]
    assemblies = [task.class_assemblies[digit].indices for digit in range(10)]
    neuron_class = np.empty(N, dtype=np.int64)
    for digit, indices in enumerate(assemblies):
        neuron_class[indices] = digit
    lambda_matrix = support_matrix(network, task)
    scalar_gains = np.diag(lambda_matrix)
    rows: list[dict[str, object]] = []

    for image, target, instance_id in zip(images, labels, ids):
        sensory_assembly = task.encoder.encode(image)
        stimulus = np.zeros(sensory_n, dtype=float)
        stimulus[sensory_assembly.indices] = 1.0
        network.activations[sensory] = np.array([], dtype=np.int64)
        network.activations[coding] = np.array([], dtype=np.int64)
        network.step_count = 0

        for readout in range(1, READOUTS + 1):
            network.step(
                external_stimuli={sensory: stimulus},
                plasticity_on=False,
                biases={coding: task.coding_bias},
            )
            active = network.activations[coding]
            overlaps = np.bincount(neuron_class[active], minlength=10) / K
            recurrent_to_neurons = np.asarray(recurrent[active].sum(axis=0)).ravel()
            exact = np.asarray(
                [recurrent_to_neurons[indices].sum() / K for indices in assemblies]
            )
            scalar = scalar_gains * overlaps
            matrix_prediction = lambda_matrix @ overlaps
            scalar_correct, scalar_comparable = pairwise_ordering(exact, scalar)
            matrix_correct, matrix_comparable = pairwise_ordering(exact, matrix_prediction)

            top_two = np.sort(overlaps)[-2:]
            decision_margin = float(top_two[-1] - top_two[-2])
            for digit in range(10):
                rows.append(
                    {
                        "seed": seed,
                        "instance_id": int(instance_id),
                        "target": int(target),
                        "readout_r": readout,
                        "class": digit,
                        "overlap": float(overlaps[digit]),
                        "decision_margin": decision_margin,
                        "exact_recurrent_input": float(exact[digit]),
                        "scalar_prediction": float(scalar[digit]),
                        "matrix_prediction": float(matrix_prediction[digit]),
                        "scalar_abs_error": float(abs(exact[digit] - scalar[digit])),
                        "matrix_abs_error": float(
                            abs(exact[digit] - matrix_prediction[digit])
                        ),
                        "scalar_pairwise_correct": scalar_correct,
                        "scalar_pairwise_comparable": scalar_comparable,
                        "matrix_pairwise_correct": matrix_correct,
                        "matrix_pairwise_comparable": matrix_comparable,
                    }
                )
    return rows, lambda_matrix


def regression_metrics(group: pd.DataFrame, prediction: str) -> dict[str, float]:
    x = group[prediction].to_numpy(dtype=float)
    y = group["exact_recurrent_input"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "pearson_r": float(stats.pearsonr(x, y).statistic),
        "spearman_rho": float(stats.spearmanr(x, y).statistic),
        "slope": float(slope),
        "intercept": float(intercept),
        "mae": float(np.mean(np.abs(y - x))),
        "normalised_mae": float(np.mean(np.abs(y - x)) / np.mean(np.abs(y))),
    }


def seed_readout_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, readout), group in frame.groupby(["seed", "readout_r"]):
        row: dict[str, object] = {"seed": int(seed), "readout_r": int(readout)}
        for name, column in (
            ("scalar", "scalar_prediction"),
            ("matrix", "matrix_prediction"),
        ):
            values = regression_metrics(group, column)
            row.update({f"{name}_{key}": value for key, value in values.items()})
            state_rows = group[group["class"] == 0]
            correct = state_rows[f"{name}_pairwise_correct"].sum()
            comparable = state_rows[f"{name}_pairwise_comparable"].sum()
            row[f"{name}_ordering_accuracy"] = float(correct / comparable)
            row[f"{name}_ordering_coverage"] = float(comparable / (len(state_rows) * 45))
        rows.append(row)
    return pd.DataFrame(rows)


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    return mean, mean - half, mean + half


def summarise_readouts(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value_columns = [column for column in metrics.columns if column not in {"seed", "readout_r"}]
    for readout, group in metrics.groupby("readout_r"):
        row: dict[str, object] = {"readout_r": int(readout)}
        for column in value_columns:
            mean, low, high = mean_ci(group[column])
            row[column] = mean
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
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


def make_figure(frame, matrix_frame, summary, output: Path) -> None:
    configure_plotting()
    mean_matrix = (
        matrix_frame.groupby(["target_class", "source_class"])["support"]
        .mean()
        .unstack("source_class")
        .to_numpy()
    )
    positive = mean_matrix[mean_matrix > 0]
    norm = colors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.05), gridspec_kw={"width_ratios": [1.0, 1.18, 1.0]})
    heat = axes[0].imshow(mean_matrix, cmap="viridis", norm=norm, aspect="equal")
    axes[0].set_xlabel("Source class $d$")
    axes[0].set_ylabel("Target class $c$")
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    axes[0].set_title("(a) Class-to-class support", loc="left", fontweight="bold")
    for digit in range(10):
        axes[0].text(digit, digit, f"{mean_matrix[digit, digit]:.2f}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    figure.colorbar(heat, ax=axes[0], fraction=0.046, pad=0.04)

    display = frame.sample(n=min(25000, len(frame)), random_state=42)
    axes[1].scatter(
        display["scalar_prediction"],
        display["exact_recurrent_input"],
        s=7,
        alpha=0.12,
        color="#009E8E",
        edgecolors="none",
        rasterized=True,
    )
    upper = float(max(frame["scalar_prediction"].max(), frame["exact_recurrent_input"].max()))
    axes[1].plot([0, upper], [0, upper], color="#202020", linewidth=1.0, linestyle="--")
    axes[1].set_xlim(0, upper * 1.02)
    axes[1].set_ylim(0, upper * 1.02)
    axes[1].set_xlabel(r"Predicted $\lambda_c o_{A,c}(r)$")
    axes[1].set_ylabel(r"Exact $R_c(r)$")
    axes[1].set_title("(b) Scalar approximation", loc="left", fontweight="bold")

    readouts = summary["readout_r"].to_numpy()
    means = summary["scalar_mae"].to_numpy()
    low = summary["scalar_mae_ci_low"].to_numpy()
    high = summary["scalar_mae_ci_high"].to_numpy()
    axes[2].plot(
        readouts,
        means,
        marker="o",
        markersize=4,
        linewidth=1.8,
        color="#009E8E",
    )
    axes[2].fill_between(
        readouts, low, high, color="#009E8E", alpha=0.14, linewidth=0
    )
    axes[2].set_xticks(readouts)
    axes[2].set_xlabel("Readout $r$")
    axes[2].set_ylabel("Mean absolute error")
    axes[2].set_title("(c) Error by readout", loc="left", fontweight="bold")
    axes[2].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.19, wspace=0.42)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=tuple(range(42, 52)))
    parser.add_argument("--base-revision", default="unrecorded")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))
    import pyac.tasks.mnist as mnist

    train = mnist.load_mnist_split(args.data_dir, "train")
    test = mnist.load_mnist_split(args.data_dir, "test")
    train_images, train_labels, _ = balanced_subset(train.images, train.labels, TRAIN_PER_CLASS)
    test_images, test_labels, test_ids = balanced_subset(test.images, test.labels, TEST_PER_CLASS)

    all_rows = []
    matrix_rows = []
    stability_rows = []
    selectivity_rows = []
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        network, task, transitions = train_model(seed, train_images, train_labels, mnist)
        rows, matrix = evaluate_model(seed, network, task, test_images, test_labels, test_ids)
        all_rows.extend(rows)
        for target in range(10):
            for source in range(10):
                matrix_rows.append(
                    {
                        "seed": seed,
                        "target_class": target,
                        "source_class": source,
                        "support": float(matrix[target, source]),
                    }
                )
        for digit, values in transitions.items():
            for transition, value in enumerate(values, start=2):
                stability_rows.append(
                    {
                        "seed": seed,
                        "class": digit,
                        "training_presentation": transition,
                        "eta": value,
                    }
                )
        coding = task.area_map["coding"]
        recurrent = network.weights[(coding, coding)]
        diagonal_weight = float(K * np.trace(matrix))
        total_weight = float(recurrent.sum())
        selectivity_rows.append(
            {
                "seed": seed,
                "diagonal_weight_fraction": diagonal_weight / total_weight,
                "diagonal_support_mean": float(np.diag(matrix).mean()),
                "off_diagonal_support_mean": float(matrix[~np.eye(10, dtype=bool)].mean()),
                "off_to_diagonal_support_ratio": float(
                    matrix[~np.eye(10, dtype=bool)].mean() / np.diag(matrix).mean()
                ),
            }
        )

    frame = pd.DataFrame(all_rows)
    matrix_frame = pd.DataFrame(matrix_rows)
    stability_frame = pd.DataFrame(stability_rows)
    selectivity_frame = pd.DataFrame(selectivity_rows)
    metrics = seed_readout_metrics(frame)
    summary = summarise_readouts(metrics)

    frame.to_csv(args.output / "class_level_exact_predictions.csv", index=False)
    matrix_frame.to_csv(args.output / "class_level_support_matrix.csv", index=False)
    stability_frame.to_csv(args.output / "class_level_training_stability.csv", index=False)
    selectivity_frame.to_csv(args.output / "class_level_selectivity.csv", index=False)
    metrics.to_csv(args.output / "class_level_metrics_per_seed.csv", index=False)
    summary.to_csv(args.output / "class_level_metrics_by_readout.csv", index=False)
    make_figure(frame, matrix_frame, summary, args.output / "class_level_approximation")

    overlap_bins = pd.cut(
        frame["overlap"],
        bins=[-1e-12, 1e-12, 0.1, 0.25, 1.0],
        labels=["zero", "weak", "moderate", "large"],
        include_lowest=True,
    )
    residual_by_overlap = (
        frame.assign(overlap_bin=overlap_bins)
        .groupby("overlap_bin", observed=True)
        .agg(
            observations=("scalar_abs_error", "size"),
            mean_overlap=("overlap", "mean"),
            scalar_mae=("scalar_abs_error", "mean"),
            matrix_mae=("matrix_abs_error", "mean"),
            mean_exact=("exact_recurrent_input", "mean"),
        )
        .reset_index()
    )
    residual_by_overlap["scalar_relative_mae"] = residual_by_overlap["scalar_mae"] / residual_by_overlap["mean_exact"]
    residual_by_overlap.to_csv(args.output / "class_level_error_by_overlap.csv", index=False)

    state_keys = ["seed", "instance_id", "readout_r"]
    top_two_indices = (
        frame.groupby(state_keys)["overlap"]
        .nlargest(2)
        .reset_index(level=state_keys, drop=True)
        .index
    )
    boundary_rows: list[dict[str, object]] = []
    for scope, scoped in (("all classes", frame), ("top two classes", frame.loc[top_two_indices])):
        scoped = scoped.assign(
            near_boundary=scoped["decision_margin"] <= 0.05 + 1e-12
        )
        for near_boundary, group in scoped.groupby("near_boundary"):
            per_seed = group.groupby("seed").agg(
                mae=("scalar_abs_error", "mean"),
                mean_exact=("exact_recurrent_input", "mean"),
            )
            mae_mean, mae_low, mae_high = mean_ci(per_seed["mae"])
            boundary_rows.append(
                {
                    "scope": scope,
                    "near_boundary": bool(near_boundary),
                    "margin_threshold": 0.05,
                    "observations": len(group),
                    "scalar_mae": mae_mean,
                    "scalar_mae_ci_low": mae_low,
                    "scalar_mae_ci_high": mae_high,
                    "scalar_relative_mae": float(
                        group["scalar_abs_error"].mean()
                        / group["exact_recurrent_input"].mean()
                    ),
                }
            )
    boundary_frame = pd.DataFrame(boundary_rows)
    boundary_frame.to_csv(args.output / "class_level_error_by_boundary.csv", index=False)

    top_two_ordering_rows: list[dict[str, object]] = []
    for (seed, readout), group in frame.groupby(["seed", "readout_r"]):
        correct = 0
        comparable = 0
        states = 0
        for _, state in group.groupby(["instance_id"]):
            top = state.nlargest(2, "overlap")
            first, second = top.iloc[0], top.iloc[1]
            predicted_delta = first["scalar_prediction"] - second["scalar_prediction"]
            exact_delta = first["exact_recurrent_input"] - second["exact_recurrent_input"]
            states += 1
            if abs(predicted_delta) <= 1e-14 or abs(exact_delta) <= 1e-14:
                continue
            comparable += 1
            correct += np.sign(predicted_delta) == np.sign(exact_delta)
        top_two_ordering_rows.append(
            {
                "seed": int(seed),
                "readout_r": int(readout),
                "ordering_accuracy": correct / comparable,
                "ordering_coverage": comparable / states,
            }
        )
    top_two_ordering = pd.DataFrame(top_two_ordering_rows)
    top_two_ordering.to_csv(
        args.output / "class_level_top_two_ordering.csv", index=False
    )

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        "protocol": {
            "seeds": list(args.seeds),
            "train_per_class": TRAIN_PER_CLASS,
            "test_per_class": TEST_PER_CLASS,
            "readouts": list(range(1, READOUTS + 1)),
            "n": N,
            "k": K,
            "p": P,
            "beta": BETA,
            "input_k": RAW_INPUT_K,
            "training": "class organised, 50 distinct images per class",
            "evaluation": "held input, plasticity off",
            "normalisation": "single incoming budget across sensory and recurrent fibres",
            "weight_orientation": "rows are presynaptic; columns are postsynaptic",
        },
        "training_cap_stability": {
            "mean": float(stability_frame["eta"].mean()),
            "seed_means": stability_frame.groupby("seed")["eta"].mean().to_dict(),
            "minimum": float(stability_frame["eta"].min()),
            "fifth_percentile": float(stability_frame["eta"].quantile(0.05)),
        },
        "recurrent_selectivity": {
            column: {
                "mean": mean_ci(selectivity_frame[column])[0],
                "ci_low": mean_ci(selectivity_frame[column])[1],
                "ci_high": mean_ci(selectivity_frame[column])[2],
            }
            for column in selectivity_frame.columns
            if column != "seed"
        },
        "by_readout": summary.to_dict(orient="records"),
        "error_by_overlap": residual_by_overlap.to_dict(orient="records"),
        "error_by_decision_boundary": boundary_frame.to_dict(orient="records"),
        "top_two_ordering": summarise_readouts(
            top_two_ordering.rename(
                columns={
                    "ordering_accuracy": "scalar_ordering_accuracy",
                    "ordering_coverage": "scalar_ordering_coverage",
                }
            )
        ).to_dict(orient="records"),
        "software": {
            "base_revision": args.base_revision,
            "pyac_source_tree_sha256": source_tree_sha256(
                args.ac_root / "pyac" / "src" / "pyac"
            ),
            "experiment_script_sha256": script_hash,
        },
        "data": {
            path.name: file_sha256(path)
            for path in sorted(args.data_dir.glob("*.gz"))
        },
    }
    (args.output / "class_level_approximation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
