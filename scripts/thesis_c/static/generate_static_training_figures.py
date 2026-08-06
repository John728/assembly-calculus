from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as student_t


COLOURS = {
    "distinct_1": "#CC3311",
    "distinct_10": "#0077BB",
    "distinct_50": "#009988",
    "gain_balanced": "#332288",
    "gain_reversed": "#EE7733",
    "repeat_1x50": "#CC6677",
    "blocked_10x5": "#AA4499",
    "interleaved_10x5": "#44AA99",
}

LABELS = {
    "distinct_1": "1 distinct image",
    "distinct_10": "10 distinct images",
    "distinct_50": "50 distinct images",
    "gain_balanced": "Gain balanced",
    "gain_reversed": "Gain order reversed",
    "repeat_1x50": "1 repeated",
    "blocked_10x5": "10 blocked",
    "interleaved_10x5": "10 interleaved",
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def mean_interval(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, mean, mean
    critical = float(student_t.ppf(0.975, len(data) - 1))
    half = critical * float(np.std(data, ddof=1)) / np.sqrt(len(data))
    return mean, mean - half, mean + half


def plot_time_curve(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    condition: str,
    metric: str,
    max_readout: int,
) -> None:
    subset = frame[
        (frame["condition"] == condition)
        & (frame["readout_r"] <= max_readout)
    ]
    readouts = np.sort(subset["readout_r"].unique())
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for readout in readouts:
        values = subset.loc[subset["readout_r"] == readout, metric]
        mean, low, high = mean_interval(values)
        means.append(mean)
        lower.append(low)
        upper.append(high)
    axis.plot(
        readouts,
        means,
        color=COLOURS[condition],
        lw=1.7,
        label=LABELS[condition],
    )
    axis.fill_between(
        readouts,
        np.clip(lower, 0, 1),
        np.clip(upper, 0, 1),
        color=COLOURS[condition],
        alpha=0.12,
        linewidth=0,
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def exposure_figure(time_frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), sharex=True)
    conditions = ("distinct_1", "distinct_10", "distinct_50")
    for condition in conditions:
        plot_time_curve(
            axes[0],
            time_frame,
            condition=condition,
            metric="accuracy",
            max_readout=20,
        )
        plot_time_curve(
            axes[1],
            time_frame,
            condition=condition,
            metric="unsettled",
            max_readout=20,
        )
    axes[0].set(
        xlabel="Internal update $r$",
        ylabel="Classification accuracy",
        xlim=(1, 20),
        ylim=(0.25, 0.65),
    )
    axes[1].set(
        xlabel="Internal update $r$",
        ylabel="Trajectories changing again",
        xlim=(1, 20),
        ylim=(0, 0.35),
    )
    axes[0].text(-0.14, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.14, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save_figure(fig, output, "mnist_static_settling")


def imprint_figure(
    class_frame: pd.DataFrame,
    trajectory_frame: pd.DataFrame,
    output: Path,
) -> None:
    baseline_classes = class_frame[class_frame["condition"] == "distinct_50"]
    digit_means = (
        baseline_classes.groupby("digit", as_index=False)
        .agg(
            sensory_pair_overlap=("sensory_pair_overlap", "mean"),
            within_class_gain=("within_class_gain", "mean"),
        )
        .sort_values("digit")
    )
    switched = trajectory_frame[
        (trajectory_frame["condition"] == "distinct_50")
        & (trajectory_frame["switch_count"] > 0)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    axis = axes[0]
    x = digit_means["sensory_pair_overlap"].to_numpy()
    y = digit_means["within_class_gain"].to_numpy()
    axis.scatter(x, y, s=40, color="#0077BB", edgecolor="white", linewidth=0.6)
    for row in digit_means.itertuples():
        axis.annotate(
            str(int(row.digit)),
            (row.sensory_pair_overlap, row.within_class_gain),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        axis.plot(line_x, intercept + slope * line_x, color="#555555", lw=1.1)
    r_value = float(pearsonr(x, y).statistic)
    axis.text(
        0.04,
        0.07,
        rf"$r={r_value:.2f}$",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    axis.set(
        xlabel="Mean overlap between training-image caps",
        ylabel="Learned recurrent gain $\\lambda_c$",
    )

    axis = axes[1]
    switched = switched.assign(
        display_type=switched["transition_type"].where(
            switched["transition_type"].isin(["corrected", "corrupted"]),
            "other revision",
        )
    )
    transition_colours = {
        "corrected": "#117733",
        "corrupted": "#CC3311",
        "other revision": "#777777",
    }
    for transition_type in ("corrected", "corrupted", "other revision"):
        group = switched[switched["display_type"] == transition_type]
        if group.empty:
            continue
        axis.scatter(
            group["initial_gain"],
            group["final_gain"],
            s=22,
            alpha=0.58,
            color=transition_colours.get(transition_type, "#777777"),
            edgecolor="none",
            label=transition_type.capitalize(),
        )
    values = np.concatenate(
        [switched["initial_gain"].to_numpy(), switched["final_gain"].to_numpy()]
    )
    if len(values):
        lower = float(np.min(values)) - 0.02
        upper = float(np.max(values)) + 0.02
    else:
        lower, upper = 0.0, 1.0
    axis.plot([lower, upper], [lower, upper], color="#555555", ls="--", lw=1.0)
    higher = int((switched["gain_change"] > 0).sum())
    lower_count = int((switched["gain_change"] < 0).sum())
    axis.text(
        0.04,
        0.94,
        f"{higher} higher; {lower_count} lower",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    axis.set(
        xlabel="Gain of first predicted class",
        ylabel="Gain of settled class",
        xlim=(lower, upper),
        ylim=(lower, upper),
        aspect="equal",
    )
    if len(switched):
        axis.legend(frameon=False, loc="lower right")

    axes[0].text(-0.14, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.14, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output, "mnist_static_training_imprint")


def intervention_figure(time_frame: pd.DataFrame, output: Path) -> None:
    conditions = ("distinct_50", "gain_balanced", "gain_reversed")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), sharex=True)
    for condition in conditions:
        plot_time_curve(
            axes[0],
            time_frame,
            condition=condition,
            metric="accuracy",
            max_readout=20,
        )
        plot_time_curve(
            axes[1],
            time_frame,
            condition=condition,
            metric="changed_from_initial",
            max_readout=20,
        )
    axes[0].set(
        xlabel="Internal update $r$",
        ylabel="Classification accuracy",
        xlim=(1, 20),
        ylim=(0.28, 0.42),
    )
    axes[1].set(
        xlabel="Internal update $r$",
        ylabel="Prediction differs from $r=1$",
        xlim=(1, 20),
        ylim=(0, 0.12),
    )
    axes[0].text(-0.14, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.14, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save_figure(fig, output, "mnist_static_gain_intervention")


def point_interval_plot(
    axis: plt.Axes,
    values_by_condition: dict[str, np.ndarray],
    conditions: tuple[str, ...],
    ylabel: str,
) -> None:
    rng = np.random.default_rng(19)
    all_values: list[float] = []
    for position, condition in enumerate(conditions):
        values = np.asarray(values_by_condition[condition], dtype=float)
        all_values.extend(values.tolist())
        jitter = rng.uniform(-0.08, 0.08, len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=15,
            color=COLOURS[condition],
            alpha=0.45,
            edgecolor="none",
        )
        mean, low, high = mean_interval(values)
        axis.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            ms=5.5,
            color=COLOURS[condition],
            capsize=3,
            lw=1.2,
        )
    tick_labels = {
        "repeat_1x50": "1 image\nx50",
        "blocked_10x5": "10 images\n5-step blocks",
        "interleaved_10x5": "10 images\ncycled",
        "distinct_50": "50 images\nonce",
    }
    axis.set_xticks(
        range(len(conditions)),
        [tick_labels[condition] for condition in conditions],
    )
    axis.tick_params(axis="x", labelsize=8)
    axis.set_ylabel(ylabel)
    upper = min(1.0, max(0.25, max(all_values, default=0.0) * 1.25 + 0.03))
    axis.set_ylim(0, upper)


def schedule_figure(
    class_frame: pd.DataFrame,
    trajectory_frame: pd.DataFrame,
    output: Path,
) -> None:
    conditions = (
        "repeat_1x50",
        "blocked_10x5",
        "interleaved_10x5",
        "distinct_50",
    )
    network_gain = (
        class_frame[class_frame["condition"].isin(conditions)]
        .groupby(["condition", "seed"], as_index=False)["within_class_gain"]
        .mean()
    )
    switch_rate = (
        trajectory_frame[trajectory_frame["condition"].isin(conditions)]
        .assign(switched=lambda frame: frame["switch_count"] > 0)
        .groupby(["condition", "seed"], as_index=False)["switched"]
        .mean()
    )
    gains = {
        condition: network_gain.loc[
            network_gain["condition"] == condition, "within_class_gain"
        ].to_numpy()
        for condition in conditions
    }
    switches = {
        condition: switch_rate.loc[
            switch_rate["condition"] == condition, "switched"
        ].to_numpy()
        for condition in conditions
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    point_interval_plot(
        axes[0],
        gains,
        conditions,
        "Mean recurrent gain",
    )
    point_interval_plot(
        axes[1],
        switches,
        conditions,
        "Trajectories that change class",
    )
    axes[0].text(-0.14, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.14, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output, "mnist_static_training_schedule")


def seed_condition_summary(
    trajectory_frame: pd.DataFrame,
) -> pd.DataFrame:
    return (
        trajectory_frame.assign(switched=lambda frame: frame["switch_count"] > 0)
        .groupby(["condition", "seed"], as_index=False)
        .agg(
            initial_accuracy=("initial_correct", "mean"),
            final_accuracy=("final_correct", "mean"),
            switch_rate=("switched", "mean"),
            maximum_settling_readout=("settling_readout", "max"),
        )
        .assign(
            accuracy_change=lambda frame: frame["final_accuracy"]
            - frame["initial_accuracy"]
        )
    )


def figure_statistics(
    class_frame: pd.DataFrame,
    trajectory_frame: pd.DataFrame,
) -> dict[str, object]:
    seed_summary = seed_condition_summary(trajectory_frame)
    baseline_classes = class_frame[class_frame["condition"] == "distinct_50"]
    digit_means = baseline_classes.groupby("digit", as_index=False).mean(
        numeric_only=True
    )
    switched = trajectory_frame[
        (trajectory_frame["condition"] == "distinct_50")
        & (trajectory_frame["switch_count"] > 0)
    ]

    condition_stats: dict[str, object] = {}
    for condition, group in seed_summary.groupby("condition"):
        condition_stats[str(condition)] = {
            metric: {
                "mean": mean_interval(group[metric])[0],
                "lower_95": mean_interval(group[metric])[1],
                "upper_95": mean_interval(group[metric])[2],
            }
            for metric in (
                "initial_accuracy",
                "final_accuracy",
                "accuracy_change",
                "switch_rate",
            )
        }
        condition_stats[str(condition)]["maximum_settling_readout"] = int(
            group["maximum_settling_readout"].max()
        )

    paired: dict[str, object] = {}
    baseline = seed_summary[seed_summary["condition"] == "distinct_50"].set_index(
        "seed"
    )
    for condition in ("gain_balanced", "gain_reversed"):
        comparison = seed_summary[
            seed_summary["condition"] == condition
        ].set_index("seed")
        common = baseline.index.intersection(comparison.index)
        paired[condition] = {}
        for metric in ("final_accuracy", "switch_rate"):
            difference = (
                comparison.loc[common, metric] - baseline.loc[common, metric]
            )
            mean, low, high = mean_interval(difference)
            paired[condition][f"{metric}_difference"] = {
                "mean": mean,
                "lower_95": low,
                "upper_95": high,
            }

    return {
        "condition_seed_statistics": condition_stats,
        "baseline_class_correlations": {
            "sensory_pair_overlap_vs_gain_pearson": float(
                pearsonr(
                    digit_means["sensory_pair_overlap"],
                    digit_means["within_class_gain"],
                ).statistic
            ),
            "sensory_pair_overlap_vs_gain_spearman": float(
                spearmanr(
                    digit_means["sensory_pair_overlap"],
                    digit_means["within_class_gain"],
                ).statistic
            ),
            "predicted_vs_measured_gain_pearson": float(
                pearsonr(
                    digit_means["predicted_within_gain"],
                    digit_means["within_class_gain"],
                ).statistic
            ),
        },
        "baseline_switch_directions": {
            "total": int(len(switched)),
            "higher_gain": int((switched["gain_change"] > 0).sum()),
            "lower_gain": int((switched["gain_change"] < 0).sum()),
            "equal_gain": int((switched["gain_change"] == 0).sum()),
        },
        "paired_intervention_differences": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    time_frame = pd.read_csv(args.results / "mnist_static_time_series.csv")
    trajectory_frame = pd.read_csv(
        args.results / "mnist_static_trajectory_summary.csv"
    )
    class_frame = pd.read_csv(args.results / "mnist_static_class_metrics.csv")

    exposure_figure(time_frame, args.output)
    imprint_figure(class_frame, trajectory_frame, args.output)
    intervention_figure(time_frame, args.output)
    schedule_figure(class_frame, trajectory_frame, args.output)
    statistics = figure_statistics(class_frame, trajectory_frame)
    (args.output / "mnist_static_figure_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
