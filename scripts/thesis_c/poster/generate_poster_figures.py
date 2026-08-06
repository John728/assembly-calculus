from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import PercentFormatter
from scipy.stats import t as student_t


TEAL = "#008F7A"
CORAL = "#E45756"
INDIGO = "#3D4FA3"
CHARCOAL = "#202A33"
MID_GREY = "#64717D"
GRID = "#DDE4E8"
PALE = "#F4F7F8"
WHITE = "#FFFFFF"

STATIC_CONDITIONS = {
    "distinct_50": ("Original learned gains", TEAL),
    "gain_balanced": ("Balanced gains", INDIGO),
    "gain_reversed": ("Reversed gains", CORAL),
}

# This is the seed-43 transition table used to produce dfa_trace.json.
DFA_TRANSITIONS = {
    (0, 0): 2,
    (0, 1): 2,
    (1, 0): 4,
    (1, 1): 1,
    (2, 0): 4,
    (2, 1): 3,
    (3, 0): 1,
    (3, 1): 1,
    (4, 0): 1,
    (4, 1): 2,
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
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
            "savefig.dpi": 400,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.9,
            "grid.linewidth": 0.8,
        }
    )


def mean_interval(values: pd.Series) -> tuple[float, float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, mean, mean
    half = (
        float(student_t.ppf(0.975, len(data) - 1))
        * float(np.std(data, ddof=1))
        / np.sqrt(len(data))
    )
    return mean, mean - half, mean + half


def save_square(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(6.4, 6.4, forward=True)
    for extension in ("pdf", "svg"):
        fig.savefig(
            output / f"{stem}.{extension}",
            facecolor=WHITE,
            metadata={"Title": stem, "Creator": "Matplotlib"},
        )
    fig.savefig(output / f"{stem}.png", dpi=400, facecolor=WHITE)
    plt.close(fig)


def plot_static(time_series: Path, output: Path) -> dict[str, object]:
    frame = pd.read_csv(time_series)
    frame = frame[
        frame["condition"].isin(STATIC_CONDITIONS)
        & (frame["readout_r"] <= 20)
    ]

    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.85)

    summaries: dict[str, dict[str, float]] = {}
    for condition, (label, colour) in STATIC_CONDITIONS.items():
        subset = frame[frame["condition"] == condition]
        readouts = np.sort(subset["readout_r"].unique())
        means: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for readout in readouts:
            mean, low, high = mean_interval(
                subset.loc[subset["readout_r"] == readout, "accuracy"]
            )
            means.append(mean)
            lower.append(low)
            upper.append(high)

        axis.fill_between(
            readouts,
            lower,
            upper,
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
            label=label,
        )
        axis.scatter(
            [readouts[0], readouts[-1]],
            [means[0], means[-1]],
            s=42,
            color=colour,
            edgecolor=WHITE,
            linewidth=1.1,
            zorder=4,
        )
        summaries[condition] = {
            "initial_accuracy": means[0],
            "readout_20_accuracy": means[-1],
        }
        axis.annotate(
            f"{means[-1]:.1%}",
            (readouts[-1], means[-1]),
            xytext=(-8, 9 if condition != "gain_reversed" else -15),
            textcoords="offset points",
            ha="right",
            color=colour,
            fontsize=11,
            fontweight="semibold",
        )

    initial_values = [
        summaries[condition]["initial_accuracy"] for condition in STATIC_CONDITIONS
    ]
    if not np.allclose(initial_values, initial_values[0]):
        raise ValueError("gain interventions do not share the first prediction")

    axis.annotate(
        f"Same first readout: {initial_values[0]:.1%}",
        xy=(1, initial_values[0]),
        xytext=(3.1, 0.405),
        arrowprops={
            "arrowstyle": "-|>",
            "color": MID_GREY,
            "lw": 1.4,
            "connectionstyle": "arc3,rad=0.12",
        },
        color=MID_GREY,
        fontsize=10.5,
    )
    axis.set(
        xlim=(1, 20.6),
        ylim=(0.28, 0.42),
        xlabel="Internal update",
        ylabel="Classification accuracy",
        xticks=(1, 5, 10, 15, 20),
        yticks=(0.28, 0.32, 0.36, 0.40, 0.42),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.025),
        frameon=False,
        ncol=1,
        handlelength=2.5,
        borderaxespad=0,
    )
    axis.text(
        0.025,
        0.035,
        "Mean and 95% t-interval, 10 networks",
        transform=axis.transAxes,
        color=MID_GREY,
        fontsize=8.8,
    )

    save_square(fig, output, "poster_static_recurrent_gain")
    return {
        "source": str(time_series),
        "conditions": summaries,
        "networks": int(frame["seed"].nunique()),
        "maximum_readout_shown": 20,
    }


def plot_static_margin_revision(
    trajectory_summary: Path,
    output: Path,
) -> dict[str, object]:
    frame = pd.read_csv(trajectory_summary)
    required_columns = {
        "seed",
        "condition",
        "instance_id",
        "transition_type",
        "switch_count",
        "initial_overlap",
        "initial_strongest_rival_overlap",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "trajectory summary is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame[frame["condition"].eq("distinct_50")].copy()
    if frame.empty:
        raise ValueError("trajectory summary has no distinct_50 results")

    frame["initial_margin"] = (
        frame["initial_overlap"]
        - frame["initial_strongest_rival_overlap"]
    )
    if frame["initial_margin"].min() < -1e-9:
        raise ValueError("initial top-two overlap gap cannot be negative")

    bin_edges = np.asarray((-1e-12, 0.025, 0.05, 0.08, 0.12, np.inf))
    bin_labels = ("0-2.5%", "2.5-5%", "5-8%", "8-12%", "12%+")
    frame["margin_bin"] = pd.cut(
        frame["initial_margin"],
        bins=bin_edges,
        labels=False,
        right=False,
    )
    if frame["margin_bin"].isna().any():
        raise ValueError("an initial overlap gap falls outside the plot bins")
    frame["margin_bin"] = frame["margin_bin"].astype(int)
    frame["corrected"] = frame["transition_type"].eq("corrected")
    frame["corrupted"] = frame["transition_type"].eq("corrupted")

    per_seed = (
        frame.groupby(["seed", "margin_bin"], as_index=False)
        .agg(
            trajectories=("instance_id", "size"),
            corrections=("corrected", "sum"),
            corruptions=("corrupted", "sum"),
        )
        .sort_values(["seed", "margin_bin"])
    )
    expected_rows = frame["seed"].nunique() * len(bin_labels)
    if (
        len(per_seed) != expected_rows
        or per_seed["trajectories"].eq(0).any()
    ):
        raise ValueError("every network must contribute to every margin bin")
    per_seed["correction_rate"] = (
        per_seed["corrections"] / per_seed["trajectories"]
    )
    per_seed["corruption_rate"] = (
        per_seed["corruptions"] / per_seed["trajectories"]
    )

    correction_means: list[float] = []
    correction_lower: list[float] = []
    correction_upper: list[float] = []
    corruption_means: list[float] = []
    corruption_lower: list[float] = []
    corruption_upper: list[float] = []
    for margin_bin in range(len(bin_labels)):
        subset = per_seed[per_seed["margin_bin"].eq(margin_bin)]
        mean, lower, upper = mean_interval(subset["correction_rate"])
        correction_means.append(mean)
        correction_lower.append(max(0.0, lower))
        correction_upper.append(min(1.0, upper))
        mean, lower, upper = mean_interval(subset["corruption_rate"])
        corruption_means.append(mean)
        corruption_lower.append(max(0.0, lower))
        corruption_upper.append(min(1.0, upper))

    correction_means_array = np.asarray(correction_means)
    corruption_means_array = np.asarray(corruption_means)
    correction_error = np.vstack(
        (
            correction_means_array - np.asarray(correction_lower),
            np.asarray(correction_upper) - correction_means_array,
        )
    )
    corruption_error = np.vstack(
        (
            np.asarray(corruption_upper) - corruption_means_array,
            corruption_means_array - np.asarray(corruption_lower),
        )
    )

    confident = frame["initial_margin"].ge(0.12)
    if frame.loc[confident, "switch_count"].gt(0).any():
        raise ValueError("a trajectory with at least 12% initial margin revised")

    x = np.arange(len(bin_labels))
    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.82)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    axis.axhline(0, color=CHARCOAL, linewidth=1.1, zorder=2)
    axis.bar(
        x,
        correction_means_array,
        width=0.64,
        color=TEAL,
        edgecolor=WHITE,
        linewidth=0.9,
        yerr=correction_error,
        error_kw={
            "ecolor": CHARCOAL,
            "elinewidth": 1.2,
            "capsize": 3.5,
            "capthick": 1.2,
        },
        label="Initial error corrected",
        zorder=3,
    )
    axis.bar(
        x,
        -corruption_means_array,
        width=0.64,
        color=CORAL,
        edgecolor=WHITE,
        linewidth=0.9,
        yerr=corruption_error,
        error_kw={
            "ecolor": CHARCOAL,
            "elinewidth": 1.2,
            "capsize": 3.5,
            "capthick": 1.2,
        },
        label="Correct prediction corrupted",
        zorder=3,
    )
    axis.text(
        x[-1],
        0.012,
        "No revisions",
        ha="center",
        va="bottom",
        color=MID_GREY,
        fontsize=9.5,
        fontweight="semibold",
    )
    axis.set(
        xlim=(-0.62, len(bin_labels) - 0.38),
        ylim=(-0.055, 0.32),
        xlabel="Initial top-two assembly-overlap gap",
        ylabel="Share of trajectories in each bin",
        xticks=x,
        xticklabels=bin_labels,
        yticks=(-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.025),
        frameon=False,
        ncol=1,
        handlelength=1.5,
        borderaxespad=0,
    )
    axis.text(
        0.015,
        0.025,
        "Mean and 95% t-interval, 10 networks",
        transform=axis.transAxes,
        color=MID_GREY,
        fontsize=8.8,
    )

    pooled = (
        frame.groupby("margin_bin", as_index=False)
        .agg(
            trajectories=("instance_id", "size"),
            corrections=("corrected", "sum"),
            corruptions=("corrupted", "sum"),
        )
        .sort_values("margin_bin")
    )
    pooled["correction_rate"] = (
        pooled["corrections"] / pooled["trajectories"]
    )
    pooled["corruption_rate"] = (
        pooled["corruptions"] / pooled["trajectories"]
    )
    bins = []
    for label, row in zip(bin_labels, pooled.itertuples()):
        bins.append(
            {
                "label": label,
                "trajectories": int(row.trajectories),
                "corrections": int(row.corrections),
                "corruptions": int(row.corruptions),
                "pooled_correction_rate": float(row.correction_rate),
                "pooled_corruption_rate": float(row.corruption_rate),
            }
        )

    save_square(fig, output, "poster_static_margin_revision")
    revised = frame[frame["switch_count"].gt(0)]
    return {
        "source": str(trajectory_summary),
        "condition": "distinct_50",
        "networks": int(frame["seed"].nunique()),
        "trajectories": int(len(frame)),
        "bin_edges": [0.0, 0.025, 0.05, 0.08, 0.12, "infinity"],
        "bins": bins,
        "total_corrections": int(frame["corrected"].sum()),
        "total_corruptions": int(frame["corrupted"].sum()),
        "maximum_margin_with_any_revision": float(
            revised["initial_margin"].max()
        ),
        "no_revisions_at_or_above_margin": 0.12,
    }


def plot_static_overlap_and_correction(
    representative_trace: Path,
    trajectory_summary: Path,
    output: Path,
) -> dict[str, object]:
    trace_row = json.loads(representative_trace.read_text(encoding="utf-8"))
    trace = np.asarray(trace_row["overlap_trajectory"], dtype=float)
    target = int(trace_row["target"])
    cue_end = int(trace_row["cue_duration_s"])
    rivals = [index for index in range(trace.shape[1]) if index != target]
    rival = rivals[int(np.argmax(trace[cue_end, rivals]))]
    updates = np.arange(trace.shape[0])

    frame = pd.read_csv(trajectory_summary)
    required_columns = {
        "seed",
        "condition",
        "instance_id",
        "initial_correct",
        "transition_type",
        "initial_overlap",
        "initial_strongest_rival_overlap",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "trajectory summary is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    frame = frame[frame["condition"].eq("distinct_50")].copy()
    if frame.empty:
        raise ValueError("trajectory summary has no distinct_50 results")

    frame["initial_margin"] = (
        frame["initial_overlap"]
        - frame["initial_strongest_rival_overlap"]
    )
    if frame["initial_margin"].min() < -1e-9:
        raise ValueError("initial top-two overlap gap cannot be negative")

    bin_edges = np.asarray((-1e-12, 0.025, 0.05, 0.08, 0.12, np.inf))
    bin_labels = ("0-2.5%", "2.5-5%", "5-8%", "8-12%", "12%+")
    frame["margin_bin"] = pd.cut(
        frame["initial_margin"],
        bins=bin_edges,
        labels=False,
        right=False,
    )
    if frame["margin_bin"].isna().any():
        raise ValueError("an initial overlap gap falls outside the plot bins")
    frame["margin_bin"] = frame["margin_bin"].astype(int)

    initial_errors = frame[~frame["initial_correct"].astype(bool)].copy()
    initial_errors["corrected"] = initial_errors["transition_type"].eq(
        "corrected"
    )
    correction_summary = (
        initial_errors.groupby("margin_bin", as_index=False)
        .agg(
            initial_errors=("instance_id", "size"),
            corrections=("corrected", "sum"),
        )
        .sort_values("margin_bin")
    )
    if len(correction_summary) != len(bin_labels):
        raise ValueError("every margin bin must contain an initial error")
    correction_summary["correction_rate"] = (
        correction_summary["corrections"]
        / correction_summary["initial_errors"]
    )

    initial_correct = frame[frame["initial_correct"].astype(bool)]
    corruptions = int(
        initial_correct["transition_type"].eq("corrupted").sum()
    )
    if corruptions != 1:
        raise ValueError("expected one corrupted first prediction")

    fig = plt.figure()
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.98, 1.02),
        left=0.14,
        right=0.97,
        bottom=0.12,
        top=0.91,
        hspace=0.43,
    )

    overlap_axis = fig.add_subplot(grid[0])
    overlap_axis.grid(axis="x", visible=False)
    for digit in range(trace.shape[1]):
        if digit in (target, rival):
            continue
        overlap_axis.plot(
            updates,
            trace[:, digit],
            color=GRID,
            linewidth=0.8,
            alpha=0.8,
            zorder=1,
        )
    overlap_axis.plot(
        updates,
        trace[:, target],
        color=TEAL,
        linewidth=2.8,
        solid_capstyle="round",
        label=f"True assembly ({target})",
        zorder=4,
    )
    overlap_axis.plot(
        updates,
        trace[:, rival],
        color=CORAL,
        linewidth=2.2,
        linestyle="--",
        solid_capstyle="round",
        label=f"Strongest rival ({rival})",
        zorder=3,
    )
    overlap_axis.axvline(
        cue_end,
        color=CHARCOAL,
        linewidth=1.2,
        linestyle=":",
        label="Cue removed",
        zorder=2,
    )
    overlap_axis.set(
        xlim=(-1, int(updates[-1]) + 1),
        ylim=(-0.03, 1.04),
        xlabel="Internal update",
        ylabel="Assembly overlap",
        xticks=(0, 10, 20, 30, 40, 50),
        yticks=(0.0, 0.5, 1.0),
    )
    overlap_axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=3,
        handlelength=2.3,
        columnspacing=1.3,
        borderaxespad=0,
        fontsize=9.2,
    )

    correction_axis = fig.add_subplot(grid[1])
    correction_axis.grid(axis="x", visible=False)
    correction_axis.set_axisbelow(True)
    x = np.arange(len(bin_labels))
    correction_rates = correction_summary["correction_rate"].to_numpy(
        dtype=float
    )
    bars = correction_axis.bar(
        x,
        correction_rates,
        width=0.64,
        color=TEAL,
        edgecolor=WHITE,
        linewidth=0.9,
        zorder=3,
    )
    for bar, rate in zip(bars, correction_rates):
        correction_axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(rate + 0.009, 0.009),
            f"{rate:.0%}",
            ha="center",
            va="bottom",
            color=CHARCOAL,
            fontsize=10,
            fontweight="semibold",
        )
    correction_axis.set(
        xlim=(-0.62, len(bin_labels) - 0.38),
        ylim=(0.0, 0.31),
        xlabel="Initial confidence (top-two overlap gap)",
        ylabel="Initial errors corrected",
        xticks=x,
        xticklabels=bin_labels,
        yticks=(0.0, 0.10, 0.20, 0.30),
    )
    correction_axis.yaxis.set_major_formatter(
        PercentFormatter(1.0, decimals=0)
    )
    correction_axis.text(
        0.98,
        0.91,
        (
            f"Only {corruptions} of {len(initial_correct):,} correct first "
            "choices became wrong"
        ),
        transform=correction_axis.transAxes,
        ha="right",
        va="top",
        color=CORAL,
        fontsize=9.2,
        fontweight="semibold",
    )
    correction_axis.text(
        0.98,
        0.81,
        (
            f"{len(initial_errors):,} initial errors across "
            f"{frame['seed'].nunique()} networks"
        ),
        transform=correction_axis.transAxes,
        ha="right",
        va="top",
        color=MID_GREY,
        fontsize=8.8,
    )

    save_square(fig, output, "poster_static_overlap_and_correction")
    bins = []
    for label, row in zip(bin_labels, correction_summary.itertuples()):
        bins.append(
            {
                "label": label,
                "initial_errors": int(row.initial_errors),
                "corrections": int(row.corrections),
                "correction_rate": float(row.correction_rate),
            }
        )
    return {
        "representative_trace_source": str(representative_trace),
        "trajectory_summary_source": str(trajectory_summary),
        "condition": "distinct_50",
        "networks": int(frame["seed"].nunique()),
        "trajectories": int(len(frame)),
        "initial_errors": int(len(initial_errors)),
        "initially_correct": int(len(initial_correct)),
        "corruptions": corruptions,
        "bins": bins,
        "representative_target": target,
        "representative_rival": rival,
        "cue_removed_at_update": cue_end,
    }


def plot_static_overlap_and_accuracy(
    representative_traces: Path,
    time_series: Path,
    output: Path,
) -> dict[str, object]:
    trace_rows = json.loads(representative_traces.read_text(encoding="utf-8"))
    corrected_rows = [
        row
        for row in trace_rows
        if row["condition"] == "distinct_50"
        and row["transition_type"] == "corrected"
    ]
    if len(corrected_rows) != 1:
        raise ValueError("expected one representative corrected trajectory")
    trace_row = corrected_rows[0]
    trace = np.asarray(trace_row["overlaps"], dtype=float)
    predictions = np.asarray(trace_row["predictions"], dtype=int)
    target = int(trace_row["target"])
    initial_winner = int(trace_row["initial_prediction"])
    if initial_winner == target or int(trace_row["final_prediction"]) != target:
        raise ValueError("representative trajectory is not a correction")

    correction_indices = np.flatnonzero(predictions == target)
    if not len(correction_indices):
        raise ValueError("representative trajectory never becomes correct")
    correction_update = int(correction_indices[0] + 1)
    if not np.all(predictions[correction_update - 1 :] == target):
        raise ValueError("representative correction is not retained")

    horizon = 20
    if trace.shape[0] < horizon:
        raise ValueError("representative trajectory is shorter than 20 updates")
    updates = np.arange(1, horizon + 1)

    frame = pd.read_csv(time_series)
    required_columns = {"seed", "condition", "readout_r", "accuracy"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "time series is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    frame = frame[
        frame["condition"].eq("distinct_50")
        & frame["readout_r"].between(1, horizon)
    ].copy()
    if frame.empty:
        raise ValueError("time series has no distinct_50 results")

    readouts = np.sort(frame["readout_r"].unique())
    if not np.array_equal(readouts, updates):
        raise ValueError("time series does not contain updates 1 through 20")
    networks_per_readout = frame.groupby("readout_r")["seed"].nunique()
    if networks_per_readout.nunique() != 1:
        raise ValueError("readouts do not contain the same network seeds")

    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for readout in readouts:
        mean, low, high = mean_interval(
            frame.loc[frame["readout_r"].eq(readout), "accuracy"]
        )
        means.append(mean)
        lower.append(max(0.0, low))
        upper.append(min(1.0, high))
    means_array = np.asarray(means)
    final_accuracy = float(means_array[-1])
    plateau_candidates = [
        index
        for index in range(len(means_array))
        if np.allclose(means_array[index:], final_accuracy)
    ]
    if not plateau_candidates:
        raise ValueError("accuracy has no observed plateau")
    plateau_index = plateau_candidates[0]
    plateau_update = int(readouts[plateau_index])

    fig = plt.figure()
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.0),
        left=0.14,
        right=0.97,
        bottom=0.12,
        top=0.91,
        hspace=0.38,
    )

    overlap_axis = fig.add_subplot(grid[0])
    overlap_axis.grid(axis="x", visible=False)
    for digit in range(trace.shape[1]):
        if digit in (target, initial_winner):
            continue
        overlap_axis.plot(
            updates,
            trace[:horizon, digit],
            color=GRID,
            linewidth=0.8,
            alpha=0.8,
            zorder=1,
        )
    overlap_axis.plot(
        updates,
        trace[:horizon, target],
        color=TEAL,
        linewidth=2.8,
        solid_capstyle="round",
        label=f"True assembly ({target})",
        zorder=4,
    )
    overlap_axis.plot(
        updates,
        trace[:horizon, initial_winner],
        color=CORAL,
        linewidth=2.2,
        linestyle="--",
        solid_capstyle="round",
        label=f"Initial winner ({initial_winner})",
        zorder=3,
    )
    overlap_axis.axvline(
        correction_update,
        color=CHARCOAL,
        linewidth=1.2,
        linestyle=":",
        label=f"Correct from update {correction_update}",
        zorder=2,
    )
    overlap_axis.set(
        xlim=(0.7, horizon + 0.3),
        ylim=(-0.03, 1.04),
        ylabel="Assembly overlap",
        xticks=(1, 5, 10, 15, 20),
        yticks=(0.0, 0.5, 1.0),
    )
    overlap_axis.tick_params(axis="x", labelbottom=False)
    overlap_axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=3,
        handlelength=2.3,
        columnspacing=1.2,
        borderaxespad=0,
        fontsize=9.2,
    )

    accuracy_axis = fig.add_subplot(grid[1], sharex=overlap_axis)
    accuracy_axis.grid(axis="x", visible=False)
    accuracy_axis.set_axisbelow(True)
    accuracy_axis.fill_between(
        readouts,
        lower,
        upper,
        color=TEAL,
        alpha=0.14,
        linewidth=0,
    )
    accuracy_axis.plot(
        readouts,
        means_array,
        color=TEAL,
        linewidth=2.8,
        marker="o",
        markersize=4.5,
        markeredgecolor=WHITE,
        markeredgewidth=0.8,
        solid_capstyle="round",
        zorder=3,
    )
    accuracy_axis.axvline(
        plateau_update,
        color=MID_GREY,
        linewidth=1.1,
        linestyle=":",
        zorder=2,
    )
    accuracy_axis.scatter(
        [readouts[0], plateau_update],
        [means_array[0], means_array[plateau_index]],
        s=52,
        color=TEAL,
        edgecolor=WHITE,
        linewidth=1.0,
        zorder=4,
    )
    accuracy_axis.annotate(
        f"{means_array[0]:.1%}",
        (readouts[0], means_array[0]),
        xytext=(8, -14),
        textcoords="offset points",
        color=CHARCOAL,
        fontsize=10,
        fontweight="semibold",
    )
    accuracy_axis.annotate(
        f"{means_array[plateau_index]:.1%}",
        (plateau_update, means_array[plateau_index]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color=CHARCOAL,
        fontsize=10,
        fontweight="semibold",
    )
    accuracy_axis.text(
        13.2,
        0.399,
        f"No further change after update {plateau_update}",
        ha="center",
        va="top",
        color=MID_GREY,
        fontsize=9.2,
        fontweight="semibold",
    )
    accuracy_axis.text(
        0.98,
        0.06,
        (
            "Mean and 95% t-interval across "
            f"{int(networks_per_readout.iloc[0])} networks"
        ),
        transform=accuracy_axis.transAxes,
        ha="right",
        color=MID_GREY,
        fontsize=8.8,
    )
    accuracy_axis.set(
        xlim=(0.7, horizon + 0.3),
        ylim=(0.325, 0.405),
        xlabel="Internal update",
        ylabel="Classification accuracy",
        xticks=(1, 5, 10, 15, 20),
        yticks=(0.33, 0.35, 0.37, 0.39, 0.40),
    )
    accuracy_axis.yaxis.set_major_formatter(
        PercentFormatter(1.0, decimals=0)
    )

    save_square(fig, output, "poster_static_overlap_and_accuracy")
    return {
        "representative_trace_source": str(representative_traces),
        "time_series_source": str(time_series),
        "condition": "distinct_50",
        "networks": int(networks_per_readout.iloc[0]),
        "maximum_update": horizon,
        "initial_accuracy": float(means_array[0]),
        "plateau_accuracy": final_accuracy,
        "accuracy_gain_percentage_points": float(
            100.0 * (final_accuracy - means_array[0])
        ),
        "plateau_update": plateau_update,
        "representative_target": target,
        "representative_initial_winner": initial_winner,
        "representative_correction_update": correction_update,
    }


def plot_static_overlap_summary(
    overlap_time_series: Path,
    output: Path,
    *,
    band_mode: str = "confidence",
    stem: str = "poster_static_overlap_trajectory",
) -> dict[str, object]:
    if band_mode not in {"confidence", "range"}:
        raise ValueError(f"unknown overlap band mode: {band_mode}")
    frame = pd.read_csv(overlap_time_series)
    required_columns = {
        "seed",
        "condition",
        "cohort",
        "readout_r",
        "examples",
        "correct_class_overlap",
        "strongest_rival_overlap",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "overlap time series is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    maximum_update = 15
    frame = frame[
        frame["condition"].eq("distinct_50")
        & frame["cohort"].eq("final_correct")
        & frame["readout_r"].between(1, maximum_update)
    ].copy()
    if frame.empty:
        raise ValueError("overlap time series has no final-correct cohort")

    readouts = np.sort(frame["readout_r"].unique())
    expected_readouts = np.arange(1, maximum_update + 1)
    if not np.array_equal(readouts, expected_readouts):
        raise ValueError("overlap time series does not contain updates 1 to 15")
    networks_per_readout = frame.groupby("readout_r")["seed"].nunique()
    if networks_per_readout.nunique() != 1:
        raise ValueError("readouts do not contain the same network seeds")

    series = (
        (
            "correct_class_overlap",
            "Correct-class assembly",
            TEAL,
        ),
        (
            "strongest_rival_overlap",
            "Strongest rival",
            CORAL,
        ),
    )
    summaries: dict[str, dict[str, float]] = {}
    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.80)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)

    for column, label, colour in series:
        means: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for readout in readouts:
            values = frame.loc[frame["readout_r"].eq(readout), column]
            mean = float(values.mean())
            if band_mode == "confidence":
                _, low, high = mean_interval(values)
            else:
                low = float(values.min())
                high = float(values.max())
            means.append(mean)
            lower.append(max(0.0, low))
            upper.append(min(1.0, high))
        means_array = np.asarray(means)
        axis.fill_between(
            readouts,
            lower,
            upper,
            color=colour,
            alpha=0.24 if band_mode == "confidence" else 0.17,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            readouts,
            means_array,
            color=colour,
            linewidth=2.8,
            marker="o",
            markersize=6.2,
            markeredgecolor=WHITE,
            markeredgewidth=1.1,
            solid_capstyle="round",
            label=label,
            zorder=4,
        )
        summaries[column] = {
            "update_1_mean": float(means_array[0]),
            "update_15_mean": float(means_array[-1]),
        }

    trajectories = int(
        frame.loc[frame["readout_r"].eq(1), "examples"].sum()
    )
    networks = int(networks_per_readout.iloc[0])
    axis.set(
        xlim=(0.8, maximum_update + 0.2),
        ylim=(-0.025, 1.025),
        xlabel="Internal update",
        ylabel="Assembly overlap",
        xticks=(1, 5, 10, 15),
        yticks=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.025),
        frameon=False,
        ncol=2,
        handlelength=2.3,
        columnspacing=1.6,
        borderaxespad=0,
    )
    axis.xaxis.label.set_color("#000000")
    axis.yaxis.label.set_color("#000000")
    axis.tick_params(axis="both", colors="#000000")
    axis.spines["left"].set_color("#000000")
    axis.spines["bottom"].set_color("#000000")
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)

    save_square(fig, output, stem)
    uncertainty = (
        "95% t-interval across per-network means"
        if band_mode == "confidence"
        else "minimum to maximum of per-network means"
    )
    return {
        "source": str(overlap_time_series),
        "condition": "distinct_50",
        "cohort": "final_correct",
        "cohort_definition": (
            "fixed subset classified correctly at the final readout"
        ),
        "networks": networks,
        "trajectories": trajectories,
        "maximum_update": maximum_update,
        "series": summaries,
        "aggregation": "mean of per-network means",
        "uncertainty": uncertainty,
        "uncertainty_shown": True,
    }


def plot_static_correction_summary(
    overlap_time_series: Path,
    output: Path,
) -> dict[str, object]:
    frame = pd.read_csv(overlap_time_series)
    required_columns = {
        "seed",
        "condition",
        "cohort",
        "readout_r",
        "examples",
        "correct_class_overlap",
        "initial_winner_overlap",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "overlap time series is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    maximum_update = 10
    frame = frame[
        frame["condition"].eq("distinct_50")
        & frame["cohort"].eq("corrected")
        & frame["readout_r"].between(1, maximum_update)
    ].copy()
    if frame.empty:
        raise ValueError("overlap time series has no corrected cohort")

    readouts = np.sort(frame["readout_r"].unique())
    expected_readouts = np.arange(1, maximum_update + 1)
    if not np.array_equal(readouts, expected_readouts):
        raise ValueError("corrected cohort does not contain updates 1 to 10")
    networks_per_readout = frame.groupby("readout_r")["seed"].nunique()
    if networks_per_readout.nunique() != 1:
        raise ValueError("readouts do not contain the same network seeds")

    series = (
        (
            "correct_class_overlap",
            "Correct-class assembly",
            TEAL,
        ),
        (
            "initial_winner_overlap",
            "Initial winner",
            CORAL,
        ),
    )
    summaries: dict[str, dict[str, float]] = {}
    mean_series: dict[str, np.ndarray] = {}
    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.85)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)

    for column, label, colour in series:
        means: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for readout in readouts:
            values = frame.loc[frame["readout_r"].eq(readout), column]
            means.append(float(values.mean()))
            lower.append(max(0.0, float(values.min())))
            upper.append(min(1.0, float(values.max())))
        means_array = np.asarray(means)
        mean_series[column] = means_array
        axis.fill_between(
            readouts,
            lower,
            upper,
            color=colour,
            alpha=0.17,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            readouts,
            means_array,
            color=colour,
            linewidth=2.8,
            marker="o",
            markersize=6.2,
            markeredgecolor=WHITE,
            markeredgewidth=1.1,
            solid_capstyle="round",
            label=label,
            zorder=4,
        )
        summaries[column] = {
            "update_1_mean": float(means_array[0]),
            "update_2_mean": float(means_array[1]),
            "update_10_mean": float(means_array[-1]),
        }

    crossing_indices = np.flatnonzero(
        mean_series["correct_class_overlap"]
        >= mean_series["initial_winner_overlap"]
    )
    if len(crossing_indices) == 0:
        raise ValueError("mean corrected trajectory never crosses")
    crossing_update = int(readouts[crossing_indices[0]])

    trajectories = int(
        frame.loc[frame["readout_r"].eq(1), "examples"].sum()
    )
    networks = int(networks_per_readout.iloc[0])
    axis.set(
        xlim=(0.8, maximum_update + 0.2),
        ylim=(-0.025, 1.025),
        xlabel="Internal update",
        ylabel="Assembly overlap",
        xticks=(1, 2, 4, 6, 8, 10),
        yticks=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.025),
        frameon=False,
        ncol=2,
        handlelength=2.3,
        columnspacing=1.6,
        borderaxespad=0,
    )
    axis.xaxis.label.set_color("#000000")
    axis.yaxis.label.set_color("#000000")
    axis.tick_params(axis="both", colors="#000000")
    axis.spines["left"].set_color("#000000")
    axis.spines["bottom"].set_color("#000000")
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)

    save_square(fig, output, "poster_static_correction_trajectory")
    return {
        "source": str(overlap_time_series),
        "condition": "distinct_50",
        "cohort": "corrected",
        "cohort_definition": (
            "wrong at update 1 and correct at the final readout"
        ),
        "networks": networks,
        "trajectories": trajectories,
        "maximum_update": maximum_update,
        "crossing_update": crossing_update,
        "series": summaries,
        "aggregation": "mean of per-network means",
        "uncertainty": "minimum to maximum of per-network means",
        "uncertainty_shown": True,
    }


def plot_static_combined_summary(
    overlap_time_series: Path,
    output: Path,
) -> dict[str, object]:
    frame = pd.read_csv(overlap_time_series)
    required_columns = {
        "seed",
        "condition",
        "cohort",
        "readout_r",
        "examples",
        "correct_class_overlap",
        "strongest_rival_overlap",
        "initial_winner_overlap",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "overlap time series is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    maximum_update = 15
    panel_specs = (
        {
            "cohort": "final_correct",
            "title": "(a) Correct at final update",
            "series": (
                ("correct_class_overlap", "Correct class", TEAL),
                ("strongest_rival_overlap", "Strongest rival", CORAL),
            ),
        },
        {
            "cohort": "corrected",
            "title": "(b) Initially wrong, later corrected",
            "series": (
                ("correct_class_overlap", "Correct class", TEAL),
                ("initial_winner_overlap", "Initial winner", CORAL),
            ),
        },
    )

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.16,
        right=0.97,
        bottom=0.12,
        top=0.91,
        hspace=0.58,
    )
    summaries: dict[str, object] = {}
    expected_readouts = np.arange(1, maximum_update + 1)

    for axis, panel in zip(axes, panel_specs):
        cohort = str(panel["cohort"])
        subset = frame[
            frame["condition"].eq("distinct_50")
            & frame["cohort"].eq(cohort)
            & frame["readout_r"].between(1, maximum_update)
        ].copy()
        if subset.empty:
            raise ValueError(f"overlap time series has no {cohort} cohort")

        readouts = np.sort(subset["readout_r"].unique())
        if not np.array_equal(readouts, expected_readouts):
            raise ValueError(
                f"{cohort} cohort does not contain updates 1 to 15"
            )
        networks_per_readout = subset.groupby("readout_r")["seed"].nunique()
        if networks_per_readout.nunique() != 1:
            raise ValueError(
                f"{cohort} readouts do not contain the same network seeds"
            )

        panel_series: dict[str, dict[str, float]] = {}
        for column, label, colour in panel["series"]:
            means: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for readout in readouts:
                values = subset.loc[
                    subset["readout_r"].eq(readout),
                    column,
                ]
                means.append(float(values.mean()))
                lower.append(max(0.0, float(values.min())))
                upper.append(min(1.0, float(values.max())))
            means_array = np.asarray(means)
            axis.fill_between(
                readouts,
                lower,
                upper,
                color=colour,
                alpha=0.17,
                linewidth=0,
                zorder=2,
            )
            axis.plot(
                readouts,
                means_array,
                color=colour,
                linewidth=2.4,
                marker="o",
                markersize=4.8,
                markeredgecolor=WHITE,
                markeredgewidth=0.9,
                solid_capstyle="round",
                label=label,
                zorder=4,
            )
            panel_series[column] = {
                "update_1_mean": float(means_array[0]),
                "update_2_mean": float(means_array[1]),
                "update_15_mean": float(means_array[-1]),
            }

        trajectories = int(
            subset.loc[subset["readout_r"].eq(1), "examples"].sum()
        )
        title = f"{panel['title']} ($n={trajectories}$)"
        axis.text(
            0.0,
            1.17,
            title,
            transform=axis.transAxes,
            fontsize=10.8,
            fontweight="semibold",
            ha="left",
            va="bottom",
        )
        axis.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.015),
            frameon=False,
            ncol=2,
            fontsize=8.8,
            handlelength=2.0,
            columnspacing=1.25,
            borderaxespad=0,
        )
        axis.set(
            xlim=(0.8, maximum_update + 0.2),
            ylim=(-0.025, 1.025),
            yticks=(0.0, 0.5, 1.0),
        )
        axis.grid(axis="x", visible=False)
        axis.set_axisbelow(True)
        axis.xaxis.label.set_color("#000000")
        axis.yaxis.label.set_color("#000000")
        axis.tick_params(axis="both", colors="#000000")
        axis.spines["left"].set_color("#000000")
        axis.spines["bottom"].set_color("#000000")
        axis.spines["left"].set_linewidth(1.2)
        axis.spines["bottom"].set_linewidth(1.2)

        summaries[cohort] = {
            "networks": int(networks_per_readout.iloc[0]),
            "trajectories": trajectories,
            "series": panel_series,
        }

    axes[-1].set(
        xlabel="Internal update",
        xticks=(1, 5, 10, 15),
    )
    fig.supylabel("Assembly overlap", x=0.035, fontsize=13)
    save_square(fig, output, "poster_static_overlap_and_corrections")
    return {
        "source": str(overlap_time_series),
        "condition": "distinct_50",
        "maximum_update": maximum_update,
        "panels": summaries,
        "aggregation": "mean of per-network means",
        "uncertainty": "minimum to maximum of per-network means",
        "uncertainty_shown": True,
    }


def edge_label_position(
    start: np.ndarray,
    end: np.ndarray,
    curvature: float,
    along: float = 0.50,
) -> np.ndarray:
    midpoint = start + along * (end - start)
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return midpoint
    perpendicular = np.array([-direction[1], direction[0]]) / norm
    return midpoint + perpendicular * curvature * 0.42


def add_transition_label(
    axis: plt.Axes,
    position: np.ndarray,
    symbols: tuple[int, ...],
) -> None:
    text_areas = []
    for index, symbol in enumerate(symbols):
        if index:
            text_areas.append(
                TextArea(
                    ",",
                    textprops={"color": MID_GREY, "size": 9.5},
                )
            )
        text_areas.append(
            TextArea(
                str(symbol),
                textprops={
                    "color": CORAL if symbol == 0 else INDIGO,
                    "size": 9.5,
                    "weight": "bold",
                },
            )
        )
    label = HPacker(children=text_areas, align="center", pad=0, sep=1)
    annotation = AnnotationBbox(
        label,
        position,
        frameon=True,
        bboxprops={
            "boxstyle": "round,pad=0.20",
            "facecolor": WHITE,
            "edgecolor": GRID,
            "linewidth": 0.8,
        },
        zorder=3,
    )
    axis.add_artist(annotation)


def draw_transition(
    axis: plt.Axes,
    positions: dict[int, np.ndarray],
    source: int,
    target: int,
    symbols: tuple[int, ...],
    *,
    curvature: float,
    label_along: float = 0.50,
) -> None:
    if source == target:
        centre = positions[source]
        start = centre + np.array([-0.16, -0.17])
        end = centre + np.array([0.16, -0.17])
        patch = FancyArrowPatch(
            start,
            end,
            connectionstyle="arc3,rad=1.8",
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=CHARCOAL,
            zorder=1,
        )
        label_position = centre + np.array([0.0, -0.50])
    else:
        start = positions[source]
        end = positions[target]
        patch = FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={curvature}",
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=CHARCOAL,
            shrinkA=22,
            shrinkB=22,
            zorder=1,
        )
        label_position = edge_label_position(start, end, curvature, label_along)
    axis.add_patch(patch)
    add_transition_label(axis, label_position, symbols)


def plot_dfa(trace_file: Path, output: Path) -> dict[str, object]:
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    sequence = [int(value) for value in trace["sequence"]]
    true_states = [int(value) for value in trace["true_states"]]
    decoded_states = [int(value) for value in trace["decoded_states"]]
    overlaps = np.asarray(trace["overlaps"], dtype=float)

    if true_states != decoded_states:
        raise ValueError("the selected DFA trace is not exact")
    if overlaps.shape != (len(sequence) + 1, 5):
        raise ValueError("unexpected DFA overlap shape")
    current = true_states[0]
    reconstructed = [current]
    for symbol in sequence:
        current = DFA_TRANSITIONS[(current, symbol)]
        reconstructed.append(current)
    if reconstructed != true_states:
        raise ValueError("the state diagram does not generate the recorded trace")

    fig = plt.figure()
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.08, 0.92),
        left=0.10,
        right=0.94,
        bottom=0.12,
        top=0.94,
        hspace=0.30,
    )

    diagram = fig.add_subplot(grid[0])
    diagram.set_aspect("equal")
    diagram.axis("off")
    positions = {
        0: np.array([0.0, 1.10]),
        2: np.array([0.0, 0.35]),
        3: np.array([-1.05, -0.08]),
        4: np.array([1.05, -0.08]),
        1: np.array([0.0, -0.88]),
    }

    edge_specs = [
        (0, 2, (0, 1), 0.00, 0.50),
        (2, 3, (1,), 0.00, 0.52),
        (2, 4, (0,), 0.22, 0.50),
        (3, 1, (0, 1), 0.00, 0.48),
        (4, 2, (1,), 0.22, 0.50),
        (4, 1, (0,), 0.22, 0.50),
        (1, 4, (0,), 0.22, 0.50),
        (1, 1, (1,), 0.00, 0.50),
    ]
    for source, target, symbols, curvature, label_along in edge_specs:
        for symbol in symbols:
            if DFA_TRANSITIONS[(source, symbol)] != target:
                raise ValueError(
                    "edge specification disagrees with DFA transition table"
                )
        draw_transition(
            diagram,
            positions,
            source,
            target,
            symbols,
            curvature=curvature,
            label_along=label_along,
        )

    for state, centre in positions.items():
        node = Circle(
            centre,
            radius=0.21,
            facecolor=WHITE,
            edgecolor=TEAL if state == true_states[0] else CHARCOAL,
            linewidth=2.6 if state == true_states[0] else 1.8,
            zorder=2,
        )
        diagram.add_patch(node)
        diagram.text(
            centre[0],
            centre[1],
            rf"$q_{state}$",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="semibold",
            zorder=4,
        )
    diagram.set(xlim=(-1.65, 1.65), ylim=(-1.48, 1.42))
    diagram.set_title("One update executes one exact transition", pad=5)
    diagram.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CORAL,
                markeredgecolor=CORAL,
                markersize=6,
                label="Input 0",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=INDIGO,
                markeredgecolor=INDIGO,
                markersize=6,
                label="Input 1",
            ),
        ],
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1.04, 1.04),
    )

    trajectory = fig.add_subplot(grid[1])
    colourmap = LinearSegmentedColormap.from_list(
        "poster_overlap",
        [PALE, TEAL],
    )
    trajectory.imshow(
        overlaps.T,
        cmap=colourmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    trajectory.grid(False)
    trajectory.set(
        xlabel="Internal update / logical checkpoint",
        ylabel="State assembly",
        xticks=np.arange(overlaps.shape[0]),
        yticks=np.arange(5),
        yticklabels=[rf"$q_{state}$" for state in range(5)],
    )
    trajectory.set_xticks(
        np.arange(-0.5, overlaps.shape[0], 1),
        minor=True,
    )
    trajectory.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    trajectory.grid(which="minor", color=WHITE, linewidth=1.4)
    trajectory.tick_params(which="minor", bottom=False, left=False)
    for checkpoint, state in enumerate(true_states):
        trajectory.text(
            checkpoint,
            state,
            str(state),
            color=WHITE,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    input_axis = trajectory.secondary_xaxis("top")
    input_axis.set_xticks(np.arange(1, len(sequence) + 1))
    input_axis.set_xticklabels([str(symbol) for symbol in sequence])
    input_axis.set_xlabel("Input symbol", labelpad=5, fontsize=10.5)
    input_axis.tick_params(axis="x", length=0, pad=3)
    for tick, symbol in zip(input_axis.get_xticklabels(), sequence):
        tick.set_color(CORAL if symbol == 0 else INDIGO)
        tick.set_fontweight("bold")
    save_square(fig, output, "poster_iterative_dfa")
    return {
        "source": str(trace_file),
        "sequence": sequence,
        "states": true_states,
        "transitions": {
            f"{state},{symbol}": target
            for (state, symbol), target in sorted(DFA_TRANSITIONS.items())
        },
        "checkpoints": len(true_states),
        "all_checkpoints_exact": True,
    }


def plot_time_size(raw_file: Path, output: Path) -> dict[str, object]:
    frame = pd.read_csv(raw_file)
    if not (
        frame["mlp_path_correct"].eq(1).all()
        and frame["ac_path_correct"].eq(1).all()
    ):
        raise ValueError("the matched time-size data contain a failed path")

    summary = (
        frame.groupby("L", as_index=False)
        .agg(
            ac_updates=("ac_updates", "first"),
            mlp_slots=("mlp_dense_parameter_slots", "first"),
            mlp_nonzero=("mlp_nonzero_coefficients", "first"),
        )
        .sort_values("L")
    )
    if not np.array_equal(
        summary["ac_updates"].to_numpy(dtype=int),
        summary["L"].to_numpy(dtype=int),
    ):
        raise ValueError("AC updates do not equal pointer depth")
    slots_per_hop = int(summary.iloc[0]["mlp_slots"])
    expected_slots = slots_per_hop * summary["L"].to_numpy(dtype=int)
    if not np.array_equal(
        summary["mlp_slots"].to_numpy(dtype=int),
        expected_slots,
    ):
        raise ValueError("MLP slots are not an exact linear unrolling")

    selected_depths = (1, 10, 20, 30, 40)
    selected = summary[summary["L"].isin(selected_depths)]
    x = summary["ac_updates"].to_numpy(dtype=float)
    y = summary["mlp_slots"].to_numpy(dtype=float) / 1000.0

    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.15, top=0.88)
    axis.fill_between(x, 0, y, color=CORAL, alpha=0.075, linewidth=0)
    axis.plot(
        x,
        y,
        color=CORAL,
        linewidth=3.2,
        solid_capstyle="round",
    )
    axis.scatter(
        selected["ac_updates"],
        selected["mlp_slots"] / 1000.0,
        s=60,
        color=CORAL,
        edgecolor=WHITE,
        linewidth=1.2,
        zorder=4,
    )
    for row in selected.itertuples():
        axis.annotate(
            rf"$L={int(row.L)}$",
            (row.ac_updates, row.mlp_slots / 1000.0),
            xytext=(7, -12 if row.L == 40 else 5),
            textcoords="offset points",
            color=CORAL,
            fontsize=10,
            fontweight="semibold",
        )

    axis.annotate(
        "Untied MLP\n+2,550 slots per hop",
        xy=(29, 73.95),
        xytext=(12.5, 88),
        arrowprops={
            "arrowstyle": "-|>",
            "color": CORAL,
            "lw": 1.6,
            "connectionstyle": "arc3,rad=-0.10",
        },
        color=CORAL,
        fontsize=11,
        fontweight="semibold",
        ha="left",
    )
    axis.annotate(
        "Fixed AC model\n+1 update per hop",
        xy=(34, 8),
        xytext=(12, 8),
        arrowprops={
            "arrowstyle": "-|>",
            "color": TEAL,
            "lw": 2.1,
        },
        color=TEAL,
        fontsize=11,
        fontweight="semibold",
        va="center",
    )
    axis.text(
        0.04,
        0.93,
        rf"Exact relation: $P={slots_per_hop:,}t$",
        transform=axis.transAxes,
        fontsize=12,
        fontweight="semibold",
        color=CHARCOAL,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": WHITE,
            "edgecolor": GRID,
            "linewidth": 1.0,
        },
    )
    axis.set(
        xlim=(0, 42),
        ylim=(0, 110),
        xlabel="AC internal updates",
        ylabel="Untied MLP parameter slots (thousands)",
        xticks=(0, 10, 20, 30, 40),
        yticks=(0, 25, 50, 75, 100),
    )
    save_square(fig, output, "poster_time_size_trade")
    return {
        "source": str(raw_file),
        "tables": int(frame["seed"].nunique()),
        "starts_per_table": int(frame["start_node"].nunique()),
        "maximum_depth": int(summary["L"].max()),
        "dense_slots_per_hop": slots_per_hop,
        "nonzero_coefficients_per_hop": int(summary.iloc[0]["mlp_nonzero"]),
        "relation": f"P = {slots_per_hop} t",
        "all_complete_paths_exact": True,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ac-root",
        type=Path,
        default=Path("/home/johnh/Documents/assembly-calculus"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.ac_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "results" / "thesis_c" / "poster"
    )
    static_overlap_source = (
        root
        / "results"
        / "thesis_c"
        / "static"
        / "training_dynamics"
        / "mnist_static_overlap_time_series.csv"
    )
    dfa_source = (
        root / "results" / "thesis_c" / "iterative" / "dfa_trace.json"
    )
    time_size_source = (
        root
        / "results"
        / "thesis_c"
        / "resource"
        / "seen_time_size_raw.csv"
    )
    for source in (
        static_overlap_source,
        dfa_source,
        time_size_source,
    ):
        if not source.exists():
            raise FileNotFoundError(source)

    configure_plotting()
    manifest = {
        "visual_system": {
            "size_inches": [6.4, 6.4],
            "raster_pixels": [2560, 2560],
            "colours": {
                "recurrent_teal": TEAL,
                "spatial_or_reversed_coral": CORAL,
                "balanced_or_symbol_one_indigo": INDIGO,
                "charcoal": CHARCOAL,
            },
        },
        "source_sha256": {
            str(static_overlap_source): sha256(static_overlap_source),
            str(dfa_source): sha256(dfa_source),
            str(time_size_source): sha256(time_size_source),
        },
        "static": plot_static_overlap_summary(
            static_overlap_source,
            output,
        ),
        "static_range_preview": plot_static_overlap_summary(
            static_overlap_source,
            output,
            band_mode="range",
            stem="poster_static_overlap_trajectory_range",
        ),
        "static_correction": plot_static_correction_summary(
            static_overlap_source,
            output,
        ),
        "static_combined": plot_static_combined_summary(
            static_overlap_source,
            output,
        ),
        "iterative": plot_dfa(dfa_source, output),
        "time_size": plot_time_size(time_size_source, output),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "poster_results_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
