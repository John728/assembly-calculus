from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


COLOURS = {1: "#CC3311", 10: "#0077BB", 100: "#009988"}
LABELS = {1: "1 image/class", 10: "10 images/class", 100: "Full training pool"}


def seed_interval(values: pd.Series) -> tuple[float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return mean, 0.0
    critical = float(student_t.ppf(0.975, len(data) - 1))
    half_width = critical * float(data.std(ddof=1)) / np.sqrt(len(data))
    return mean, half_width


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    frame = pd.read_csv(args.raw)
    frame["correct"] = frame["correct"].astype(bool)
    frame["unsettled"] = frame["settling_readout"] > frame["readout_r"]
    per_seed = (
        frame.groupby(["presentation_rounds", "readout_r", "seed"], as_index=False)
        .agg(accuracy=("correct", "mean"), unsettled=("unsettled", "mean"))
    )
    per_seed = per_seed[per_seed["readout_r"] <= 20]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharex=True)
    for exposure in (1, 10, 100):
        subset = per_seed[per_seed["presentation_rounds"] == exposure]
        readouts = np.sort(subset["readout_r"].unique())
        for axis, metric in zip(axes, ("accuracy", "unsettled")):
            means: list[float] = []
            intervals: list[float] = []
            for readout in readouts:
                values = subset.loc[subset["readout_r"] == readout, metric]
                mean, interval = seed_interval(values)
                means.append(mean)
                intervals.append(interval)
            mean_array = np.asarray(means)
            interval_array = np.asarray(intervals)
            axis.plot(
                readouts,
                mean_array,
                color=COLOURS[exposure],
                lw=1.55,
                label=LABELS[exposure],
            )
            axis.fill_between(
                readouts,
                np.clip(mean_array - interval_array, 0, 1),
                np.clip(mean_array + interval_array, 0, 1),
                color=COLOURS[exposure],
                alpha=0.12,
                linewidth=0,
            )

    axes[0].set(
        xlabel="Internal update $r$",
        ylabel="Classification accuracy",
        xlim=(1, 20),
        ylim=(0.32, 0.72),
    )
    axes[1].set(
        xlabel="Internal update $r$",
        ylabel="Trajectories not yet settled",
        xlim=(1, 20),
        ylim=(-0.01, 0.38),
    )
    axes[0].text(-0.15, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.15, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)

    stem = args.output / "mnist_static_settling"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
