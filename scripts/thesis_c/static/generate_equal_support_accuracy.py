from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_HORIZON = 15


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-series", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.time_series)
    configure_plotting()
    figure, axis = plt.subplots(figsize=(4.05, 3.15))
    styles = {
        "learned recurrence": ("#009E8E", "Learned recurrence", 4),
        "mean-balanced recurrence": ("#7A5AA6", "Mean-balanced recurrence", 3),
        "no recurrence": ("#4D4D4D", "No recurrence", 2),
    }

    for condition, (colour, label, zorder) in styles.items():
        subset = summary[
            (summary["condition"] == condition)
            & (summary["readout_r"] <= PLOT_HORIZON)
        ]
        x = subset["readout_r"].to_numpy()
        mean = subset["accuracy"].to_numpy()
        low = subset["accuracy_ci_low"].to_numpy()
        high = subset["accuracy_ci_high"].to_numpy()
        axis.plot(
            x,
            mean,
            color=colour,
            linewidth=1.9,
            marker="o",
            markersize=3.6,
            label=label,
            zorder=zorder,
        )
        axis.fill_between(
            x,
            low,
            high,
            color=colour,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )

    axis.set_title("Test accuracy", loc="left", fontweight="bold")
    axis.set_xlabel("Readout $r$")
    axis.set_ylabel("Classification accuracy")
    axis.set_xlim(0.8, PLOT_HORIZON + 0.2)
    axis.set_ylim(0.28, 0.42)
    axis.set_xticks([1, 3, 5, 7, 10, 15])
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=7.4, loc="lower right")
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.88)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(
            args.output.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
