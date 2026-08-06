from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


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


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    # Average within each network seed first, then form the interval across seeds.
    by_seed = (
        frame.groupby(["seed", "training_presentation"], as_index=False)["eta"]
        .mean()
        .rename(columns={"eta": "seed_mean"})
    )
    rows = []
    for presentation, group in by_seed.groupby("training_presentation"):
        values = group["seed_mean"].to_numpy(dtype=float)
        mean = float(values.mean())
        if len(values) > 1:
            half_width = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
        else:
            half_width = 0.0
        rows.append(
            {
                "training_presentation": int(presentation),
                "mean": mean,
                "ci_low": mean - half_width,
                "ci_high": mean + half_width,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("training_presentation")


def draw_panel(axis: plt.Axes, summary: pd.DataFrame, *, zoom: bool) -> None:
    if zoom:
        summary = summary[summary["training_presentation"] <= 9]
    x = summary["training_presentation"].to_numpy()
    mean = summary["mean"].to_numpy()
    low = summary["ci_low"].to_numpy()
    high = summary["ci_high"].to_numpy()

    colour = "#009E8E"
    axis.fill_between(x, low, high, color=colour, alpha=0.16, linewidth=0)
    axis.plot(
        x,
        mean,
        color=colour,
        linewidth=1.9,
        marker="o" if zoom else None,
        markersize=4,
        label="Mean +/- 95% seed interval",
    )
    axis.axhline(1.0, color="#707070", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Training presentation")
    axis.set_ylabel(r"Consecutive cap overlap $\eta$")
    axis.set_ylim(0.4 if zoom else 0.80, 1.005)
    axis.set_xlim(1.8 if zoom else 1.5, 9.2 if zoom else 50.5)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if zoom:
        axis.set_xticks(range(2, 10))
        axis.set_title("Formation transient", loc="left", fontweight="bold")
        for presentation in (2, 3, 4, 5):
            value = float(
                summary.loc[
                    summary["training_presentation"] == presentation, "mean"
                ].iloc[0]
            )
            offset = 8 if presentation in (2, 3) else -14
            axis.annotate(
                f"{value:.3f}",
                (presentation, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#303030",
            )
    else:
        axis.set_xticks([2, 10, 20, 30, 40, 50])
        axis.set_title("Full training block", loc="left", fontweight="bold")
        axis.axvline(9, color="#707070", linewidth=0.8, linestyle=":")
        axis.text(
            9.7,
            0.812,
            "identical caps\nfrom presentation 9",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#505050",
        )
        axis.legend(frameon=False, loc="lower right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    configure_plotting()
    summary = summarise(pd.read_csv(args.source))
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"), index=False)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.25),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    draw_panel(axes[0], summary, zoom=False)
    draw_panel(axes[1], summary, zoom=True)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.87, wspace=0.30)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(
            args.output.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
