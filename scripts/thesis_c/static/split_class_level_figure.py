from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#202020",
            "axes.linewidth": 0.8,
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "savefig.dpi": 300,
        }
    )


def save(figure: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(
            output / f"{name}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


def support_matrix_figure(matrix_frame: pd.DataFrame, output: Path) -> None:
    mean_matrix = (
        matrix_frame.groupby(["target_class", "source_class"])["support"]
        .mean()
        .unstack("source_class")
        .to_numpy()
    )
    positive = mean_matrix[mean_matrix > 0]
    norm = colors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))

    figure, axis = plt.subplots(figsize=(4.0, 3.75))
    heatmap = axis.imshow(mean_matrix, cmap="viridis", norm=norm, aspect="equal")
    axis.set_xlabel("Source class $d$")
    axis.set_ylabel("Target class $c$")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    axis.set_title("Class-to-class recurrent support", fontweight="bold")
    for digit in range(10):
        axis.text(
            digit,
            digit,
            f"{mean_matrix[digit, digit]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    colourbar = figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    colourbar.set_label("Mean support (log scale)")
    figure.subplots_adjust(left=0.16, right=0.84, bottom=0.14, top=0.88)
    save(figure, output, "class_level_support_matrix")


def scalar_figure(frame: pd.DataFrame, output: Path) -> None:
    display = frame.sample(n=min(25000, len(frame)), random_state=42)
    figure, axis = plt.subplots(figsize=(4.0, 3.75))
    axis.scatter(
        display["scalar_prediction"],
        display["exact_recurrent_input"],
        s=8,
        alpha=0.13,
        color="#009E8E",
        edgecolors="none",
        rasterized=True,
    )
    upper = float(
        max(frame["scalar_prediction"].max(), frame["exact_recurrent_input"].max())
    )
    axis.plot(
        [0, upper],
        [0, upper],
        color="#202020",
        linewidth=1.0,
        linestyle="--",
    )
    axis.set_xlim(0, upper * 1.02)
    axis.set_ylim(0, upper * 1.02)
    axis.set_xlabel(r"Predicted $\lambda_c o_{A,c}(r)$")
    axis.set_ylabel(r"Exact $R_c(r)$")
    axis.set_title("Scalar recurrent-input approximation", fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.17, right=0.96, bottom=0.15, top=0.88)
    save(figure, output, "class_level_scalar_approximation")


def error_figure(summary: pd.DataFrame, output: Path) -> None:
    readouts = summary["readout_r"].to_numpy()
    means = summary["scalar_mae"].to_numpy()
    low = summary["scalar_mae_ci_low"].to_numpy()
    high = summary["scalar_mae_ci_high"].to_numpy()

    figure, axis = plt.subplots(figsize=(4.0, 3.75))
    axis.plot(
        readouts,
        means,
        marker="o",
        markersize=5,
        linewidth=2.0,
        color="#009E8E",
    )
    axis.fill_between(
        readouts,
        low,
        high,
        color="#009E8E",
        alpha=0.16,
        linewidth=0,
    )
    axis.set_xticks(readouts)
    axis.set_xlabel("Readout $r$")
    axis.set_ylabel("Mean absolute error")
    axis.set_title("Approximation error by readout", fontweight="bold")
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.18, right=0.96, bottom=0.15, top=0.88)
    save(figure, output, "class_level_error_by_readout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    frame = pd.read_csv(args.source / "class_level_exact_predictions.csv")
    matrix_frame = pd.read_csv(args.source / "class_level_support_matrix.csv")
    summary = pd.read_csv(args.source / "class_level_metrics_by_readout.csv")
    support_matrix_figure(matrix_frame, args.output)
    scalar_figure(frame, args.output)
    error_figure(summary, args.output)


if __name__ == "__main__":
    main()
