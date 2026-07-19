from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D9E0E6"
BLUE = "#1769AA"
ORANGE = "#E58B2A"


def main() -> None:
    output = Path(__file__).resolve().parent
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 17,
            "pdf.fonttype": 42,
            "savefig.dpi": 320,
        }
    )

    depth = np.arange(1, 41)
    recurrent_size = np.ones_like(depth)
    unrolled_size = depth

    fig, ax = plt.subplots(figsize=(8.4, 5.4), facecolor="white")
    ax.plot(
        depth,
        unrolled_size,
        color=ORANGE,
        linewidth=3.0,
        label="Spatially unrolled AC",
        zorder=3,
    )
    ax.plot(
        depth,
        recurrent_size,
        color=BLUE,
        linewidth=3.2,
        label="Recurrent AC",
        zorder=4,
    )
    ax.scatter([40], [40], color=ORANGE, s=75, edgecolor="white", linewidth=1.3, zorder=5)
    ax.scatter([40], [1], color=BLUE, s=75, edgecolor="white", linewidth=1.3, zorder=5)

    ax.annotate(
        "40 module copies",
        xy=(40, 40),
        xytext=(33.2, 36.0),
        color=ORANGE,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 1.4},
    )
    ax.annotate(
        "1 reusable module",
        xy=(40, 1),
        xytext=(31.0, 5.0),
        color=BLUE,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 1.4},
    )

    ax.set_title("Model size required as pointer depth grows", loc="left", color=INK, fontweight="bold", pad=16)
    ax.text(
        0.0,
        1.015,
        "The recurrent AC reuses one transition module; the unrolled AC copies it",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10.5,
        ha="left",
        va="bottom",
    )
    ax.set_xlabel(r"Required pointer depth $L$", color=INK, labelpad=10)
    ax.set_ylabel("Relative model size\n(one transition module = 1)", color=INK, labelpad=12)
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 43)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.grid(True, color=GRID, linewidth=0.85)
    ax.tick_params(axis="both", colors=MUTED, length=0, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=10.5)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.16)
    fig.savefig(output / "recurrent_vs_unrolled_size.pdf", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(output / "recurrent_vs_unrolled_size.png", bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


if __name__ == "__main__":
    main()
