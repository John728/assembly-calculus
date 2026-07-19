from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D9E0E6"
BLUE = "#1769AA"
AMBER = "#E58B2A"
TEAL = "#07877B"
AMBER_PALE = "#FFF3E5"


def main() -> None:
    output = Path(__file__).resolve().parent
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 18,
            "pdf.fonttype": 42,
            "savefig.dpi": 320,
        }
    )

    bank_counts = [1, 2, 3, 4, 5, 6]
    query_updates = [80, 40, 20, 10, 6, 4]

    fig, ax = plt.subplots(figsize=(9.2, 5.9), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlim(0.82, 52)
    ax.set_ylim(1.55, 112)

    ax.plot(
        bank_counts,
        query_updates,
        color=AMBER,
        linewidth=3.0,
        marker="o",
        markersize=7.5,
        markerfacecolor="white",
        markeredgecolor=AMBER,
        markeredgewidth=2.2,
        zorder=3,
    )
    ax.plot([6, 40], [4, 2], color=MUTED, linewidth=1.8, linestyle=(0, (4, 4)), zorder=2)

    ax.scatter([1], [80], s=150, color=BLUE, edgecolor="white", linewidth=1.5, zorder=5)
    ax.scatter([6], [4], s=170, color=AMBER, edgecolor="white", linewidth=1.5, zorder=5)
    ax.scatter([40], [2], s=170, color=TEAL, edgecolor="white", linewidth=1.5, zorder=5)

    ax.axhspan(1.55, 4.9, color="#E8F5F2", alpha=0.72, zorder=0)
    ax.text(
        0.97,
        98,
        "REUSE IN TIME",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        48,
        1.68,
        "PRECOMPUTE IN SPACE",
        color=TEAL,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    ax.annotate(
        "1 bank\n80 updates",
        xy=(1, 80),
        xytext=(1.35, 72),
        color=BLUE,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 1.4},
    )
    ax.annotate(
        "6 power banks\n4 updates",
        xy=(6, 4),
        xytext=(8.2, 7.4),
        color=AMBER,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": AMBER, "linewidth": 1.4},
    )
    ax.annotate(
        "40 direct banks\n2 updates",
        xy=(40, 2),
        xytext=(25.5, 3.1),
        color=TEAL,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": TEAL, "linewidth": 1.4},
    )

    ax.text(
        6,
        2.35,
        r"$20\times$ fewer updates",
        color=AMBER,
        fontsize=11.5,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": AMBER_PALE, "edgecolor": "none"},
    )
    ax.annotate(
        "more computation stored before the query",
        xy=(30, 5.3),
        xytext=(8.5, 16),
        color=MUTED,
        fontsize=10.5,
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.3},
    )

    ax.set_title("Stored composition buys shorter queries", loc="left", color=INK, fontweight="bold", pad=20)
    ax.text(
        0.0,
        1.015,
        r"Same depth-40 pointer query $M^{40}(v_0)$; frozen modular controller",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=11,
        ha="left",
        va="bottom",
    )
    ax.text(
        1.0,
        1.015,
        "1 memory query = 2 internal updates",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        ha="right",
        va="bottom",
    )

    ax.set_xlabel("Stored pointer operators (memory banks)", color=INK, labelpad=12)
    ax.set_ylabel("Internal updates for the query", color=INK, labelpad=12)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5, 6, 10, 20, 40]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "2", "3", "4", "5", "6", "10", "20", "40"]))
    ax.yaxis.set_major_locator(FixedLocator([2, 4, 6, 10, 20, 40, 80]))
    ax.yaxis.set_major_formatter(FixedFormatter(["2", "4", "6", "10", "20", "40", "80"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(which="major", color=GRID, linewidth=0.85, alpha=0.85)
    ax.tick_params(axis="both", colors=MUTED, length=0, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    fig.text(
        0.11,
        0.012,
        "Derived plan costs. Compiling and writing additional banks is paid before frozen query execution.",
        color=MUTED,
        fontsize=9.4,
        ha="left",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.18)
    fig.savefig(output / "ac_time_size_frontier_plot.pdf", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(output / "ac_time_size_frontier_plot.png", bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


if __name__ == "__main__":
    main()
