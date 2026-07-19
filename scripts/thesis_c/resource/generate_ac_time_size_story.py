from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#17212B"
MUTED = "#5F6B76"
RULE = "#D7DEE5"
BLUE = "#1769AA"
BLUE_PALE = "#EAF3FA"
AMBER = "#E58B2A"
AMBER_PALE = "#FFF3E5"
TEAL = "#07877B"
TEAL_PALE = "#E7F5F2"
WHITE = "#FFFFFF"


def memory_bank(ax, x: float, y: float, label: str, colour: str, scale: float = 1.0) -> None:
    width = 0.54 * scale
    height = 0.38 * scale
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.055",
            facecolor=WHITE,
            edgecolor=colour,
            linewidth=1.5,
        )
    )
    ax.add_patch(
        Rectangle(
            (x - width / 2 + 0.055, y + height / 2 - 0.09),
            width - 0.11,
            0.035,
            facecolor=colour,
            edgecolor="none",
        )
    )
    ax.text(x, y - 0.025, label, ha="center", va="center", color=INK, fontsize=9.5)


def node(ax, x: float, y: float, label: str, edge: str) -> None:
    ax.add_patch(Circle((x, y), 0.245, facecolor=WHITE, edgecolor=edge, linewidth=2.0))
    ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=10.5, fontweight="bold")


def arrow(ax, x0: float, x1: float, y: float, colour: str, width: float = 1.8) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=colour,
            shrinkA=0,
            shrinkB=0,
        )
    )


def operator(ax, x: float, y: float, label: str, colour: str, width: float = 0.72) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.22),
            width,
            0.44,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=colour,
            edgecolor=colour,
            linewidth=1.0,
        )
    )
    ax.text(x, y, label, ha="center", va="center", color=WHITE, fontsize=10.5, fontweight="bold")


def draw_temporal(ax, y: float) -> None:
    memory_bank(ax, 4.05, y, r"$M$", BLUE, 1.12)
    ax.text(4.05, y - 0.49, "1 bank", ha="center", color=MUTED, fontsize=9.5)

    node(ax, 6.25, y, r"$v_0$", BLUE)
    operator(ax, 7.15, y, r"$M$", BLUE, 0.60)
    arrow(ax, 6.53, 6.82, y, BLUE)
    arrow(ax, 7.48, 7.75, y, BLUE)

    for tick in range(9):
        x = 7.92 + tick * 0.19
        ax.plot([x, x], [y - 0.13, y + 0.13], color=BLUE, linewidth=1.5, alpha=0.78)
    ax.text(8.68, y + 0.36, "repeat 40 times", ha="center", color=BLUE, fontsize=9.5, fontweight="bold")
    arrow(ax, 9.62, 10.15, y, BLUE)
    node(ax, 10.48, y, r"$v_{40}$", BLUE)


def draw_binary(ax, y: float) -> None:
    labels = (r"$M$", r"$M^2$", r"$M^4$", r"$M^8$", r"$M^{16}$", r"$M^{32}$")
    for index, label in enumerate(labels):
        x = 3.55 + (index % 3) * 0.69
        yy = y + 0.25 - (index // 3) * 0.50
        memory_bank(ax, x, yy, label, AMBER, 0.92)
    ax.text(4.24, y - 0.58, "6 banks", ha="center", color=MUTED, fontsize=9.5)

    node(ax, 6.25, y, r"$v_0$", AMBER)
    arrow(ax, 6.53, 6.80, y, AMBER)
    operator(ax, 7.22, y, r"$M^{32}$", AMBER, 0.78)
    arrow(ax, 7.63, 8.05, y, AMBER)
    operator(ax, 8.48, y, r"$M^8$", AMBER, 0.72)
    arrow(ax, 8.86, 10.15, y, AMBER)
    node(ax, 10.48, y, r"$v_{40}$", AMBER)
    ax.text(7.86, y + 0.42, r"$40=32+8$", ha="center", color=AMBER, fontsize=10, fontweight="bold")


def draw_direct(ax, y: float) -> None:
    start_x = 3.18
    start_y = y + 0.36
    size = 0.15
    gap = 0.045
    for row in range(4):
        for col in range(10):
            index = row * 10 + col + 1
            x = start_x + col * (size + gap)
            yy = start_y - row * (size + gap)
            ax.add_patch(
                Rectangle(
                    (x, yy - size),
                    size,
                    size,
                    facecolor=TEAL if index == 40 else WHITE,
                    edgecolor=TEAL,
                    linewidth=1.0,
                )
            )
    ax.text(4.06, y - 0.55, r"40 banks: $M^1,\ldots,M^{40}$", ha="center", color=MUTED, fontsize=9.5)

    node(ax, 6.25, y, r"$v_0$", TEAL)
    arrow(ax, 6.53, 7.48, y, TEAL)
    operator(ax, 8.02, y, r"$M^{40}$", TEAL, 0.92)
    arrow(ax, 8.50, 10.15, y, TEAL)
    node(ax, 10.48, y, r"$v_{40}$", TEAL)
    ax.text(8.02, y + 0.42, "direct composed operator", ha="center", color=TEAL, fontsize=9.5, fontweight="bold")


def main() -> None:
    output = Path(__file__).resolve().parent
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "mathtext.fontset": "dejavusans",
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(14.6, 8.0), facecolor=WHITE)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    ax.text(
        0.55,
        7.48,
        "Reuse computation in time, or store it in space",
        color=INK,
        fontsize=25,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.58,
        6.94,
        "Three ways to answer the same 40-hop pointer query with a frozen controller",
        color=MUTED,
        fontsize=12.5,
        va="top",
    )
    ax.text(0.60, 6.34, "COMPUTATION", color=MUTED, fontsize=9.5, fontweight="bold")
    ax.text(3.00, 6.34, "STORED OPERATORS", color=MUTED, fontsize=9.5, fontweight="bold")
    ax.text(6.05, 6.34, "FROZEN QUERY EXECUTION", color=MUTED, fontsize=9.5, fontweight="bold")
    ax.text(12.00, 6.34, "INTERNAL UPDATES", color=MUTED, fontsize=9.5, fontweight="bold", ha="center")
    ax.plot([0.55, 14.05], [6.20, 6.20], color=RULE, linewidth=1.2)

    rows = (
        (5.28, BLUE_PALE, BLUE, "TEMPORAL REUSE", "One operator, reused"),
        (3.69, AMBER_PALE, AMBER, "BINARY LIBRARY", "Composed powers"),
        (2.10, TEAL_PALE, TEAL, "DIRECT LIBRARY", "Every supported depth"),
    )
    for y, pale, colour, title, subtitle in rows:
        ax.add_patch(
            FancyBboxPatch(
                (0.50, y - 0.67),
                13.58,
                1.30,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=pale,
                edgecolor="none",
            )
        )
        ax.add_patch(Rectangle((0.50, y - 0.67), 0.09, 1.30, facecolor=colour, edgecolor="none"))
        ax.text(0.82, y + 0.12, title, color=colour, fontsize=13, fontweight="bold", va="center")
        ax.text(0.82, y - 0.23, subtitle, color=MUTED, fontsize=10, va="center")

    draw_temporal(ax, 5.28)
    draw_binary(ax, 3.69)
    draw_direct(ax, 2.10)

    for y, value, queries, colour in (
        (5.28, "80", "40 queries", BLUE),
        (3.69, "4", "2 queries", AMBER),
        (2.10, "2", "1 query", TEAL),
    ):
        ax.text(12.00, y + 0.10, value, ha="center", va="center", color=colour, fontsize=28, fontweight="bold")
        ax.text(12.00, y - 0.32, queries, ha="center", va="center", color=MUTED, fontsize=10)

    ax.text(
        0.58,
        0.78,
        "LESS STORED STRUCTURE",
        color=BLUE,
        fontsize=9.5,
        fontweight="bold",
        va="center",
    )
    arrow(ax, 2.40, 10.95, 0.78, INK, width=1.6)
    ax.text(
        11.18,
        0.78,
        "MORE PRECOMPUTED STRUCTURE",
        color=TEAL,
        fontsize=9.5,
        fontweight="bold",
        va="center",
    )
    ax.text(
        0.58,
        0.29,
        "Query cost uses the modular pointer controller's two-update memory operation. "
        "Compiling and writing additional banks is a separate setup cost.",
        color=MUTED,
        fontsize=9.2,
        va="center",
    )

    fig.savefig(output / "ac_time_size_story.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output / "ac_time_size_story.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    main()
