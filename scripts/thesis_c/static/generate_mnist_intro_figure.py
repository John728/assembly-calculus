from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


FULL_POOL = 100
SEED = 42


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
        }
    )


def select_examples(frame: pd.DataFrame) -> list[dict[str, int | str]]:
    condition = frame[
        (frame["seed"] == SEED)
        & (frame["presentation_rounds"] == FULL_POOL)
        & (frame["readout_r"].isin([1, 100]))
    ]
    endpoints = condition.pivot(
        index=["instance_id", "target"],
        columns="readout_r",
        values="prediction",
    ).reset_index()
    endpoints = endpoints.rename(columns={1: "first", 100: "settled"})

    stable = endpoints[
        (endpoints["first"] == endpoints["target"])
        & (endpoints["settled"] == endpoints["target"])
    ]
    stable_distinct = stable.drop_duplicates(subset="target").head(2)
    corrected = endpoints[
        (endpoints["first"] != endpoints["target"])
        & (endpoints["settled"] == endpoints["target"])
    ].head(2)
    selected = pd.concat([stable_distinct, corrected], ignore_index=True)

    if len(selected) != 4 or len(corrected) != 2:
        raise RuntimeError("Could not select two stable and two corrected examples")

    records: list[dict[str, int | str]] = []
    for position, row in selected.iterrows():
        records.append(
            {
                "instance_id": int(row["instance_id"]),
                "target": int(row["target"]),
                "first": int(row["first"]),
                "settled": int(row["settled"]),
                "kind": "stable" if position < 2 else "corrected",
            }
        )
    return records


def rounded_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    colour: str,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#303030",
        facecolor=colour,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        linespacing=1.25,
    )
    return box


def downward_arrow(axis: plt.Axes, y_top: float, y_bottom: float) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (0.5, y_top),
            (0.5, y_bottom),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color="#555555",
        )
    )


def draw_architecture(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    rounded_box(
        axis,
        (0.15, 0.81),
        0.70,
        0.105,
        "28 x 28 greyscale image",
        "#F2F2F2",
    )
    rounded_box(
        axis,
        (0.15, 0.60),
        0.70,
        0.13,
        "Sensory area $X$\n784 neurons; top 200 active",
        "#FFF1CC",
    )
    rounded_box(
        axis,
        (0.15, 0.34),
        0.70,
        0.18,
        "Recurrent coding area $A$\n2,000 neurons; $k=200$\nupdated $r$ times",
        "#DDEAF7",
    )
    rounded_box(
        axis,
        (0.15, 0.10),
        0.70,
        0.16,
        "Largest overlap with the\nten learned digit assemblies",
        "#E1F1E7",
    )

    downward_arrow(axis, 0.81, 0.735)
    downward_arrow(axis, 0.60, 0.525)
    downward_arrow(axis, 0.34, 0.265)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))

    from pyac.tasks.mnist import load_mnist_split  # type: ignore

    configure_plotting()
    frame = pd.read_csv(args.raw)
    examples = select_examples(frame)
    test = load_mnist_split(args.ac_root / "data" / "mnist", "test")

    fig = plt.figure(figsize=(7.15, 3.6))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.18, 1.0),
        wspace=0.24,
        left=0.055,
        right=0.96,
        top=0.82,
        bottom=0.20,
    )
    left = outer[0].subgridspec(2, 2, hspace=0.48, wspace=0.16)

    for index, example in enumerate(examples):
        axis = fig.add_subplot(left[index // 2, index % 2])
        instance_id = int(example["instance_id"])
        axis.imshow(test.images[instance_id], cmap="gray_r", vmin=0, vmax=1)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#B5B5B5")
            spine.set_linewidth(0.8)

        target = int(example["target"])
        first = int(example["first"])
        settled = int(example["settled"])
        axis.set_title(f"True digit: {target}", pad=4, fontsize=9)
        result_colour = "#117733"
        axis.text(
            0.5,
            -0.13,
            rf"model: ${first}\,\rightarrow\,{settled}$",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color=result_colour,
            fontsize=9,
            fontweight="bold",
        )

    fig.text(
        0.285,
        0.90,
        "Real test images and model readouts",
        ha="center",
        fontweight="bold",
        fontsize=10,
    )
    fig.text(
        0.285,
        0.105,
        r"first readout $\rightarrow$ readout after 100 fixed-input updates",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )

    architecture = fig.add_subplot(outer[1])
    draw_architecture(architecture)
    fig.text(
        0.755,
        0.90,
        "What the model computes",
        ha="center",
        fontweight="bold",
        fontsize=10,
    )
    fig.text(0.018, 0.955, "(a)", fontweight="bold")
    fig.text(0.555, 0.955, "(b)", fontweight="bold")

    stem = args.output / "mnist_examples_architecture"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "source_results": str(args.raw),
        "condition": {"seed": SEED, "presentation_rounds": FULL_POOL},
        "selection_rule": (
            "First two distinct-label examples correct at both readouts, followed "
            "by the first two examples incorrect at readout 1 and correct at 100."
        ),
        "examples": examples,
    }
    stem.with_name(f"{stem.name}_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
