from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


FULL_POOL = "distinct_50"
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
        & (frame["condition"] == FULL_POOL)
    ]
    priorities = ("stable correct", "corrected", "corrupted", "stable wrong")
    selected: list[pd.Series] = []
    used_targets: set[int] = set()
    for transition_type in priorities:
        candidates = condition[
            condition["transition_type"] == transition_type
        ].sort_values("instance_id")
        distinct = candidates[~candidates["target"].isin(used_targets)]
        if len(distinct):
            choice = distinct.iloc[0]
        elif len(candidates):
            choice = candidates.iloc[0]
        else:
            continue
        selected.append(choice)
        used_targets.add(int(choice["target"]))
    if len(selected) < 4:
        used_ids = {int(row["instance_id"]) for row in selected}
        remaining = condition[
            ~condition["instance_id"].isin(used_ids)
        ].sort_values("instance_id")
        selected.extend(row for _, row in remaining.head(4 - len(selected)).iterrows())

    records: list[dict[str, int | str]] = []
    for row in selected[:4]:
        records.append(
            {
                "instance_id": int(row["instance_id"]),
                "target": int(row["target"]),
                "first": int(row["initial_prediction"]),
                "settled": int(row["final_prediction"]),
                "kind": str(row["transition_type"]),
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
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))

    from pyac.tasks.mnist import load_mnist_split  # type: ignore

    configure_plotting()
    frame = pd.read_csv(args.raw)
    examples = select_examples(frame)
    test = load_mnist_split(args.data_dir, "test")

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
        result_colour = {
            "corrected": "#117733",
            "corrupted": "#CC3311",
            "stable correct": "#117733",
        }.get(str(example["kind"]), "#555555")
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
        "condition": {"seed": SEED, "condition": FULL_POOL},
        "selection_rule": (
            "Lowest source-index example in each available transition category: "
            "stable correct, corrected, corrupted, and stable wrong; distinct "
            "targets are preferred."
        ),
        "examples": examples,
    }
    stem.with_name(f"{stem.name}_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
