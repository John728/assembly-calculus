from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from scipy.stats import t as student_t


COLOURS = {1: "#CC3311", 2: "#EE7733", 4: "#0077BB", 10: "#009988"}


def configure() -> None:
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


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def mean_interval(values: pd.Series) -> tuple[float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return mean, 0.0
    critical = float(student_t.ppf(0.975, len(data) - 1))
    half = critical * float(data.std(ddof=1)) / np.sqrt(len(data))
    return mean, half


def plot_architecture(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.1, 2.8))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4)
    axis.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, subtitle: str, colour: str):
        patch = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=colour, edgecolor="0.2", linewidth=1.0,
        )
        axis.add_patch(patch)
        axis.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontweight="bold")
        axis.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=8)

    box(0.20, 1.35, 1.45, 1.35, "Current", "input/output\nstate", "#E8F1F8")
    box(2.30, 1.05, 1.85, 1.95, "Control", "query / writeback\nassemblies", "#E5F4EE")
    box(5.00, 1.45, 1.45, 1.30, "Source", "memory key", "#FFF2CC")
    box(7.50, 1.45, 1.55, 1.30, "Destination", "memory value", "#FFF2CC")

    def arrow(start: tuple[float, float], end: tuple[float, float], colour: str, label: str, yoff: float = 0):
        axis.annotate(
            "", xy=end, xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": colour},
        )
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + yoff,
                  label, ha="center", va="center", fontsize=8, color=colour)

    arrow((1.65, 2.35), (2.30, 2.35), "#0077BB", "query", 0.22)
    arrow((4.15, 2.35), (5.00, 2.35), "#0077BB", "route key", 0.22)
    arrow((6.45, 2.35), (7.50, 2.35), "#CC3311", "episodic pointer", 0.22)
    arrow((7.50, 1.67), (4.15, 1.38), "#009988", "writeback", -0.34)
    arrow((2.30, 1.42), (1.65, 1.42), "#009988", "next state", -0.24)

    axis.text(7.02, 3.15, "rewritten for each unseen table", ha="center", fontsize=8,
              color="#8A270E")
    axis.text(3.22, 0.40, "controller weights frozen during evaluation", ha="center",
              fontsize=8, color="0.25")
    axis.text(8.25, 0.82, "one hop", ha="center", fontweight="bold")
    axis.text(8.25, 0.50, "query + writeback", ha="center", fontsize=8, color="0.25")
    save(fig, output, "pointer_unseen_architecture")


def plot_write_strength(frame: pd.DataFrame, output: Path) -> None:
    per_seed = (
        frame.groupby(["write_rounds", "L", "seed"], as_index=False)
        .agg(path_survival=("path_correct", "mean"))
    )
    fig, axis = plt.subplots(figsize=(5.9, 3.25))
    available_rounds = sorted(per_seed["write_rounds"].unique())
    for rounds in available_rounds:
        if rounds == 10 and 4 in available_rounds:
            continue
        subset = per_seed[per_seed["write_rounds"] == rounds]
        depths = sorted(subset["L"].unique())
        means, intervals = [], []
        for depth in depths:
            mean, interval = mean_interval(subset.loc[subset["L"] == depth, "path_survival"])
            means.append(mean)
            intervals.append(interval)
        means_array = np.asarray(means)
        intervals_array = np.asarray(intervals)
        colour = COLOURS.get(int(rounds), "0.3")
        if rounds == 4 and 10 in available_rounds:
            label = "4 or 10 write passes"
        else:
            label = f"{rounds} write pass{'es' if rounds != 1 else ''}"
        axis.plot(depths, means_array, marker="o", ms=3.5, lw=1.6, color=colour,
                  label=label)
        axis.fill_between(
            depths,
            np.clip(means_array - intervals_array, 0, 1),
            np.clip(means_array + intervals_array, 0, 1),
            color=colour,
            alpha=0.12,
            linewidth=0,
        )
    node_count = int(max(frame["target"].max(), frame["prediction"].max())) + 1
    axis.axhline(1.0 / node_count, color="0.4", lw=0.8, ls=":")
    axis.set(xlabel="Pointer depth $L$", ylabel="Complete-path accuracy", ylim=(-0.02, 1.02))
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    save(fig, output, "pointer_unseen_write_strength")


def plot_trace(trace_path: Path, output: Path) -> None:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    true_path = [int(value) for value in trace["true_path"]]
    phases = trace["phase_rows"]
    hops = min(len(phases), 8)
    phases = phases[:hops]
    true_path = true_path[: hops + 1]
    matrix = np.full((4, hops + 1), np.nan)
    matrix[0, :] = true_path
    matrix[1, 0] = true_path[0]
    for column, phase in enumerate(phases, start=1):
        matrix[1, column] = int(phase["current_node"])
        matrix[2, column] = int(phase["source_node"])
        matrix[3, column] = int(phase["destination_node"])

    fig, axis = plt.subplots(figsize=(7.1, 2.65))
    node_count = len(trace["pointer"])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = axis.imshow(
        matrix, cmap=cmap, vmin=-0.5, vmax=node_count - 0.5, aspect="auto"
    )
    axis.set_xticks(range(hops + 1), labels=[str(value) for value in range(hops + 1)])
    axis.set_yticks(
        range(4),
        labels=["Required state", "Current after writeback", "Queried source", "Retrieved destination"],
    )
    axis.set_xlabel("Pointer hop")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                fraction = matrix[row, column] / max(node_count - 1, 1)
                colour = "white" if fraction < 0.55 else "black"
                axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center",
                          color=colour, fontsize=8, fontweight="bold")
    ticks = sorted(set([0, node_count // 4, node_count // 2, 3 * node_count // 4, node_count - 1]))
    colourbar = fig.colorbar(image, ax=axis, pad=0.02, ticks=ticks)
    colourbar.set_label("Node identity")
    axis.grid(False)
    fig.tight_layout()
    save(fig, output, "pointer_unseen_trace")


def budget_frame(frame: pd.DataFrame, write_rounds: int) -> pd.DataFrame:
    locked = frame[frame["write_rounds"] == write_rounds].copy()
    keys = ["seed", "table_index", "start_node"]
    rows: list[dict[str, int | float]] = []
    for key, group in locked.groupby(keys):
        group = group.sort_values("L")
        predictions = {0: int(group.iloc[0]["start_node"])}
        targets = {}
        for row in group.itertuples():
            predictions[int(row.L)] = int(row.prediction)
            targets[int(row.L)] = int(row.target)
        maximum = max(targets)
        for depth in range(1, maximum + 1):
            for budget in range(0, 2 * maximum + 1):
                executed = min(budget // 2, depth)
                prediction = predictions[executed]
                rows.append(
                    {
                        "seed": int(key[0]),
                        "table_index": int(key[1]),
                        "start_node": int(key[2]),
                        "L": depth,
                        "t": budget,
                        "executed_hops": executed,
                        "target": targets[depth],
                        "prediction": prediction,
                        "accuracy": float(prediction == targets[depth]),
                    }
                )
    return pd.DataFrame(rows)


def plot_reach(frame: pd.DataFrame, output: Path, write_rounds: int = 4) -> None:
    budget = budget_frame(frame, write_rounds)
    budget.to_csv(output / "pointer_unseen_budget_raw.csv", index=False)
    summary = budget.groupby(["L", "t"], as_index=False).agg(accuracy=("accuracy", "mean"))
    depths = sorted(summary["L"].unique())
    budgets = sorted(summary["t"].unique())
    matrix = np.full((len(depths), len(budgets)), np.nan)
    for row_index, depth in enumerate(depths):
        for column_index, time in enumerate(budgets):
            value = summary.loc[(summary["L"] == depth) & (summary["t"] == time), "accuracy"]
            matrix[row_index, column_index] = float(value.iloc[0])

    fig, axis = plt.subplots(figsize=(6.4, 3.35))
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="cividis", vmin=0, vmax=1)
    axis.set_xticks(range(len(budgets)), labels=[str(value) for value in budgets])
    axis.set_yticks(range(len(depths)), labels=[str(value) for value in depths])
    axis.set(xlabel="Internal update budget $t$", ylabel="Pointer depth $L$")
    for depth in depths:
        boundary = 2 * depth
        axis.plot(budgets.index(boundary), depths.index(depth), marker="s", ms=5,
                  markerfacecolor="none", markeredgecolor="white", markeredgewidth=0.8)
    colourbar = fig.colorbar(image, ax=axis, pad=0.02)
    colourbar.set_label("Final-state accuracy")
    fig.tight_layout()
    save(fig, output, "pointer_unseen_reach_boundary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    frame = pd.read_csv(args.raw)
    plot_architecture(args.output)
    plot_write_strength(frame, args.output)
    plot_trace(args.trace, args.output)
    plot_reach(frame[frame["L"] <= 8].copy(), args.output)


if __name__ == "__main__":
    main()
