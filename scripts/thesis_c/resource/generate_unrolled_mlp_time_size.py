from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from scipy.sparse import csr_matrix


BLUE = "#1769AA"
ORANGE = "#E58B2A"
TEAL = "#07877B"
INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D9E0E6"
DEPTH_MARKERS = (4, 8, 16, 24, 32, 40)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "pdf.fonttype": 42,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.85)
    ax.tick_params(axis="both", colors=MUTED, length=0, pad=6)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


class ExactPointerBlock:
    """A constructed one-hidden-layer ReLU MLP implementing one table lookup."""

    def __init__(self, nodes: int):
        self.nodes = nodes
        self.table_dim = nodes * nodes
        self.input_dim = self.table_dim + nodes
        self.hidden_dim = self.table_dim
        self.output_dim = nodes

        w1_rows: list[int] = []
        w1_cols: list[int] = []
        for source in range(nodes):
            for destination in range(nodes):
                hidden = source * nodes + destination
                w1_rows.extend((hidden, self.table_dim + source))
                w1_cols.extend((hidden, hidden))
        self.w1 = csr_matrix(
            (
                np.ones(len(w1_rows), dtype=np.float64),
                (np.asarray(w1_rows), np.asarray(w1_cols)),
            ),
            shape=(self.input_dim, self.hidden_dim),
        )
        self.b1 = -np.ones(self.hidden_dim, dtype=np.float64)

        w2_rows = np.arange(self.hidden_dim, dtype=np.int64)
        w2_cols = np.tile(np.arange(nodes, dtype=np.int64), nodes)
        self.w2 = csr_matrix(
            (np.ones(self.hidden_dim, dtype=np.float64), (w2_rows, w2_cols)),
            shape=(self.hidden_dim, self.output_dim),
        )

    @property
    def dense_parameter_count(self) -> int:
        return (self.input_dim + 1) * self.hidden_dim + (self.hidden_dim + 1) * self.output_dim

    @property
    def nonzero_coefficient_count(self) -> int:
        return int(self.w1.nnz + np.count_nonzero(self.b1) + self.w2.nnz)

    def predict(self, encoded_table: np.ndarray, states: np.ndarray) -> np.ndarray:
        batch = len(states)
        inputs = np.zeros((batch, self.input_dim), dtype=np.float64)
        inputs[:, : self.table_dim] = encoded_table
        inputs[np.arange(batch), self.table_dim + states] = 1.0
        hidden = np.asarray(inputs @ self.w1) + self.b1
        np.maximum(hidden, 0.0, out=hidden)
        logits = np.asarray(hidden @ self.w2)
        return logits.argmax(axis=1).astype(np.int64)


def verify_arbitrary_maps(block: ExactPointerBlock, trials: int = 100, depth: int = 40) -> None:
    """Check exact composition on deterministic maps outside the cycle protocol."""
    rng = np.random.default_rng(20260713)
    eye = np.eye(block.nodes, dtype=np.float64)
    for _ in range(trials):
        pointer = rng.integers(0, block.nodes, size=block.nodes, dtype=np.int64)
        pointer[-1] = pointer[0]  # Guarantee a collision, so the map is not a permutation.
        encoded_table = eye[pointer].reshape(-1)
        predicted = np.arange(block.nodes, dtype=np.int64)
        target = predicted.copy()
        for _ in range(depth):
            predicted = block.predict(encoded_table, predicted)
            target = pointer[target]
            assert np.array_equal(predicted, target)


def evaluate_unrolled(pointer_raw: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(pointer_raw)
    locked = raw[(raw["write_rounds"] == 4) & (raw["L"] == raw["L"].max())].copy()
    nodes = int(json.loads(locked.iloc[0]["pointer"]).__len__())
    block = ExactPointerBlock(nodes)
    assert block.nonzero_coefficient_count == 4 * nodes * nodes
    verify_arbitrary_maps(block)
    eye = np.eye(nodes, dtype=np.float64)
    rows: list[dict[str, object]] = []

    for (seed, table_index, pointer_json), table_rows in locked.groupby(
        ["seed", "table_index", "pointer"], sort=True
    ):
        pointer = np.asarray(json.loads(pointer_json), dtype=np.int64)
        starts = table_rows["start_node"].to_numpy(dtype=np.int64)
        predicted = starts.copy()
        target = starts.copy()
        encoded_table = eye[pointer].reshape(-1)
        for depth in range(1, 41):
            predicted = block.predict(encoded_table, predicted)
            target = pointer[target]
            for start, prediction, expected in zip(starts, predicted, target):
                rows.append(
                    {
                        "seed": int(seed),
                        "table_index": int(table_index),
                        "start_node": int(start),
                        "L": depth,
                        "prediction": int(prediction),
                        "target": int(expected),
                        "accuracy": float(prediction == expected),
                        "unrolled_blocks": depth,
                        "dense_parameters": depth * block.dense_parameter_count,
                        "nonzero_coefficients": depth * block.nonzero_coefficient_count,
                    }
                )

    unrolled = pd.DataFrame(rows)
    assert bool((unrolled["accuracy"] == 1.0).all())
    unrolled.to_csv(output / "unrolled_mlp_unseen_pointer_raw.csv", index=False)

    ac = raw[raw["write_rounds"] == 4].groupby(["seed", "L", "t"], as_index=False).agg(
        complete_path_accuracy=("path_correct", "mean")
    )
    ac.to_csv(output / "ac_unseen_time_frontier_raw.csv", index=False)
    assert float(ac.loc[ac["t"] == 2 * ac["L"], "complete_path_accuracy"].min()) == 1.0

    metadata: dict[str, object] = {
        "task": "unseen full-cycle pointer tables matched to Chapter 6",
        "nodes": nodes,
        "depths": list(range(1, 41)),
        "seeds": sorted(int(value) for value in unrolled["seed"].unique()),
        "tables_per_seed": int(unrolled["table_index"].nunique()),
        "starts_per_table": int(unrolled.groupby(["seed", "table_index"]).start_node.nunique().iloc[0]),
        "mlp_block": {
            "input": f"flattened {nodes}x{nodes} table plus current-node one-hot",
            "hidden_units": block.hidden_dim,
            "activation": "ReLU",
            "output_units": nodes,
            "dense_parameters_per_hop": block.dense_parameter_count,
            "nonzero_coefficients_per_hop": block.nonzero_coefficient_count,
            "construction": "exact table-independent pointer lookup",
            "weight_sharing_between_hops": False,
            "arbitrary_non_permutation_audit": {
                "maps": 100,
                "starts_per_map": nodes,
                "depth": 40,
                "seed": 20260713,
            },
        },
        "ac_condition": "four episodic write passes; two internal updates per hop",
        "interpretation": "constructive upper baseline, not a lower bound for all feedforward networks",
    }
    (output / "unrolled_mlp_time_size_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return unrolled, ac, metadata


def frontier_frames(unrolled: pd.DataFrame, ac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mlp = (
        unrolled.groupby(["L", "dense_parameters"], as_index=False)
        .agg(accuracy=("accuracy", "mean"))
        .query("L in @DEPTH_MARKERS")
        .copy()
    )
    ac_frontier = (
        ac[ac["t"] == 2 * ac["L"]]
        .groupby(["L", "t"], as_index=False)
        .agg(accuracy=("complete_path_accuracy", "mean"))
        .query("L in @DEPTH_MARKERS")
        .copy()
    )
    return mlp, ac_frontier


def plot_mlp_size(mlp: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = mlp["dense_parameters"] / 1_000_000
    ax.plot(x, mlp["L"], color=ORANGE, linewidth=2.5, marker="o", markersize=6)
    ax.set(
        xlabel="Dense feedforward parameter slots (millions)",
        ylabel="Maximum supported pointer depth",
        xlim=(0, float(x.max()) * 1.07),
        ylim=(0, 43),
        title="Feedforward depth is instantiated in model size",
    )
    ax.text(0.02, 0.94, "Untied one-hop block per pointer hop", transform=ax.transAxes, color=MUTED, va="top")
    ax.annotate(
        f"Depth 40\n{float(x.max()):.1f}M dense parameters",
        (float(x.max()), 40),
        xytext=(-12, -38),
        textcoords="offset points",
        ha="right",
        color=ORANGE,
        fontweight="bold",
    )
    style_axis(ax)
    fig.tight_layout()
    save(fig, output, "mlp_size_extends_depth")


def plot_ac_time(ac: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(ac["t"], ac["L"], color=BLUE, linewidth=2.5, marker="o", markersize=6)
    ax.set(
        xlabel="AC internal updates $t$",
        ylabel="Maximum supported pointer depth",
        xlim=(0, float(ac["t"].max()) * 1.07),
        ylim=(0, 43),
        title="The fixed recurrent AC spends execution time",
    )
    ax.text(0.02, 0.94, "One frozen controller; two updates per pointer hop", transform=ax.transAxes, color=MUTED, va="top")
    ax.annotate(
        "Depth 40\n80 internal updates",
        (80, 40),
        xytext=(-12, -38),
        textcoords="offset points",
        ha="right",
        color=BLUE,
        fontweight="bold",
    )
    style_axis(ax)
    fig.tight_layout()
    save(fig, output, "ac_time_extends_depth")


def plot_matched_tradeoff(mlp: pd.DataFrame, ac: pd.DataFrame, output: Path, per_hop: int) -> None:
    matched = mlp.merge(ac[["L", "t"]], on="L", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(
        matched["t"],
        matched["dense_parameters"] / 1_000_000,
        color=TEAL,
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    ax.set(
        xlabel="AC internal updates for the same depth",
        ylabel="Unrolled dense parameter slots (millions)",
        xlim=(0, float(matched["t"].max()) * 1.07),
        ylim=(0, float(matched["dense_parameters"].max() / 1_000_000) * 1.08),
        title="Equal depth: recurrent time or feedforward size",
    )
    ax.text(
        0.02,
        0.94,
        f"Each additional hop: +2 AC updates or +{per_hop / 1_000_000:.2f}M dense parameter slots",
        transform=ax.transAxes,
        color=MUTED,
        va="top",
    )
    ax.annotate(
        "Depth 40",
        (80, float(matched["dense_parameters"].max() / 1_000_000)),
        xytext=(-12, -25),
        textcoords="offset points",
        ha="right",
        color=TEAL,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    style_axis(ax)
    fig.tight_layout()
    save(fig, output, "matched_ac_time_mlp_size")


def plot_combined(mlp: pd.DataFrame, ac: pd.DataFrame, output: Path, per_hop: int) -> None:
    matched = mlp.merge(ac[["L", "t"]], on="L", validate="one_to_one")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    axes[0].plot(mlp["dense_parameters"] / 1_000_000, mlp["L"], color=ORANGE, marker="o", linewidth=2.2)
    axes[0].set(xlabel="Dense parameter slots (millions)", ylabel="Supported depth", title="Feedforward: add size")
    axes[1].plot(ac["t"], ac["L"], color=BLUE, marker="o", linewidth=2.2)
    axes[1].set(xlabel="AC internal updates", ylabel="Supported depth", title="Recurrent: add time")
    axes[2].plot(matched["t"], matched["dense_parameters"] / 1_000_000, color=TEAL, marker="o", linewidth=2.2)
    axes[2].set(xlabel="AC internal updates", ylabel="Dense parameter slots (millions)", title="Matched depth")
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        style_axis(ax)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", fontweight="bold", color=INK)
    fig.suptitle(
        f"One more pointer hop: 2 recurrent updates or {per_hop / 1_000_000:.2f}M dense feedforward parameter slots",
        color=INK,
        fontweight="bold",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0)
    save(fig, output, "time_size_three_panel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    unrolled, ac, metadata = evaluate_unrolled(args.pointer_raw, args.output)
    mlp_frontier, ac_frontier = frontier_frames(unrolled, ac)
    per_hop = int(metadata["mlp_block"]["dense_parameters_per_hop"])
    plot_mlp_size(mlp_frontier, args.output)
    plot_ac_time(ac_frontier, args.output)
    plot_matched_tradeoff(mlp_frontier, ac_frontier, args.output, per_hop)
    plot_combined(mlp_frontier, ac_frontier, args.output, per_hop)
    print(mlp_frontier.to_string(index=False))
    print(ac_frontier.to_string(index=False))


if __name__ == "__main__":
    main()
