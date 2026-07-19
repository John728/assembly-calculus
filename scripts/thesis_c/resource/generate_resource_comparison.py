from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import t as student_t


SEEDS = (42, 43, 44, 45, 46)
WIDTHS = (64, 128, 256, 512)
COLOURS = ("#0077BB", "#EE7733", "#009988", "#CC3311")


def configure_plotting() -> None:
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


def generate_cycles(seed: int | np.random.SeedSequence, count: int = 10, n: int = 10) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pointers = []
    for _ in range(count):
        order = rng.permutation(n)
        pointer = np.zeros(n, dtype=np.int64)
        pointer[order] = np.roll(order, -1)
        pointers.append(pointer)
    return pointers


def encode(pointers: list[np.ndarray], max_hop: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(pointers[0])
    eye = np.eye(n, dtype=np.float64)
    inputs, targets, depths = [], [], []
    for pointer in pointers:
        encoded_table = eye[pointer].reshape(-1)
        for start in range(n):
            node = start
            for depth in range(1, max_hop + 1):
                node = int(pointer[node])
                inputs.append(
                    np.concatenate([encoded_table, eye[start], np.asarray([depth / max_hop])])
                )
                targets.append(node)
                depths.append(depth)
    return np.stack(inputs), np.asarray(targets), np.asarray(depths)


class OneHiddenLayerMLP:
    def __init__(self, input_dim: int, width: int, output_dim: int, rng: np.random.Generator):
        self.parameters = {
            "w1": rng.normal(0, np.sqrt(2 / input_dim), (input_dim, width)),
            "b1": np.zeros(width),
            "w2": rng.normal(0, np.sqrt(2 / width), (width, output_dim)),
            "b2": np.zeros(output_dim),
        }
        self.first_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.second_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters.values())

    def step(self, inputs: np.ndarray, targets: np.ndarray, iteration: int, learning_rate: float = 0.01) -> float:
        hidden_pre = inputs @ self.parameters["w1"] + self.parameters["b1"]
        hidden = np.maximum(hidden_pre, 0)
        logits = hidden @ self.parameters["w2"] + self.parameters["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean()

        output_gradient = probabilities
        output_gradient[np.arange(len(targets)), targets] -= 1
        output_gradient /= len(targets)
        gradients = {
            "w2": hidden.T @ output_gradient,
            "b2": output_gradient.sum(axis=0),
        }
        hidden_gradient = output_gradient @ self.parameters["w2"].T
        hidden_gradient[hidden_pre <= 0] = 0
        gradients["w1"] = inputs.T @ hidden_gradient
        gradients["b1"] = hidden_gradient.sum(axis=0)

        for name in self.parameters:
            self.first_moment[name] = 0.9 * self.first_moment[name] + 0.1 * gradients[name]
            self.second_moment[name] = 0.999 * self.second_moment[name] + 0.001 * gradients[name] ** 2
            first = self.first_moment[name] / (1 - 0.9**iteration)
            second = self.second_moment[name] / (1 - 0.999**iteration)
            self.parameters[name] -= learning_rate * first / (np.sqrt(second) + 1e-8)
        return float(loss)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.maximum(inputs @ self.parameters["w1"] + self.parameters["b1"], 0)
        return (hidden @ self.parameters["w2"] + self.parameters["b2"]).argmax(axis=1)


def run_mlp() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        list_seed = np.random.SeedSequence(seed).spawn(3)[0]
        inputs, targets, depths = encode(generate_cycles(list_seed))
        training = depths <= 4
        for width in WIDTHS:
            model = OneHiddenLayerMLP(inputs.shape[1], width, 10, np.random.default_rng(seed + width))
            loss = 0.0
            for epoch in range(1, 601):
                loss = model.step(inputs[training], targets[training], epoch)
            predictions = model.predict(inputs)
            for depth in range(1, 9):
                mask = depths == depth
                rows.append(
                    {
                        "seed": seed,
                        "width": width,
                        "parameters": model.parameter_count,
                        "L": depth,
                        "accuracy": float(np.mean(predictions[mask] == targets[mask])),
                        "trained_depth": depth <= 4,
                        "final_training_loss": loss,
                    }
                )
    return pd.DataFrame(rows)


def mean_interval(values: pd.Series) -> tuple[float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return mean, 0.0
    half = float(student_t.ppf(0.975, len(data) - 1)) * float(data.std(ddof=1)) / np.sqrt(len(data))
    return mean, half


def plot_accuracy(mlp: pd.DataFrame, pointer_raw: Path, output: Path) -> None:
    pointer = pd.read_csv(pointer_raw)
    pointer = pointer[pointer["t"] == pointer["L"]]
    pointer_seed = pointer.groupby(["seed", "L"], as_index=False).agg(accuracy=("accuracy", "mean"))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    axes[0].axvspan(0.5, 4.5, color="0.92", zorder=0)
    axes[0].axvline(4.5, color="0.45", lw=0.9, ls=":")
    axes[0].axhline(0.1, color="0.4", lw=0.8, ls=":")
    for colour, width in zip(COLOURS, WIDTHS):
        subset = mlp[mlp["width"] == width]
        means, intervals = [], []
        for depth in range(1, 9):
            mean, interval = mean_interval(subset.loc[subset["L"] == depth, "accuracy"])
            means.append(mean)
            intervals.append(interval)
        axes[0].errorbar(
            range(1, 9), means, yerr=intervals, marker="o", ms=3.1, lw=1.4,
            capsize=2.0, elinewidth=0.8, color=colour, label=f"Width {width}",
        )
    axes[0].text(2.5, 0.05, "Training depths", ha="center", color="0.3", fontsize=8)
    axes[0].set(xlabel="Pointer depth $L$", ylabel="Accuracy", ylim=(-0.02, 1.02), title="Fixed-depth MLP")
    axes[0].legend(frameon=False, ncol=2, loc="center left")

    means, intervals = [], []
    for depth in range(1, 9):
        mean, interval = mean_interval(pointer_seed.loc[pointer_seed["L"] == depth, "accuracy"])
        means.append(mean)
        intervals.append(interval)
    axes[1].errorbar(
        range(1, 9), means, yerr=intervals, marker="o", ms=3.4, lw=1.6,
        capsize=2.0, color="#0077BB",
    )
    axes[1].set(xlabel="Pointer depth $L$", title="Recurrent AC, $t=L$")
    axes[0].text(-0.17, 1.08, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.17, 1.08, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.8)
    save(fig, output, "mlp_ac_depth_accuracy")


def plot_frontier(mlp: pd.DataFrame, pointer_raw: Path, output: Path, threshold: float = 0.9) -> None:
    pointer = pd.read_csv(pointer_raw)
    pointer_summary = pointer.groupby(["L", "t"], as_index=False).agg(accuracy=("accuracy", "mean"))
    mlp_summary = mlp.groupby(["width", "parameters", "L"], as_index=False).agg(accuracy=("accuracy", "mean"))
    mlp_frontier = []
    for (width, parameters), subset in mlp_summary.groupby(["width", "parameters"]):
        reliable = subset.loc[subset["accuracy"] >= threshold, "L"]
        mlp_frontier.append((int(parameters), int(reliable.max()) if not reliable.empty else 0, int(width)))
    mlp_frontier.sort()

    budgets = sorted(pointer_summary["t"].astype(int).unique())
    ac_frontier = []
    for budget in budgets:
        reliable = pointer_summary.loc[
            (pointer_summary["t"] == budget) & (pointer_summary["accuracy"] >= threshold), "L"
        ]
        ac_frontier.append(int(reliable.max()) if not reliable.empty else 0)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True)
    parameters = [row[0] for row in mlp_frontier]
    depths = [row[1] for row in mlp_frontier]
    axes[0].plot(parameters, depths, marker="o", lw=1.5, color="#CC3311")
    for parameter, depth, width in mlp_frontier:
        axes[0].annotate(str(width), (parameter, depth), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=7)
    axes[0].set(xlabel="MLP parameters", ylabel="Maximum reliable depth", ylim=(-0.2, 8.5))
    axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    axes[1].plot(budgets, ac_frontier, marker="o", lw=1.5, color="#0077BB")
    axes[1].set(xlabel="AC internal update budget $t$")
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, fontweight="bold", va="top")
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.0)
    save(fig, output, "time_size_frontier")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    mlp = run_mlp()
    mlp.to_csv(args.output / "mlp_seen_pointer_raw.csv", index=False)
    metadata = {
        "seeds": list(SEEDS),
        "pointer_tables_per_seed": 10,
        "nodes_per_table": 10,
        "training_depths": [1, 2, 3, 4],
        "evaluation_depths": list(range(1, 9)),
        "widths": list(WIDTHS),
        "input_dimension": 111,
        "training_examples_per_seed": 400,
        "optimiser": "full-batch Adam",
        "updates": 600,
        "learning_rate": 0.01,
        "table_seed_protocol": "SeedSequence(seed).spawn(3)[0], matching the AC list seed",
    }
    (args.output / "mlp_seen_pointer_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_accuracy(mlp, args.pointer_raw, args.output)
    plot_frontier(mlp, args.pointer_raw, args.output)


if __name__ == "__main__":
    main()
