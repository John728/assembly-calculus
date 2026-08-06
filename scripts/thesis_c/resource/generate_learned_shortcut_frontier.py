"""Measure learned feed-forward shortcuts against recurrent pointer execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import t as student_t


EXPERIMENT_VERSION = 2
NODES = 50
DEPTH_BITS = 6
HORIZONS = (1, 5, 10, 20, 30, 40)
WIDTHS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32)
TABLE_SEEDS = tuple(range(42, 62))
RESTARTS = (0, 1, 2)
SUCCESS_THRESHOLD = 0.95
MAX_EPOCHS = 2500
EVALUATION_INTERVAL = 25
PATIENCE = 600
HOLDOUT_PER_DEPTH = 10

TEAL = "#008F7A"
CORAL = "#E45756"
INDIGO = "#3D4FA3"
CHARCOAL = "#202A33"
MID_GREY = "#64717D"
GRID = "#DDE4E8"
PALE = "#F4F7F8"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class FitRequest:
    table_seed: int
    horizon: int
    width: int
    restart: int
    holdout_per_depth: int = 0


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.9,
            "grid.linewidth": 0.8,
        }
    )


def full_cycle(nodes: int, seed: int) -> np.ndarray:
    list_seed = np.random.SeedSequence(seed).spawn(2)[0]
    rng = np.random.default_rng(list_seed)
    order = rng.permutation(nodes)
    pointer = np.empty(nodes, dtype=np.int64)
    pointer[order] = np.roll(order, -1)
    visited = [int(order[0])]
    for _ in range(1, nodes):
        visited.append(int(pointer[visited[-1]]))
    if len(set(visited)) != nodes or int(pointer[visited[-1]]) != visited[0]:
        raise AssertionError("pointer table is not one full cycle")
    return pointer


def encode_queries(
    pointer: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = len(pointer)
    eye = np.eye(nodes, dtype=np.float32)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    current = np.arange(nodes, dtype=np.int64)
    for depth in range(1, horizon + 1):
        current = pointer[current]
        bits = np.asarray(
            [(depth >> bit) & 1 for bit in range(DEPTH_BITS)],
            dtype=np.float32,
        )
        bits = bits * 2.0 - 1.0
        inputs.append(
            np.concatenate(
                [eye, np.broadcast_to(bits, (nodes, DEPTH_BITS))],
                axis=1,
            )
        )
        targets.append(current.copy())
        depths.append(np.full(nodes, depth, dtype=np.int64))
        starts.append(np.arange(nodes, dtype=np.int64))
    encoded_inputs = np.concatenate(inputs)
    encoded_targets = np.concatenate(targets)
    encoded_depths = np.concatenate(depths)
    encoded_starts = np.concatenate(starts)

    expected = encoded_starts.copy()
    for depth in range(1, horizon + 1):
        expected[encoded_depths == depth] = np.arange(nodes)
        for _ in range(depth):
            expected[encoded_depths == depth] = pointer[
                expected[encoded_depths == depth]
            ]
    if not np.array_equal(expected, encoded_targets):
        raise AssertionError("encoded targets disagree with pointer powers")
    return encoded_inputs, encoded_targets, encoded_depths, encoded_starts


class OneHiddenLayerMLP:
    def __init__(
        self,
        input_dim: int,
        width: int,
        output_dim: int,
        seed: int,
    ):
        rng = np.random.default_rng(seed)
        self.parameters = {
            "w1": rng.normal(
                0.0,
                np.sqrt(2.0 / input_dim),
                size=(input_dim, width),
            ).astype(np.float32),
            "b1": np.zeros(width, dtype=np.float32),
            "w2": rng.normal(
                0.0,
                np.sqrt(2.0 / width),
                size=(width, output_dim),
            ).astype(np.float32),
            "b2": np.zeros(output_dim, dtype=np.float32),
        }
        self.first_moment = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.second_moment = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.iteration = 0

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters.values())

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.maximum(
            inputs @ self.parameters["w1"] + self.parameters["b1"],
            0.0,
        )
        return hidden @ self.parameters["w2"] + self.parameters["b2"]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.logits(inputs).argmax(axis=1)

    def step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        hidden_pre = inputs @ self.parameters["w1"] + self.parameters["b1"]
        hidden = np.maximum(hidden_pre, 0.0)
        logits = hidden @ self.parameters["w2"] + self.parameters["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(
            probabilities[np.arange(len(targets)), targets] + 1e-12
        ).mean()

        output_gradient = probabilities
        output_gradient[np.arange(len(targets)), targets] -= 1.0
        output_gradient /= len(targets)
        gradients = {
            "w2": hidden.T @ output_gradient,
            "b2": output_gradient.sum(axis=0),
        }
        hidden_gradient = output_gradient @ self.parameters["w2"].T
        hidden_gradient[hidden_pre <= 0.0] = 0.0
        gradients["w1"] = inputs.T @ hidden_gradient
        gradients["b1"] = hidden_gradient.sum(axis=0)

        self.iteration += 1
        first_correction = 1.0 - 0.9**self.iteration
        second_correction = 1.0 - 0.999**self.iteration
        for name in self.parameters:
            self.first_moment[name] *= 0.9
            self.first_moment[name] += 0.1 * gradients[name]
            self.second_moment[name] *= 0.999
            self.second_moment[name] += 0.001 * gradients[name] ** 2
            first = self.first_moment[name] / first_correction
            second = self.second_moment[name] / second_correction
            self.parameters[name] -= (
                learning_rate * first / (np.sqrt(second) + 1e-8)
            )
        return float(loss)


def optimiser_seed(request: FitRequest) -> int:
    state = np.random.SeedSequence(
        [
            20260727,
            request.table_seed,
            request.horizon,
            request.width,
            request.restart,
            request.holdout_per_depth,
        ]
    ).generate_state(1)
    return int(state[0])


def split_queries(
    depths: np.ndarray,
    request: FitRequest,
) -> np.ndarray:
    training = np.ones(len(depths), dtype=bool)
    if not request.holdout_per_depth:
        return training
    rng = np.random.default_rng(optimiser_seed(request) + 1)
    for depth in range(2, request.horizon + 1):
        indices = np.flatnonzero(depths == depth)
        held_out = rng.choice(
            indices,
            size=request.holdout_per_depth,
            replace=False,
        )
        training[held_out] = False
    return training


def depth_accuracies(
    predictions: np.ndarray,
    targets: np.ndarray,
    depths: np.ndarray,
    mask: np.ndarray,
    horizon: int,
) -> list[float]:
    values: list[float] = []
    for depth in range(1, horizon + 1):
        selected = mask & (depths == depth)
        if np.any(selected):
            values.append(float(np.mean(predictions[selected] == targets[selected])))
        else:
            values.append(float("nan"))
    return values


def fit(request: FitRequest) -> tuple[dict[str, object], list[dict[str, object]]]:
    pointer = full_cycle(NODES, request.table_seed)
    inputs, targets, depths, starts = encode_queries(pointer, request.horizon)
    training = split_queries(depths, request)
    evaluation = ~training
    model = OneHiddenLayerMLP(
        input_dim=inputs.shape[1],
        width=request.width,
        output_dim=NODES,
        seed=optimiser_seed(request),
    )

    best_mean_accuracy = -1.0
    stagnant_epochs = 0
    loss = float("nan")
    started = time.monotonic()
    for epoch in range(1, MAX_EPOCHS + 1):
        learning_rate = 0.01 if epoch <= 1500 else 0.003
        loss = model.step(inputs[training], targets[training], learning_rate)
        if epoch % EVALUATION_INTERVAL:
            continue
        predictions = model.predict(inputs)
        training_by_depth = depth_accuracies(
            predictions,
            targets,
            depths,
            training,
            request.horizon,
        )
        finite_training = np.asarray(
            [value for value in training_by_depth if np.isfinite(value)]
        )
        mean_accuracy = float(finite_training.mean())
        minimum_accuracy = float(finite_training.min())
        if mean_accuracy > best_mean_accuracy + 1e-5:
            best_mean_accuracy = mean_accuracy
            stagnant_epochs = 0
        else:
            stagnant_epochs += EVALUATION_INTERVAL
        target = 0.99 if request.holdout_per_depth else SUCCESS_THRESHOLD
        if minimum_accuracy >= target or stagnant_epochs >= PATIENCE:
            break

    predictions = model.predict(inputs)
    training_by_depth = depth_accuracies(
        predictions,
        targets,
        depths,
        training,
        request.horizon,
    )
    evaluation_by_depth = depth_accuracies(
        predictions,
        targets,
        depths,
        evaluation,
        request.horizon,
    )
    finite_training = np.asarray(
        [value for value in training_by_depth if np.isfinite(value)]
    )
    finite_evaluation = np.asarray(
        [value for value in evaluation_by_depth if np.isfinite(value)]
    )
    summary = {
        "table_seed": request.table_seed,
        "horizon": request.horizon,
        "width": request.width,
        "restart": request.restart,
        "holdout_per_depth": request.holdout_per_depth,
        "parameters": model.parameter_count,
        "epochs": epoch,
        "elapsed_seconds": time.monotonic() - started,
        "final_loss": loss,
        "training_accuracy": float(
            np.mean(predictions[training] == targets[training])
        ),
        "minimum_training_depth_accuracy": float(finite_training.min()),
        "evaluation_accuracy": (
            float(np.mean(predictions[evaluation] == targets[evaluation]))
            if np.any(evaluation)
            else float("nan")
        ),
        "minimum_evaluation_depth_accuracy": (
            float(finite_evaluation.min())
            if finite_evaluation.size
            else float("nan")
        ),
        "success": int(
            not request.holdout_per_depth
            and finite_training.min() >= SUCCESS_THRESHOLD
        ),
    }
    per_depth = []
    for index, depth in enumerate(range(1, request.horizon + 1)):
        per_depth.append(
            {
                **{
                    key: summary[key]
                    for key in (
                        "table_seed",
                        "horizon",
                        "width",
                        "restart",
                        "holdout_per_depth",
                        "parameters",
                    )
                },
                "depth": depth,
                "training_accuracy": training_by_depth[index],
                "evaluation_accuracy": evaluation_by_depth[index],
                "training_examples": int(np.sum(training & (depths == depth))),
                "evaluation_examples": int(
                    np.sum(evaluation & (depths == depth))
                ),
                "pointer": json.dumps(pointer.tolist()),
                "starts": json.dumps(
                    starts[depths == depth].astype(int).tolist()
                ),
            }
        )
    return summary, per_depth


def parameter_count(width: int) -> int:
    return (NODES + DEPTH_BITS + 1) * width + (width + 1) * NODES


def config_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "nodes": NODES,
        "depth_bits": DEPTH_BITS,
        "horizons": list(HORIZONS),
        "widths": list(WIDTHS),
        "table_seeds": list(TABLE_SEEDS),
        "restarts": list(RESTARTS),
        "success_threshold": SUCCESS_THRESHOLD,
        "maximum_epochs": MAX_EPOCHS,
        "evaluation_interval": EVALUATION_INTERVAL,
        "patience": PATIENCE,
        "holdout_per_depth": HOLDOUT_PER_DEPTH,
    }


def config_hash(args: argparse.Namespace) -> str:
    encoded = json.dumps(
        config_payload(args),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frontier_checkpoint(
    checkpoint_dir: Path,
    table_seed: int,
    horizon: int,
) -> Path:
    return checkpoint_dir / f"frontier_table_{table_seed}_h_{horizon}.json"


def run_frontier_task(
    output: Path,
    args: argparse.Namespace,
    table_seed: int,
    horizon: int,
) -> dict[str, object]:
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = frontier_checkpoint(checkpoint_dir, table_seed, horizon)
    expected_hash = config_hash(args)
    if checkpoint.exists() and not args.force:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("config_hash") == expected_hash:
            return payload

    summaries: list[dict[str, object]] = []
    per_depth: list[dict[str, object]] = []
    robust_width = None
    for width in WIDTHS:
        width_summaries = []
        for restart in RESTARTS:
            summary, depth_rows = fit(
                FitRequest(
                    table_seed=table_seed,
                    horizon=horizon,
                    width=width,
                    restart=restart,
                )
            )
            width_summaries.append(summary)
            summaries.append(summary)
            per_depth.extend(depth_rows)
        if sum(int(row["success"]) for row in width_summaries) >= 2:
            robust_width = width
            break
    if robust_width is None:
        raise RuntimeError(
            f"no registered MLP solved table={table_seed}, horizon={horizon}"
        )
    payload = {
        "config_hash": expected_hash,
        "table_seed": table_seed,
        "horizon": horizon,
        "robust_width": robust_width,
        "robust_parameters": parameter_count(robust_width),
        "summaries": summaries,
        "per_depth": per_depth,
    }
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(checkpoint)
    return payload


def run_frontier(
    output: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_seeds = TABLE_SEEDS[: args.table_limit]
    tasks = [
        (table_seed, horizon)
        for table_seed in table_seeds
        for horizon in HORIZONS
    ]
    payloads = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_frontier_task,
                output,
                args,
                table_seed,
                horizon,
            ): (table_seed, horizon)
            for table_seed, horizon in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payloads.append(future.result())
            if index % 10 == 0 or index == len(tasks):
                print(f"frontier {index}/{len(tasks)}", flush=True)
    summaries = pd.DataFrame(
        row for payload in payloads for row in payload["summaries"]
    )
    per_depth = pd.DataFrame(
        row for payload in payloads for row in payload["per_depth"]
    )
    frontier = pd.DataFrame(
        {
            "table_seed": payload["table_seed"],
            "horizon": payload["horizon"],
            "width": payload["robust_width"],
            "parameters": payload["robust_parameters"],
        }
        for payload in payloads
    ).sort_values(["table_seed", "horizon"])
    return summaries, per_depth, frontier


def holdout_checkpoint(output: Path, request: FitRequest) -> Path:
    return (
        output
        / "checkpoints"
        / (
            f"holdout_table_{request.table_seed}_h_{request.horizon}"
            f"_w_{request.width}_r_{request.restart}.json"
        )
    )


def run_holdout(
    output: Path,
    frontier: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = frontier[frontier["horizon"] == max(HORIZONS)]
    requests = [
        FitRequest(
            table_seed=int(row.table_seed),
            horizon=int(row.horizon),
            width=int(row.width),
            restart=restart,
            holdout_per_depth=HOLDOUT_PER_DEPTH,
        )
        for row in selected.itertuples()
        for restart in RESTARTS[: args.holdout_restarts]
    ]

    def execute(request: FitRequest):
        checkpoint = holdout_checkpoint(output, request)
        expected_hash = config_hash(args)
        if checkpoint.exists() and not args.force:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("config_hash") == expected_hash:
                return payload
        summary, per_depth = fit(request)
        payload = {
            "config_hash": expected_hash,
            "summary": summary,
            "per_depth": per_depth,
        }
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(checkpoint)
        return payload

    payloads = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute, request): request for request in requests
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payloads.append(future.result())
            if index % 10 == 0 or index == len(requests):
                print(f"holdout {index}/{len(requests)}", flush=True)
    summaries = pd.DataFrame(payload["summary"] for payload in payloads)
    per_depth = pd.DataFrame(
        row for payload in payloads for row in payload["per_depth"]
    )
    return summaries, per_depth


def mean_t_interval(values: pd.Series) -> tuple[float, float]:
    data = values.dropna().to_numpy(dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, 0.0
    half = (
        float(student_t.ppf(0.975, len(data) - 1))
        * float(np.std(data, ddof=1))
        / np.sqrt(len(data))
    )
    return mean, half


def save_square(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.set_size_inches(6.4, 6.4, forward=True)
    for extension in ("pdf", "svg"):
        fig.savefig(
            output / f"{stem}.{extension}",
            facecolor=WHITE,
            metadata={"Title": stem, "Creator": "Matplotlib"},
        )
    fig.savefig(output / f"{stem}.png", dpi=400, facecolor=WHITE)
    plt.close(fig)


def thousands(value: float, _position: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def plot_frontier(frontier: pd.DataFrame, output: Path) -> pd.DataFrame:
    summary = (
        frontier.groupby("horizon", as_index=False)
        .agg(
            median_parameters=("parameters", "median"),
            lower_parameters=("parameters", lambda values: values.quantile(0.25)),
            upper_parameters=("parameters", lambda values: values.quantile(0.75)),
            minimum_parameters=("parameters", "min"),
            maximum_parameters=("parameters", "max"),
            median_width=("width", "median"),
        )
        .sort_values("horizon")
    )
    horizons = summary["horizon"].to_numpy(dtype=float)
    median = summary["median_parameters"].to_numpy(dtype=float)
    lower = summary["lower_parameters"].to_numpy(dtype=float)
    upper = summary["upper_parameters"].to_numpy(dtype=float)
    unrolled = horizons * (NODES * NODES + NODES)

    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.15, top=0.82)
    for table_seed, subset in frontier.groupby("table_seed"):
        axis.plot(
            subset["horizon"],
            subset["parameters"],
            color=CORAL,
            alpha=0.10,
            linewidth=0.8,
            zorder=1,
        )
    axis.fill_between(
        horizons,
        lower,
        upper,
        color=CORAL,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    axis.plot(
        horizons,
        median,
        color=CORAL,
        linewidth=3.0,
        marker="o",
        markersize=6.5,
        markeredgecolor=WHITE,
        markeredgewidth=1.1,
        zorder=4,
    )
    axis.plot(
        horizons,
        unrolled,
        color=MID_GREY,
        linestyle=(0, (5, 4)),
        linewidth=2.2,
        zorder=2,
    )
    for horizon, value in zip(horizons, median):
        axis.annotate(
            f"{int(round(value)):,}",
            (horizon, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=CORAL,
            fontsize=9.2,
            fontweight="semibold",
        )

    final_median = int(round(median[-1]))
    final_unrolled = int(round(unrolled[-1]))
    ratio = final_unrolled / final_median
    axis.annotate(
        f"Depth 40: {ratio:.0f}× below unrolling",
        xy=(horizons[-1], median[-1]),
        xytext=(20, 10_500),
        arrowprops={
            "arrowstyle": "-|>",
            "color": CORAL,
            "lw": 1.6,
            "connectionstyle": "arc3,rad=-0.16",
        },
        color=CORAL,
        fontsize=11,
        fontweight="semibold",
        ha="left",
    )
    axis.text(
        0.29,
        0.055,
        "Fixed AC architecture\nuses $t=L$ updates",
        transform=axis.transAxes,
        color=TEAL,
        fontsize=11,
        fontweight="semibold",
    )
    axis.text(
        0.03,
        0.96,
        (
            "One MLP per table answers every depth up to $L$\n"
            f"Median and IQR across {frontier['table_seed'].nunique()} tables"
        ),
        transform=axis.transAxes,
        color=MID_GREY,
        fontsize=9.2,
        va="top",
    )
    axis.text(
        21.5,
        66_000,
        "Explicit one-hop unrolling",
        color=MID_GREY,
        fontsize=10.5,
        fontweight="semibold",
        rotation=13,
        rotation_mode="anchor",
    )
    axis.text(
        16.0,
        2_200,
        "Learned shortcut MLP",
        color=CORAL,
        fontsize=10.5,
        fontweight="semibold",
    )
    axis.set(
        xlim=(0, 42),
        ylim=(200, 180_000),
        xlabel="Supported pointer depth $L$",
        ylabel="Smallest tested MLP parameters",
        xticks=HORIZONS,
    )
    axis.set_yscale("log")
    axis.set_yticks((250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000))
    axis.yaxis.set_major_formatter(FuncFormatter(thousands))
    top = axis.secondary_xaxis("top")
    top.set_xticks(HORIZONS)
    top.set_xticklabels([str(value) for value in HORIZONS])
    top.set_xlabel("AC internal updates $t=L$", labelpad=7)
    top.xaxis.label.set_color(TEAL)
    top.tick_params(length=0, pad=3, colors=TEAL)
    save_square(fig, output, "poster_time_size_learned_shortcuts")
    return summary


def plot_direct_time_size(frontier: pd.DataFrame, output: Path) -> None:
    summary = (
        frontier.groupby("horizon", as_index=False)
        .agg(
            median_parameters=("parameters", "median"),
            lower_parameters=("parameters", lambda values: values.quantile(0.25)),
            upper_parameters=("parameters", lambda values: values.quantile(0.75)),
        )
        .sort_values("horizon")
    )
    updates = summary["horizon"].to_numpy(dtype=float)
    median = summary["median_parameters"].to_numpy(dtype=float)
    lower = summary["lower_parameters"].to_numpy(dtype=float)
    upper = summary["upper_parameters"].to_numpy(dtype=float)

    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.16, top=0.95)
    axis.fill_between(
        updates,
        lower,
        upper,
        color=CORAL,
        alpha=0.17,
        linewidth=0,
        zorder=2,
    )
    axis.plot(
        updates,
        median,
        color=CORAL,
        linewidth=3.2,
        marker="o",
        markersize=7.0,
        markeredgecolor=WHITE,
        markeredgewidth=1.2,
        zorder=4,
    )

    axis.set(
        xlim=(0, 42),
        ylim=(0, 2_150),
        xlabel="AC internal updates $t$",
        ylabel="MLP parameters for the same depth",
        xticks=HORIZONS,
        yticks=(0, 500, 1_000, 1_500, 2_000),
    )
    axis.xaxis.label.set_color(TEAL)
    axis.yaxis.label.set_color(CORAL)
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{int(value):,}")
    )
    save_square(fig, output, "poster_ac_time_vs_mlp_size")


def plot_holdout(
    summaries: pd.DataFrame,
    output: Path,
) -> dict[str, float]:
    train_mean, train_half = mean_t_interval(summaries["training_accuracy"])
    test_mean, test_half = mean_t_interval(summaries["evaluation_accuracy"])
    values = [train_mean, test_mean]
    intervals = [train_half, test_half]
    colours = [CORAL, INDIGO]

    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.18, top=0.80)
    positions = np.arange(2)
    bars = axis.bar(
        positions,
        values,
        yerr=intervals,
        width=0.58,
        color=colours,
        edgecolor=WHITE,
        linewidth=1.0,
        capsize=5,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )
    for bar, value in zip(bars, values):
        inside = value > 0.90
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.055 if inside else value + 0.035,
            f"{value:.1%}",
            ha="center",
            color=WHITE if inside else CHARCOAL,
            fontsize=12,
            fontweight="semibold",
        )
    axis.axhline(
        1.0,
        color=TEAL,
        linewidth=2.2,
    )
    axis.set(
        ylim=(0, 1.18),
        ylabel="Final-node accuracy",
        xticks=positions,
        xticklabels=("Directly supervised\nqueries", "Held-out\nstart/depth pairs"),
    )
    axis.set_title("Shortcut fit does not guarantee reuse", pad=18)
    axis.text(
        0.02,
        0.96,
        (
            "Depth 40; 10 held-out starts at every depth 2-40; "
            f"{summaries['table_seed'].nunique()} tables"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=9.5,
        color=MID_GREY,
    )
    axis.text(
        1.43,
        1.025,
        "AC: 100% from one-hop training",
        color=TEAL,
        fontsize=10,
        fontweight="semibold",
        ha="right",
    )
    save_square(fig, output, "learned_shortcut_holdout_diagnostic")
    return {
        "training_mean": train_mean,
        "training_95_t_half_width": train_half,
        "holdout_mean": test_mean,
        "holdout_95_t_half_width": test_half,
    }


def git_state(root: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "scripts/thesis_c", "pyac/src/pyac"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "revision": revision,
        "relevant_paths_dirty": bool(status.strip()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reference_results(ac_root: Path) -> dict[str, object]:
    source = (
        ac_root
        / "results"
        / "thesis_c"
        / "resource"
        / "seen_time_size_raw.csv"
    )
    frame = pd.read_csv(source)
    required = {
        "seed",
        "L",
        "pointer",
        "ac_correct",
        "ac_path_correct",
        "ac_updates",
        "mlp_path_correct",
        "mlp_dense_parameter_slots",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"reference results lack {sorted(required - set(frame.columns))}")
    if set(frame["seed"].astype(int).unique()) != set(TABLE_SEEDS):
        raise ValueError("reference table seeds do not match shortcut experiment")
    if not (
        frame["ac_correct"].eq(1).all()
        and frame["ac_path_correct"].eq(1).all()
        and frame["mlp_path_correct"].eq(1).all()
    ):
        raise ValueError("reference comparison contains an inexact path")
    if not frame["ac_updates"].eq(frame["L"]).all():
        raise ValueError("reference AC does not use one update per hop")
    expected_slots = frame["L"] * (NODES * NODES + NODES)
    if not frame["mlp_dense_parameter_slots"].eq(expected_slots).all():
        raise ValueError("reference unrolling does not use 2,550 slots per hop")
    for seed in TABLE_SEEDS:
        values = frame.loc[frame["seed"] == seed, "pointer"].unique()
        if len(values) != 1:
            raise ValueError(f"seed {seed} has multiple pointer tables")
        stored = np.asarray(json.loads(values[0]), dtype=np.int64)
        if not np.array_equal(stored, full_cycle(NODES, seed)):
            raise ValueError(f"seed {seed} does not reproduce the stored table")
    return {
        "path": str(source),
        "sha256": sha256(source),
        "rows": int(len(frame)),
        "all_ac_paths_exact": True,
        "all_unrolled_paths_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--table-limit", type=int, default=10)
    parser.add_argument("--holdout-restarts", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.table_limit <= len(TABLE_SEEDS):
        parser.error(f"--table-limit must be between 1 and {len(TABLE_SEEDS)}")
    if not 1 <= args.holdout_restarts <= len(RESTARTS):
        parser.error(
            f"--holdout-restarts must be between 1 and {len(RESTARTS)}"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    configure_plotting()
    reference_results = validate_reference_results(args.ac_root)
    fit_summary, fit_depth, frontier = run_frontier(args.output, args)
    holdout_summary, holdout_depth = run_holdout(args.output, frontier, args)
    frontier_summary = plot_frontier(frontier, args.output)
    plot_direct_time_size(frontier, args.output)
    holdout_statistics = plot_holdout(holdout_summary, args.output)

    fit_summary.to_csv(args.output / "learned_shortcut_fit_summary.csv", index=False)
    fit_depth.to_csv(
        args.output / "learned_shortcut_fit_by_depth.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    frontier.to_csv(args.output / "learned_shortcut_frontier_by_table.csv", index=False)
    frontier_summary.to_csv(
        args.output / "learned_shortcut_frontier_summary.csv",
        index=False,
    )
    holdout_summary.to_csv(
        args.output / "learned_shortcut_holdout_summary.csv",
        index=False,
    )
    holdout_depth.to_csv(
        args.output / "learned_shortcut_holdout_by_depth.csv",
        index=False,
    )

    metadata = {
        **config_payload(args),
        "evaluated_table_seeds": list(TABLE_SEEDS[: args.table_limit]),
        "holdout_restarts_evaluated": list(
            RESTARTS[: args.holdout_restarts]
        ),
        "protocol": (
            "table-specific one-hidden-layer MLP jointly supports every "
            "start/depth query up to each horizon"
        ),
        "input": (
            "50-way one-hot start node plus fixed six-bit signed depth code"
        ),
        "target": "final node M^L(i)",
        "training_supervision": (
            "all 50 start/depth answers at every supported depth"
        ),
        "frontier_rule": (
            "smallest registered width for which at least two of three "
            "restarts achieve >=0.95 accuracy at every depth in the prefix"
        ),
        "explicit_unrolling_reference": {
            "dense_slots_per_hop": NODES * NODES + NODES,
            "depth_40_slots": 40 * (NODES * NODES + NODES),
            "validated_results": reference_results,
        },
        "holdout_diagnostic": {
            "selected_width": (
                "per-table robust width selected by the fully supervised "
                "depth-40 frontier"
            ),
            "depth_one_queries": "all retained for table exposure",
            "held_out_queries": (
                "10 of 50 start nodes independently held out at each depth 2-40"
            ),
            **holdout_statistics,
        },
        "limitations": [
            "The learned frontier is for the registered one-hidden-layer MLP family, not all feedforward networks.",
            "The main frontier measures fitted seen-table support and uses direct supervision at every supported depth.",
            "MLP parameters, AC synapses, and AC internal updates are not interchangeable physical units.",
        ],
        "software": git_state(args.ac_root),
    }
    metadata["candidate_table_seeds"] = metadata.pop("table_seeds")
    metadata["frontier_restarts"] = metadata.pop("restarts")
    metadata["fit_configuration_sha256"] = config_hash(args)
    (args.output / "learned_shortcut_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
