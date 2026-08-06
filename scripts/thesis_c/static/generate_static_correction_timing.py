from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as student_t


TEAL = "#009988"
INDIGO = "#332288"
CORAL = "#EE7733"
GRID = "#DDE4E8"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.8,
            "grid.linewidth": 0.7,
        }
    )


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half = (
        float(student_t.ppf(0.975, len(values) - 1))
        * float(np.std(values, ddof=1))
        / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.trajectories)
    required = {
        "seed",
        "condition",
        "transition_type",
        "switch_count",
        "settling_readout",
        "initial_target_gap",
        "gain_advantage",
        "overtaking_difficulty",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "trajectory data is missing columns: "
            + ", ".join(sorted(missing))
        )

    corrected = frame[
        frame["condition"].eq("distinct_50")
        & frame["transition_type"].eq("corrected")
    ].copy()
    if corrected.empty:
        raise ValueError("trajectory data has no corrected cases")
    if not corrected["gain_advantage"].gt(0.0).all():
        raise ValueError("a corrected case lacks positive gain advantage")
    if not corrected["switch_count"].eq(1).all():
        raise ValueError("a corrected case changes class more than once")

    bin_edges = (-np.inf, 0.25, 0.50, np.inf)
    bin_keys = ("low", "medium", "high")
    bin_labels = {
        "low": r"Low ($\chi<0.25$)",
        "medium": r"Medium ($0.25\leq\chi<0.5$)",
        "high": r"High ($\chi\geq0.5$)",
    }
    colours = {"low": TEAL, "medium": INDIGO, "high": CORAL}
    corrected["difficulty_bin"] = pd.cut(
        corrected["overtaking_difficulty"],
        bins=bin_edges,
        labels=bin_keys,
        right=False,
    )
    if corrected["difficulty_bin"].isna().any():
        raise ValueError("an overtaking difficulty falls outside the bins")

    readouts = np.arange(
        int(corrected["settling_readout"].min()),
        int(corrected["settling_readout"].max()) + 1,
    )
    counts = (
        corrected["settling_readout"]
        .value_counts()
        .reindex(readouts, fill_value=0)
        .sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9))
    axes[0].bar(
        readouts,
        counts.to_numpy(),
        color=TEAL,
        width=0.72,
        zorder=3,
    )
    for readout, count in zip(readouts, counts.to_numpy()):
        axes[0].text(
            readout,
            count + 1.1,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set(
        xlabel="Correction readout",
        ylabel="New corrections",
        xticks=readouts,
        ylim=(0, max(counts) * 1.16),
    )
    axes[0].grid(axis="x", visible=False)

    cumulative_readouts = np.arange(1, int(readouts[-1]) + 1)
    bin_summaries: dict[str, object] = {}
    for key in bin_keys:
        subset = corrected[corrected["difficulty_bin"].eq(key)]
        cumulative = np.asarray(
            [
                float(np.mean(subset["settling_readout"].le(readout)))
                for readout in cumulative_readouts
            ]
        )
        axes[1].plot(
            cumulative_readouts,
            cumulative,
            color=colours[key],
            linewidth=1.8,
            marker="o",
            markersize=3.8,
            label=bin_labels[key],
        )
        bin_summaries[key] = {
            "definition": bin_labels[key],
            "trajectories": int(len(subset)),
            "mean_correction_readout": float(
                subset["settling_readout"].mean()
            ),
            "median_correction_readout": float(
                subset["settling_readout"].median()
            ),
            "cumulative_fraction": {
                str(readout): float(value)
                for readout, value in zip(cumulative_readouts, cumulative)
            },
        }
    axes[1].set(
        xlabel="Readout",
        ylabel="Corrected by this readout",
        xlim=(1, int(readouts[-1])),
        ylim=(-0.02, 1.02),
        xticks=cumulative_readouts,
        yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="x", visible=False)
    axes[0].text(
        -0.15,
        1.02,
        "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
    )
    axes[1].text(
        -0.15,
        1.02,
        "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
    )
    fig.tight_layout(w_pad=2.2)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "mnist_static_correction_timing"
    for extension in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight")
    plt.close(fig)

    per_seed: list[dict[str, object]] = []
    for seed, subset in corrected.groupby("seed"):
        per_seed.append(
            {
                "seed": int(seed),
                "trajectories": int(len(subset)),
                "difficulty_time_spearman": float(
                    spearmanr(
                        subset["overtaking_difficulty"],
                        subset["settling_readout"],
                    ).statistic
                ),
                "gap_time_spearman": float(
                    spearmanr(
                        subset["initial_target_gap"],
                        subset["settling_readout"],
                    ).statistic
                ),
                "gain_time_spearman": float(
                    spearmanr(
                        subset["gain_advantage"],
                        subset["settling_readout"],
                    ).statistic
                ),
            }
        )
    seed_frame = pd.DataFrame(per_seed)
    difficulty_mean, difficulty_low, difficulty_high = mean_interval(
        seed_frame["difficulty_time_spearman"].to_numpy()
    )

    summary = {
        "source": str(args.trajectories.resolve()),
        "condition": "distinct_50",
        "cohort": "wrong at update 1 and correct at the final readout",
        "networks": int(corrected["seed"].nunique()),
        "trajectories": int(len(corrected)),
        "all_gain_advantages_positive": True,
        "all_switch_once": True,
        "correction_counts": {
            str(readout): int(count)
            for readout, count in zip(readouts, counts.to_numpy())
        },
        "difficulty_definition": (
            "initial winner-to-target overlap gap divided by the target's "
            "recurrent-gain advantage"
        ),
        "difficulty_bins": bin_summaries,
        "pooled_spearman": {
            "difficulty_vs_correction_readout": float(
                spearmanr(
                    corrected["overtaking_difficulty"],
                    corrected["settling_readout"],
                ).statistic
            ),
            "initial_gap_vs_correction_readout": float(
                spearmanr(
                    corrected["initial_target_gap"],
                    corrected["settling_readout"],
                ).statistic
            ),
            "gain_advantage_vs_correction_readout": float(
                spearmanr(
                    corrected["gain_advantage"],
                    corrected["settling_readout"],
                ).statistic
            ),
        },
        "per_network_spearman": per_seed,
        "mean_network_difficulty_spearman": {
            "mean": difficulty_mean,
            "lower_95": difficulty_low,
            "upper_95": difficulty_high,
        },
    }
    (args.output / "mnist_static_correction_timing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    seed_frame.to_csv(
        args.output / "mnist_static_correction_timing_per_seed.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
