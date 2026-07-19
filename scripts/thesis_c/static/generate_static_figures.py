from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


COLOURS = {
    "held": "#0072B2",
    "transient": "#D55E00",
    1: "#CC3311",
    10: "#0077BB",
    100: "#009988",
}


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


def held_transient(frame: pd.DataFrame, output: Path) -> None:
    per_seed = (
        frame.groupby(["stimulus_mode", "t", "seed"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), margin=("margin", "mean"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharex=True)
    for mode in ("held", "transient"):
        subset = per_seed[per_seed["stimulus_mode"] == mode]
        times = sorted(subset["t"].unique())
        label = "Cue held" if mode == "held" else "Cue removed"
        style = "-" if mode == "held" else "--"
        for axis, metric in zip(axes, ("accuracy", "margin")):
            means, intervals = [], []
            for time in times:
                mean, interval = seed_interval(subset.loc[subset["t"] == time, metric])
                means.append(mean)
                intervals.append(interval)
            means_array = np.asarray(means)
            interval_array = np.asarray(intervals)
            axis.plot(
                times, means_array, style, marker="o", ms=3.5,
                lw=1.6, color=COLOURS[mode], label=label,
            )
            lower = means_array - interval_array
            upper = means_array + interval_array
            if metric == "accuracy":
                lower = np.clip(lower, 0, 1)
                upper = np.clip(upper, 0, 1)
            axis.fill_between(times, lower, upper, color=COLOURS[mode],
                              alpha=0.12, linewidth=0)

    axes[0].set(xlabel="Additional internal updates", ylabel="Accuracy", ylim=(-0.02, 1.02))
    axes[0].axhline(0.1, color="0.4", lw=0.8, ls=":")
    axes[1].set(xlabel="Additional internal updates", ylabel="Mean overlap margin")
    axes[1].axhline(0, color="0.25", lw=0.8)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(-0.16, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.16, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=2.1)
    save(fig, output, "mnist_held_vs_removed")


def seed_interval(values: pd.Series) -> tuple[float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return mean, 0.0
    critical = float(student_t.ppf(0.975, len(data) - 1))
    half_width = critical * float(data.std(ddof=1)) / np.sqrt(len(data))
    return mean, half_width


def retention(frame: pd.DataFrame, output: Path) -> None:
    locked = frame[
        (frame["normalisation"] == "full_incoming")
        & (frame["cue_duration_s"].astype(int) == 2)
        & frame["presentation_rounds"].astype(int).isin([1, 10, 100])
    ].copy()
    locked["presentation_rounds"] = locked["presentation_rounds"].astype(int)
    locked["retention_ell"] = locked["retention_ell"].astype(int)

    per_seed = (
        locked.groupby(["presentation_rounds", "retention_ell", "seed"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), margin=("margin", "mean"))
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharex=True)
    for exposure in (1, 10, 100):
        subset = per_seed[per_seed["presentation_rounds"] == exposure]
        lags = sorted(subset["retention_ell"].unique())
        for axis, metric in zip(axes, ("accuracy", "margin")):
            means, intervals = [], []
            for lag in lags:
                mean, interval = seed_interval(subset.loc[subset["retention_ell"] == lag, metric])
                means.append(mean)
                intervals.append(interval)
            means_array = np.asarray(means)
            interval_array = np.asarray(intervals)
            axis.plot(lags, means_array, marker="o", ms=3.2, lw=1.5,
                      color=COLOURS[exposure], label=f"Exposure $R={exposure}$")
            if metric == "accuracy":
                lower = np.clip(means_array - interval_array, 0, 1)
                upper = np.clip(means_array + interval_array, 0, 1)
            else:
                lower = means_array - interval_array
                upper = means_array + interval_array
            axis.fill_between(lags, lower, upper, color=COLOURS[exposure], alpha=0.13, linewidth=0)

    axes[0].set(xlabel="Autonomous lag $\\ell$", ylabel="Accuracy", ylim=(-0.02, 1.02))
    axes[0].axhline(0.1, color="0.4", lw=0.8, ls=":")
    axes[1].set(xlabel="Autonomous lag $\\ell$", ylabel="Mean overlap margin")
    axes[1].axhline(0, color="0.25", lw=0.8)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(-0.16, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.16, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    save(fig, output, "mnist_retention_by_exposure")


def representative_trace(trace_path: Path, output: Path) -> None:
    row = json.loads(trace_path.read_text(encoding="utf-8"))
    trace = np.asarray(row["overlap_trajectory"], dtype=float)
    target = int(row["target"])
    cue_end = int(row["cue_duration_s"])
    rivals = [index for index in range(trace.shape[1]) if index != target]
    rival = rivals[int(np.argmax(trace[cue_end, rivals]))]
    updates = np.arange(trace.shape[0])

    fig, axis = plt.subplots(figsize=(7.1, 3.0))
    for digit in range(trace.shape[1]):
        if digit == target:
            axis.plot(updates, trace[:, digit], color="#CC3311", lw=2.0,
                      label=f"True assembly ({target})", zorder=4)
        elif digit == rival:
            axis.plot(updates, trace[:, digit], color="#0077BB", lw=1.5, ls="--",
                      label=f"Strongest rival at removal ({rival})", zorder=3)
        else:
            axis.plot(updates, trace[:, digit], color="0.72", lw=0.8, alpha=0.8)
    axis.axvline(cue_end, color="0.2", lw=1.0, ls=":", label="Cue removed")
    axis.set(xlabel="Internal update $r$", ylabel="Assembly overlap", ylim=(-0.02, 1.02))
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save(fig, output, "mnist_representative_overlap_trace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-raw", type=Path, required=True)
    parser.add_argument("--retention-raw", type=Path, required=True)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    held_transient(pd.read_csv(args.held_raw), args.output)
    frame = pd.read_csv(args.retention_raw)
    retention(frame, args.output)
    representative_trace(args.trace_json, args.output)


if __name__ == "__main__":
    main()
