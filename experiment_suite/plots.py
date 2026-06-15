from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _to_int_scalar(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if hasattr(value, "item"):
        return int(value.item())
    raise TypeError(f"Expected int-like scalar, got {type(value).__name__}")


def _sort_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(df.sort_values(by=columns))


def _format_params_short(value: Any) -> str:
    params = _to_int_scalar(value)
    if params >= 1_000_000:
        return f"{params / 1_000_000:.1f}M"
    if params >= 1_000:
        return f"{params / 1_000:.0f}k"
    return str(params)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _build_suite_comparison_df(raw_results_csv: str | Path, list_type: str) -> pd.DataFrame:
    df = pd.read_csv(raw_results_csv)
    frame = pd.DataFrame(df[df["list_type"] == list_type].copy())
    frame["k"] = frame["k_test"].astype(int)
    frame["accuracy"] = frame["accuracy"].astype(float)
    frame["model_label"] = frame["model_name"]
    frame["family_sort"] = [0 if family == "MLP" else 1 if family == "AC" else 99 for family in frame["family"].tolist()]
    return pd.DataFrame(frame.sort_values(by=["family_sort", "model_label", "k"], kind="stable").reset_index(drop=True))


def build_seen_suite_comparison_df(raw_results_csv: str | Path) -> pd.DataFrame:
    return _build_suite_comparison_df(raw_results_csv, "Seen")


def build_unseen_suite_comparison_df(raw_results_csv: str | Path) -> pd.DataFrame:
    return _build_suite_comparison_df(raw_results_csv, "Unseen")


def _best_mlp_label(comparison_df: pd.DataFrame) -> str:
    mlp_df = pd.DataFrame(comparison_df[comparison_df["family"] == "MLP"])
    grouped = pd.DataFrame(mlp_df.groupby("model_label", as_index=False)[["accuracy"]].mean())
    return str(grouped.sort_values(["accuracy", "model_label"], ascending=[False, True]).iloc[0]["model_label"])


def _max_solved_hop_df(comparison_df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in comparison_df.groupby(["family", "model_label"], observed=False):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"Expected 2-tuple groupby key, got {key!r}")
        family, model_label = key
        group_df = pd.DataFrame(group)
        solved = group_df[group_df["accuracy"] >= threshold]["k"]
        exemplar = group_df.iloc[0]
        rows.append(
            {
                "family": family,
                "model_label": model_label,
                "max_solved_hop": _to_int_scalar(solved.max()) if len(solved) > 0 else 0,
                "params": exemplar.get("params"),
                "assembly_size": exemplar.get("assembly_size"),
            }
        )
    return pd.DataFrame(rows)


def _train_limit(comparison_df: pd.DataFrame) -> int:
    if "k_train_max" in comparison_df and not comparison_df["k_train_max"].dropna().empty:
        max_value = comparison_df["k_train_max"].dropna().max()
        return _to_int_scalar(max_value)
    return 4


def _save_accuracy_vs_hop(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=comparison_df, x="k", y="accuracy", hue="model_label", style="family", markers=True, dashes=False)
    plt.axvline(x=_train_limit(comparison_df), color="red", linestyle="--", linewidth=1.5, label="Train limit")
    plt.title("Seen Lists: Accuracy vs Hop Count")
    plt.xlabel("Hop Count")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    path = output_dir / "accuracy_vs_hop_seen.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_accuracy_vs_hop_generic(comparison_df: pd.DataFrame, output_dir: Path, filename: str, title: str) -> Path:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=comparison_df, x="k", y="accuracy", hue="model_label", style="family", markers=True, dashes=False)
    plt.axvline(x=_train_limit(comparison_df), color="red", linestyle="--", linewidth=1.5, label="Train limit")
    plt.title(title)
    plt.xlabel("Hop Count")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_best_mlp_vs_ac(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    best_mlp = _best_mlp_label(comparison_df)
    subset = pd.DataFrame(comparison_df[(comparison_df["family"] == "AC") | (comparison_df["model_label"] == best_mlp)])
    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=subset, x="k", y="accuracy", hue="model_label", style="family", markers=True, dashes=False)
    plt.axvline(x=_train_limit(comparison_df), color="red", linestyle="--", linewidth=1.5, label="Train limit")
    plt.title("Seen Lists: AC vs Best MLP")
    plt.xlabel("Hop Count")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    path = output_dir / "accuracy_vs_hop_seen_best_mlp_vs_ac.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_family_heatmap(comparison_df: pd.DataFrame, family: str, output_dir: Path, filename: str, title: str, value_column: str = "model_label") -> Path:
    del value_column
    family_df = pd.DataFrame(comparison_df[comparison_df["family"] == family].copy())
    if family == "MLP":
        family_df["row_label"] = family_df.apply(lambda row: f"{row['model_label']} / {_format_params_short(row['params'])}", axis=1)
    else:
        family_df["row_label"] = family_df.apply(lambda row: f"{row['model_label']} (asm={int(row['assembly_size'])})", axis=1)
    pivot_df = family_df.pivot_table(index="row_label", columns="k", values="accuracy", aggfunc="mean")
    n_rows, n_cols = pivot_df.shape
    fig_w = max(12.0, 0.9 * n_cols + 4.0)
    fig_h = max(4.5, 1.0 * n_rows + 2.0)
    plt.figure(figsize=(fig_w, fig_h))
    annot_size = 9 if n_cols <= 16 else 7
    sns.heatmap(pivot_df, annot=True, annot_kws={"size": annot_size}, cmap="viridis", fmt=".2f", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xlabel("Hop Count")
    plt.ylabel("Model")
    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_max_solved_hop(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    ms_df = _max_solved_hop_df(comparison_df)
    plt.figure(figsize=(8.5, 5.5))
    sns.barplot(data=ms_df, x="model_label", y="max_solved_hop", hue="family")
    plt.title("Seen Lists: Max Solved Hop (Accuracy >= 0.95)")
    plt.xlabel("Model")
    plt.ylabel("Max Solved Hop")
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = output_dir / "max_solved_hop_seen.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_mlp_size_tradeoff(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    ms_df = _max_solved_hop_df(comparison_df)
    mlp_df = _sort_df(pd.DataFrame(ms_df[ms_df["family"] == "MLP"].copy()), ["params"])
    plt.figure(figsize=(8.2, 5.2))
    sns.lineplot(data=mlp_df, x="params", y="max_solved_hop", marker="o")
    plt.xscale("log")
    plt.title("MLP trades size for hop capacity")
    plt.xlabel("Parameters (log scale)")
    plt.ylabel("Max Solved Hop")
    plt.tight_layout()
    path = output_dir / "mlp_size_tradeoff_seen.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_size_vs_time_tradeoff(comparison_df: pd.DataFrame, output_dir: Path, filename: str, title: str) -> Path:
    ms_df = _max_solved_hop_df(comparison_df)
    mlp_df = _sort_df(pd.DataFrame(ms_df[ms_df["family"] == "MLP"].copy()), ["params"])
    ac_df = _sort_df(pd.DataFrame(ms_df[ms_df["family"] == "AC"].copy()), ["assembly_size"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    sns.lineplot(data=mlp_df, x="params", y="max_solved_hop", marker="o", ax=axes[0])
    axes[0].set_xscale("log")
    axes[0].set_title("MLP size tradeoff")
    axes[0].set_xlabel("Parameters")
    axes[0].set_ylabel("Max Solved Hop")
    sns.lineplot(data=ac_df, x="assembly_size", y="max_solved_hop", marker="o", ax=axes[1])
    axes[1].set_title("AC resource tradeoff")
    axes[1].set_xlabel("Assembly Size")
    axes[1].set_ylabel("Max Solved Hop")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_ac_resource_tradeoff(comparison_df: pd.DataFrame, output_dir: Path, filename: str, title: str) -> Path:
    ms_df = _max_solved_hop_df(comparison_df)
    ac_df = _sort_df(pd.DataFrame(ms_df[ms_df["family"] == "AC"].copy()), ["assembly_size"])
    plt.figure(figsize=(8.5, 5.2))
    sns.lineplot(data=ac_df, x="assembly_size", y="max_solved_hop", marker="o")
    plt.title(title)
    plt.xlabel("Assembly Size")
    plt.ylabel("Max Solved Hop")
    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _save_ac_time_sweep_unseen(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    ac_df = pd.DataFrame(comparison_df[comparison_df["family"] == "AC"].copy())
    plt.figure(figsize=(9.5, 5.8))
    sns.lineplot(data=ac_df, x="internal_steps", y="accuracy", hue="k", marker="o")
    plt.title("Unseen AC: accuracy vs internal time")
    plt.xlabel("Internal Steps")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    path = output_dir / "ac_time_sweep_unseen.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_ac_time_tradeoff(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    ac_df = pd.DataFrame(comparison_df[comparison_df["family"] == "AC"].copy())
    plt.figure(figsize=(8.5, 5.4))
    sns.lineplot(data=ac_df, x="internal_steps", y="accuracy", hue="model_label", marker="o")
    plt.title("AC trades time for computation")
    plt.xlabel("Internal Steps")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    path = output_dir / "ac_time_vs_hop.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_paper_panel(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    best_mlp = _best_mlp_label(comparison_df)
    best_subset = pd.DataFrame(comparison_df[(comparison_df["family"] == "AC") | (comparison_df["model_label"] == best_mlp)])
    mlp_ms = _max_solved_hop_df(comparison_df)
    mlp_ms = _sort_df(pd.DataFrame(mlp_ms[mlp_ms["family"] == "MLP"].copy()), ["params"])
    ac_df = pd.DataFrame(comparison_df[comparison_df["family"] == "AC"].copy())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    sns.lineplot(data=best_subset, x="k", y="accuracy", hue="model_label", style="family", markers=True, dashes=False, ax=axes[0])
    axes[0].axvline(x=_train_limit(comparison_df), color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title("Accuracy vs hop")
    axes[0].set_xlabel("Hop Count")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(-0.02, 1.05)
    sns.lineplot(data=mlp_ms, x="params", y="max_solved_hop", marker="o", ax=axes[1])
    axes[1].set_xscale("log")
    axes[1].set_title("MLP size tradeoff")
    axes[1].set_xlabel("Parameters")
    axes[1].set_ylabel("Max Solved Hop")
    sns.lineplot(data=ac_df, x="internal_steps", y="accuracy", hue="model_label", marker="o", ax=axes[2])
    axes[2].set_title("AC time tradeoff")
    axes[2].set_xlabel("Internal Steps")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(-0.02, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles=handles, labels=labels, loc="best", fontsize=8)
    legend = axes[2].get_legend()
    if legend is not None:
        legend.remove()
    fig.suptitle("Seen Lists: AC trades time, MLP trades size", fontsize=14)
    fig.tight_layout()
    path = output_dir / "paper_panel_seen_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def generate_seen_suite_plots(raw_results_csv: str | Path, output_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df = build_seen_suite_comparison_df(raw_results_csv)
    return [
        _save_accuracy_vs_hop(comparison_df, output_path),
        _save_best_mlp_vs_ac(comparison_df, output_path),
        _save_family_heatmap(comparison_df, "MLP", output_path, "mlp_accuracy_heatmap_seen.png", "MLP seen accuracy heatmap"),
        _save_family_heatmap(comparison_df, "AC", output_path, "ac_accuracy_heatmap_seen.png", "AC seen accuracy heatmap"),
        _save_mlp_size_tradeoff(comparison_df, output_path),
        _save_max_solved_hop(comparison_df, output_path),
        _save_ac_time_tradeoff(comparison_df, output_path),
        _save_paper_panel(comparison_df, output_path),
    ]


def _pick_best_t(accuracy_df: pd.DataFrame) -> int:
    best = accuracy_df.loc[accuracy_df["accuracy"].idxmax()]
    return int(best["t"])


def _confusion_matrix_image(df_slice: pd.DataFrame, output_path: Path, t_label: str) -> Path:
    labels = sorted(set(df_slice["target"].tolist()) | set(df_slice["prediction"].tolist()))
    confusion = pd.crosstab(
        df_slice["target"], df_slice["prediction"],
        rownames=["True"], colnames=["Predicted"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", vmin=0)
    plt.title(f"MNIST Confusion Matrix (t={t_label})")
    plt.tight_layout()
    path = output_path / f"mnist_confusion_{t_label}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _pair_drift_image(mnist_df: pd.DataFrame, output_path: Path) -> list[Path]:
    known_pairs = [(7, 9), (3, 5), (4, 9)]
    paths: list[Path] = []

    for a, b in known_pairs:
        pair_mask = mnist_df["target"].isin([a, b])
        if not pair_mask.any():
            continue
        pair_df = pd.DataFrame(mnist_df.loc[pair_mask].copy())
        pair_df["correct"] = _coerce_bool_series(pair_df["correct"])
        pair_acc = pd.DataFrame(
            pair_df.groupby(["t", "target"], as_index=False)["correct"].mean()
        ).rename(columns={"correct": "accuracy"})

        plt.figure(figsize=(7, 4.8))
        sns.lineplot(data=pair_acc, x="t", y="accuracy", hue="target", marker="o", palette={a: "#E24A33", b: "#348ABD"})
        plt.title(f"MNIST Pair Drift: {a}/{b}")
        plt.xlabel("t")
        plt.ylabel("Accuracy")
        plt.ylim(-0.02, 1.05)
        plt.tight_layout()
        pair_path = output_path / f"mnist_pair_drift_{a}_{b}.png"
        plt.savefig(pair_path, dpi=200)
        plt.close()
        paths.append(pair_path)

    return paths


def _seed_aggregated_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy by seed first, then across seeds (Theory Map §3)."""
    if "seed" not in df.columns:
        return pd.DataFrame(df.groupby("t", as_index=False)["correct"].agg(
            accuracy="mean", se=lambda x: x.std() / np.sqrt(len(x))
        ))
    per_seed = pd.DataFrame(df.groupby(["seed", "t"], as_index=False)["correct"].mean())
    return pd.DataFrame(per_seed.groupby("t", as_index=False)["correct"].agg(
        accuracy="mean", se=lambda x: x.std() / np.sqrt(len(x))
    ))


def _seed_aggregated_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate margin by seed first, including lower quantile (Theory Map §6)."""
    if "seed" not in df.columns:
        return pd.DataFrame(df.groupby("t", as_index=False).agg(
            margin_mean=("margin", "mean"),
            margin_q10=("margin", lambda x: x.quantile(0.1)),
        ))
    per_seed = pd.DataFrame(
        df.groupby(["seed", "t"], as_index=False).agg(
            margin_mean=("margin", "mean"),
            margin_q10=("margin", lambda x: x.quantile(0.1)),
        )
    )
    return pd.DataFrame(per_seed.groupby("t", as_index=False).agg(
        margin_mean=("margin_mean", "mean"),
        margin_q10=("margin_q10", "mean"),
        se=("margin_mean", lambda x: x.std() / np.sqrt(len(x))),
    ))


def _seed_aggregated_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate o_y and max_wrong by seed first."""
    if "seed" not in df.columns:
        return pd.DataFrame(df.groupby("t", as_index=False).agg(
            correct_overlap=("correct_overlap", "mean"),
            strongest_wrong_overlap=("strongest_wrong_overlap", "mean"),
        ))
    per_seed = pd.DataFrame(
        df.groupby(["seed", "t"], as_index=False).agg(
            correct_overlap=("correct_overlap", "mean"),
            strongest_wrong_overlap=("strongest_wrong_overlap", "mean"),
        )
    )
    return pd.DataFrame(per_seed.groupby("t", as_index=False).agg(
        correct_overlap=("correct_overlap", "mean"),
        strongest_wrong_overlap=("strongest_wrong_overlap", "mean"),
        se_correct=("correct_overlap", lambda x: x.std() / np.sqrt(len(x))),
        se_wrong=("strongest_wrong_overlap", lambda x: x.std() / np.sqrt(len(x))),
    ))


def _parse_overlap_cell(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.tolist()]
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("overlaps must serialize to a list")
    return [float(item) for item in parsed]


def _save_mnist_sequence_plots(mnist_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    required = {"sequence_step", "phase_digit", "step_in_phase", "prediction", "margin", "overlaps"}
    missing = required.difference(mnist_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"MNIST sequence raw results missing required columns: {missing_text}")

    seq_df = pd.DataFrame(mnist_df.copy())
    seq_df["sequence_step"] = pd.to_numeric(seq_df["sequence_step"]).astype(int)
    seq_df["phase_digit"] = pd.to_numeric(seq_df["phase_digit"]).astype(int)
    seq_df["step_in_phase"] = pd.to_numeric(seq_df["step_in_phase"]).astype(int)
    seq_df = pd.DataFrame(seq_df.sort_values("sequence_step", kind="stable"))

    paths: list[Path] = []
    boundaries = pd.DataFrame(
        seq_df.groupby("phase_digit", sort=False, as_index=False).agg(
            start=("sequence_step", "min"),
            end=("sequence_step", "max"),
        )
    )

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.step(seq_df["sequence_step"], seq_df["phase_digit"], where="post", linewidth=2.5, label="Presented digit")
    ax.plot(seq_df["sequence_step"], seq_df["prediction"], "o--", linewidth=2, label="Predicted digit")
    for _, row in boundaries.iterrows():
        ax.axvspan(float(row["start"]) - 0.5, float(row["end"]) + 0.5, alpha=0.08)
    ax.set_title("MNIST Sequence Probe: Presented vs Predicted Digit")
    ax.set_xlabel("Sequence step")
    ax.set_ylabel("Digit")
    ax.set_yticks(range(10))
    ax.set_ylim(-0.5, 9.5)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "mnist_sequence_predictions_0_to_9.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    overlap_rows: list[dict[str, object]] = []
    for _, row in seq_df.iterrows():
        overlaps = _parse_overlap_cell(row["overlaps"])
        for digit, overlap in enumerate(overlaps[:10]):
            overlap_rows.append(
                {
                    "sequence_step": int(row["sequence_step"]),
                    "digit": digit,
                    "overlap": overlap,
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    sns.lineplot(data=overlap_df, x="sequence_step", y="overlap", hue="digit", palette="tab10", marker="o", ax=ax)
    for _, row in boundaries.iterrows():
        ax.axvline(float(row["start"]) - 0.5, color="0.75", linestyle="--", linewidth=1)
    ax.set_title("MNIST Sequence Probe: Class Overlaps Over Time")
    ax.set_xlabel("Sequence step")
    ax.set_ylabel("Overlap with class prototype")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(title="Class", ncol=2)
    fig.tight_layout()
    path = output_dir / "mnist_sequence_overlaps_0_to_9.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="Decision boundary")
    ax.plot(seq_df["sequence_step"], seq_df["margin"], "o-", linewidth=2.2, label="Target margin")
    for _, row in boundaries.iterrows():
        ax.axvspan(float(row["start"]) - 0.5, float(row["end"]) + 0.5, alpha=0.08)
    ax.set_title("MNIST Sequence Probe: Margin After Digit Switches")
    ax.set_xlabel("Sequence step")
    ax.set_ylabel("Margin for currently presented digit")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "mnist_sequence_margin_0_to_9.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    return paths


def _save_mnist_sequence_hold_sweep_plots(sequence_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    required = {
        "hold_steps",
        "sequence_step",
        "phase_digit",
        "step_in_phase",
        "prediction",
        "correct",
    }
    missing = required.difference(sequence_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"MNIST sequence hold sweep missing required columns: {missing_text}")

    sweep_df = pd.DataFrame(sequence_df.copy())
    sweep_df["hold_steps"] = pd.to_numeric(sweep_df["hold_steps"]).astype(int)
    sweep_df["sequence_step"] = pd.to_numeric(sweep_df["sequence_step"]).astype(int)
    sweep_df["phase_digit"] = pd.to_numeric(sweep_df["phase_digit"]).astype(int)
    sweep_df["step_in_phase"] = pd.to_numeric(sweep_df["step_in_phase"]).astype(int)
    sweep_df["prediction"] = pd.to_numeric(sweep_df["prediction"]).astype(int)
    sweep_df["correct"] = _coerce_bool_series(sweep_df["correct"])
    sweep_df = pd.DataFrame(sweep_df.sort_values(["hold_steps", "sequence_step"], kind="stable"))
    sweep_df["phase_progress"] = sweep_df["phase_digit"] + sweep_df["step_in_phase"] / sweep_df["hold_steps"]

    paths: list[Path] = []
    run_columns = [
        column
        for column in ("model_name", "seed", "task_seed", "instance_id")
        if column in sweep_df.columns
    ]
    run_phase_columns = [*run_columns, "hold_steps", "phase_digit"]

    final_rows = pd.DataFrame(
        sweep_df.sort_values("step_in_phase", kind="stable")
        .groupby(run_phase_columns, as_index=False, observed=False)
        .tail(1)
    )
    final_acc = pd.DataFrame(
        final_rows.groupby("hold_steps", as_index=False, observed=False).agg(
            final_accuracy=("correct", "mean")
        )
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.barplot(data=final_acc, x="hold_steps", y="final_accuracy", color="#4C72B0", ax=ax)
    ax.set_title("MNIST Sequence Hold Sweep: Final Phase Accuracy")
    ax.set_xlabel("Steps per digit")
    ax.set_ylabel("Final-step accuracy")
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    path = output_dir / "mnist_sequence_hold_sweep_final_accuracy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    latency_rows: list[dict[str, object]] = []
    for key, group in sweep_df.groupby(run_phase_columns, observed=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_values = dict(zip(run_phase_columns, key, strict=True))
        hold_steps = int(key_values["hold_steps"])
        digit = int(key_values["phase_digit"])
        correct_steps = pd.DataFrame(group[group["correct"]])
        if correct_steps.empty:
            latency = hold_steps
            switched = False
        else:
            latency = int(correct_steps["step_in_phase"].min())
            switched = True
        row = {column: key_values[column] for column in run_columns}
        row.update({"hold_steps": hold_steps, "phase_digit": digit, "latency": latency, "switched": switched})
        latency_rows.append(row)
    latency_df = pd.DataFrame(latency_rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    sns.lineplot(data=latency_df, x="phase_digit", y="latency", hue="hold_steps", marker="o", palette="viridis", ax=ax)
    ax.set_title("MNIST Sequence Hold Sweep: First Correct Step After Switch")
    ax.set_xlabel("Presented digit")
    ax.set_ylabel("First correct step in phase (hold length means never)")
    ax.set_xticks(range(10))
    fig.tight_layout()
    path = output_dir / "mnist_sequence_hold_sweep_switch_latency.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    sns.lineplot(
        data=sweep_df,
        x="phase_progress",
        y="prediction",
        hue="hold_steps",
        marker="o",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("MNIST Sequence Hold Sweep: Prediction Timelines")
    ax.set_xlabel("Presented digit + within-digit progress")
    ax.set_ylabel("Predicted digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_ylim(-0.5, 9.5)
    ax.legend(title="Steps per digit")
    fig.tight_layout()
    path = output_dir / "mnist_sequence_hold_sweep_predictions.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    return paths


def generate_mnist_ac_plots(raw_results_csv: str | Path, plots_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    output_path = Path(plots_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_results_csv)
    required_columns = {"t", "target", "correct", "margin",
                        "correct_overlap", "strongest_wrong_overlap", "prediction"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"MNIST raw results missing required columns: {missing}")

    mnist_df = pd.DataFrame(df.copy())
    mnist_df["t"] = pd.to_numeric(mnist_df["t"])
    mnist_df["target"] = pd.to_numeric(mnist_df["target"]).astype(int)
    mnist_df["prediction"] = pd.to_numeric(mnist_df["prediction"]).astype(int)
    mnist_df["correct"] = _coerce_bool_series(mnist_df["correct"])
    mnist_df["margin"] = pd.to_numeric(mnist_df["margin"])
    mnist_df["correct_overlap"] = pd.to_numeric(mnist_df["correct_overlap"])
    mnist_df["strongest_wrong_overlap"] = pd.to_numeric(mnist_df["strongest_wrong_overlap"])

    all_paths: list[Path] = []

    if "experiment" in mnist_df.columns and (mnist_df["experiment"].astype(str) == "mnist_sequence").any():
        sequence_mask = mnist_df["experiment"].astype(str) == "mnist_sequence"
        sequence_df = pd.DataFrame(mnist_df[sequence_mask].copy())
        hold_column = "hold_steps" if "hold_steps" in sequence_df.columns else "steps_per_digit"
        hold_count = pd.to_numeric(sequence_df[hold_column]).nunique() if hold_column in sequence_df.columns else 1
        if hold_count > 1 and "hold_steps" in sequence_df.columns:
            all_paths.extend(_save_mnist_sequence_hold_sweep_plots(sequence_df, output_path))
        else:
            all_paths.extend(_save_mnist_sequence_plots(sequence_df, output_path))
        mnist_df = pd.DataFrame(mnist_df[~sequence_mask].copy())

    if mnist_df.empty:
        return all_paths

    # Seed-first aggregation (§3)
    acc_df = _seed_aggregated_accuracy(mnist_df)
    mar_df = _seed_aggregated_margin(mnist_df)
    ov_df = _seed_aggregated_overlap(mnist_df)

    has_se = "se" in acc_df.columns

    # --- §12: Accuracy vs t (global, with SE bands from seed aggregation) ---
    plt.figure(figsize=(8.5, 5.2))
    if has_se:
        plt.errorbar(acc_df["t"], acc_df["accuracy"], yerr=acc_df["se"],
                     fmt="o-", capsize=4, capthick=1.5, linewidth=2, label="Accuracy")
    else:
        sns.lineplot(data=acc_df, x="t", y="accuracy", marker="o")
    plt.title("MNIST: Accuracy vs t (mean ± SE over seeds)")
    plt.xlabel("t")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    accuracy_path = output_path / "mnist_accuracy_vs_t.png"
    plt.savefig(accuracy_path, dpi=200)
    plt.close()
    all_paths.append(accuracy_path)

    # --- §6: Per-class accuracy vs t ---
    plt.figure(figsize=(10.5, 6.0))
    sns.lineplot(data=mnist_df, x="t", y="correct", hue="target",
                 marker="o", palette="tab10",
                 estimator="mean", errorbar=("ci", 68) if has_se else None)
    plt.title("MNIST: Per-Class Accuracy vs t")
    plt.xlabel("t")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    per_class_path = output_path / "mnist_per_class_accuracy_vs_t.png"
    plt.savefig(per_class_path, dpi=200)
    plt.close()
    all_paths.append(per_class_path)

    # --- §12: Combined margin figure: o_y(t), max_{z≠y} o_z(t), m_y(t) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

    # Left: overlaps
    if has_se and "se_correct" in ov_df.columns:
        ax1.errorbar(ov_df["t"], ov_df["correct_overlap"], yerr=ov_df["se_correct"],
                     fmt="o-", capsize=3, label=r"$o_y(t)$ Correct Overlap", linewidth=2)
        ax1.errorbar(ov_df["t"], ov_df["strongest_wrong_overlap"], yerr=ov_df["se_wrong"],
                     fmt="s--", capsize=3, label=r"$\max_{z\neq y} o_z(t)$ Strongest Wrong", linewidth=2)
    else:
        ax1.plot(ov_df["t"], ov_df["correct_overlap"], "o-", label=r"$o_y(t)$")
        ax1.plot(ov_df["t"], ov_df["strongest_wrong_overlap"], "s--", label=r"$\max_{z\neq y} o_z(t)$")
    ax1.set_title("Overlap vs t")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Overlap")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend()

    # Right: margin with mean + lower quantile (§6)
    ax2.plot(mar_df["t"], mar_df["margin_mean"], "o-", label=r"$\mathbb{E}[m_y(t)]$ Mean Margin", linewidth=2)
    if "margin_q10" in mar_df.columns:
        ax2.plot(mar_df["t"], mar_df["margin_q10"], "s--", label=r"$Q_{0.1}[m_y(t)]$ Lower Quantile", linewidth=2)
    if "se" in mar_df.columns:
        ax2.fill_between(mar_df["t"],
                         mar_df["margin_mean"] - mar_df["se"],
                         mar_df["margin_mean"] + mar_df["se"],
                         alpha=0.2, label="±1 SE")
    ax2.set_title("Margin vs t")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Margin")
    ax2.legend()
    fig.suptitle("MNIST: Overlap and Margin vs t (§12 Theory-to-Experiment Map)", fontsize=14)
    fig.tight_layout()
    margin_path = output_path / "mnist_margin_vs_t.png"
    fig.savefig(margin_path, dpi=200)
    plt.close(fig)
    all_paths.append(margin_path)

    # --- Confusion matrices: early, best, late ---
    t_values = sorted(mnist_df["t"].unique().tolist())
    if len(t_values) >= 1:
        early_t = t_values[0]
        all_paths.append(_confusion_matrix_image(mnist_df[mnist_df["t"] == early_t], output_path, "early"))

    if len(t_values) >= 2:
        best_t = _pick_best_t(acc_df)
        all_paths.append(_confusion_matrix_image(mnist_df[mnist_df["t"] == best_t], output_path, "best"))

    if len(t_values) >= 3:
        late_t = t_values[-1]
        all_paths.append(_confusion_matrix_image(mnist_df[mnist_df["t"] == late_t], output_path, "late"))

    # --- Pair drift (§12) ---
    all_paths.extend(_pair_drift_image(mnist_df, output_path))

    return all_paths


def generate_seen_mlp_plots(raw_results_csv: str | Path, output_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df = build_seen_suite_comparison_df(raw_results_csv)
    mlp_df = pd.DataFrame(comparison_df[comparison_df["family"] == "MLP"].copy())
    return [
        _save_accuracy_vs_hop_generic(mlp_df, output_path, "accuracy_vs_hop_seen_mlp.png", "Seen MLP: accuracy vs hop"),
        _save_family_heatmap(mlp_df, "MLP", output_path, "seen_mlp_heatmap.png", "Seen MLP heatmap"),
        _save_mlp_size_tradeoff(mlp_df, output_path).rename(output_path / "size_tradeoff_seen_mlp.png"),
    ]


def generate_seen_ac_plots(raw_results_csv: str | Path, output_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df = build_seen_suite_comparison_df(raw_results_csv)
    ac_df = pd.DataFrame(comparison_df[comparison_df["family"] == "AC"].copy())
    return [
        _save_accuracy_vs_hop_generic(ac_df, output_path, "accuracy_vs_hop_seen_ac.png", "Seen AC: Accuracy vs Hop Count"),
        _save_family_heatmap(ac_df, "AC", output_path, "accuracy_heatmap_seen_ac.png", "Seen AC: accuracy heatmap"),
        _save_ac_resource_tradeoff(ac_df, output_path, "size_tradeoff_seen_ac.png", "Seen AC: assembly size tradeoff"),
        _save_max_solved_hop(ac_df, output_path).rename(output_path / "max_solved_hop_seen_ac.png"),
    ]


def generate_unseen_suite_plots(raw_results_csv: str | Path, output_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df = build_unseen_suite_comparison_df(raw_results_csv)
    return [
        _save_accuracy_vs_hop_generic(comparison_df, output_path, "unseen_accuracy_vs_hop.png", "Unseen Lists: Accuracy vs Hop Count"),
        _save_family_heatmap(comparison_df, "MLP", output_path, "mlp_accuracy_heatmap_unseen.png", "MLP unseen accuracy heatmap"),
        _save_family_heatmap(comparison_df, "AC", output_path, "ac_accuracy_heatmap_unseen.png", "AC unseen accuracy heatmap"),
        _save_ac_time_sweep_unseen(comparison_df, output_path),
        _save_size_vs_time_tradeoff(comparison_df, output_path, "unseen_size_vs_time_tradeoff.png", "Unseen Lists: MLP size vs AC resource tradeoff"),
    ]


def generate_unseen_mlp_plots(raw_results_csv: str | Path, output_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df = build_unseen_suite_comparison_df(raw_results_csv)
    mlp_df = pd.DataFrame(comparison_df[comparison_df["family"] == "MLP"].copy())
    return [
        _save_accuracy_vs_hop_generic(mlp_df, output_path, "accuracy_vs_hop_unseen_mlp.png", "Unseen MLP: accuracy vs hop"),
        _save_family_heatmap(mlp_df, "MLP", output_path, "unseen_mlp_heatmap.png", "Unseen MLP heatmap"),
        _save_mlp_size_tradeoff(mlp_df, output_path).rename(output_path / "size_tradeoff_unseen_mlp.png"),
    ]


def generate_unseen_ac_plots(raw_results_csv: str | Path, output_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df = build_unseen_suite_comparison_df(raw_results_csv)
    ac_df = pd.DataFrame(comparison_df[comparison_df["family"] == "AC"].copy())
    return [
        _save_accuracy_vs_hop_generic(ac_df, output_path, "accuracy_vs_hop_unseen_ac.png", "Unseen AC: Accuracy vs Hop Count"),
        _save_family_heatmap(ac_df, "AC", output_path, "accuracy_heatmap_unseen_ac.png", "Unseen AC: accuracy heatmap"),
        _save_ac_resource_tradeoff(ac_df, output_path, "size_tradeoff_unseen_ac.png", "Unseen AC: assembly size tradeoff"),
        _save_max_solved_hop(ac_df, output_path).rename(output_path / "max_solved_hop_unseen_ac.png"),
    ]


def generate_pointer_ac_plots(raw_results_csv: str | Path, plots_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    output_path = Path(plots_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_results_csv)
    required = {"experiment", "L", "t", "correct", "path_accuracy", "first_error_index"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Pointer results missing required columns: {', '.join(sorted(missing))}")

    pdf = pd.DataFrame(df[df["experiment"].astype(str).isin(["pointer_chasing", "dfa", "binary_search"])].copy())
    if pdf.empty:
        return []

    pdf["L"] = pd.to_numeric(pdf["L"]).astype(int)
    pdf["t"] = pd.to_numeric(pdf["t"]).astype(int)
    pdf["correct"] = _coerce_bool_series(pdf["correct"])
    pdf["accuracy"] = pd.to_numeric(pdf["accuracy"], errors="coerce")
    pdf["path_accuracy"] = pd.to_numeric(pdf["path_accuracy"], errors="coerce")
    pdf["first_error_index"] = pd.to_numeric(pdf["first_error_index"], errors="coerce")

    def _agg(df_sub: pd.DataFrame) -> pd.DataFrame:
        if "seed" in df_sub.columns and df_sub["seed"].nunique() > 1:
            seed_acc = df_sub.groupby(["L", "t", "seed"])["accuracy"].mean().reset_index()
            acc_by_lt = seed_acc.groupby(["L", "t"])["accuracy"].mean().reset_index()
            acc_by_lt = acc_by_lt.rename(columns={"accuracy": "accuracy_mean"})
        else:
            acc_by_lt = df_sub.groupby(["L", "t"])["accuracy"].mean().reset_index()
            acc_by_lt = acc_by_lt.rename(columns={"accuracy": "accuracy_mean"})
        return acc_by_lt

    acc_df = _agg(pdf)

    paths: list[Path] = []

    # Heatmap Acc(L,t)
    unique_L = sorted(acc_df["L"].unique())
    unique_t = sorted(acc_df["t"].unique())
    heatmap_data = np.full((len(unique_L), len(unique_t)), np.nan)
    for i, L_val in enumerate(unique_L):
        mask = acc_df["L"] == L_val
        for j, t_val in enumerate(unique_t):
            val = acc_df.loc[mask & (acc_df["t"] == t_val), "accuracy_mean"]
            if len(val) > 0:
                heatmap_data[i, j] = val.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_data, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(unique_t)))
    ax.set_xticklabels([str(t) for t in unique_t])
    ax.set_yticks(range(len(unique_L)))
    ax.set_yticklabels([str(L) for L in unique_L])
    ax.set_xlabel("t (internal time)")
    ax.set_ylabel("K (Hop Depth)")
    ax.set_title("Pointer Chasing: Accuracy Heatmap Acc(L,t)")
    plt.colorbar(im, ax=ax, label="Mean Accuracy")
    fig.tight_layout()
    heatmap_path = output_path / "pointer_accuracy_heatmap_L_t.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    paths.append(heatmap_path)

    # Accuracy vs t by L
    fig, ax = plt.subplots(figsize=(10, 6))
    for L_val in unique_L:
        L_data = acc_df[acc_df["L"] == L_val]
        ax.plot(L_data["t"], L_data["accuracy_mean"], marker="o", label=f"L={L_val}")
    ax.set_xlabel("t (internal time)")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Pointer Chasing: Accuracy vs t by Chain Length")
    ax.legend(title="L")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    t_path = output_path / "pointer_accuracy_vs_t_by_L.png"
    fig.savefig(t_path, dpi=200)
    plt.close(fig)
    paths.append(t_path)

    # Accuracy vs L by t
    fig, ax = plt.subplots(figsize=(10, 6))
    for t_val in unique_t:
        t_data = acc_df[acc_df["t"] == t_val]
        ax.plot(t_data["L"], t_data["accuracy_mean"], marker="s", label=f"t={t_val}")
    ax.set_xlabel("K (Hop Depth)")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Pointer Chasing: Accuracy vs Depth by Time Budget")
    ax.legend(title="t")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    L_path = output_path / "pointer_accuracy_vs_L_by_t.png"
    fig.savefig(L_path, dpi=200)
    plt.close(fig)
    paths.append(L_path)

    # Path accuracy vs L, preserving the time-budget dimension.
    if pdf["path_accuracy"].notna().any():
        path_acc_by_L_t = pdf.groupby(["L", "t"])["path_accuracy"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        for t_val in unique_t:
            t_data = path_acc_by_L_t[path_acc_by_L_t["t"] == t_val]
            ax.plot(t_data["L"], t_data["path_accuracy"], marker="o", label=f"t={t_val}")
        ax.set_xlabel("K (Hop Depth)")
        ax.set_ylabel("Mean Path Accuracy")
        ax.set_title("Pointer Chasing: Path Accuracy vs Depth by Time Budget")
        ax.legend(title="t")
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(output_path / "pointer_path_accuracy_vs_L.png", dpi=150)
        plt.close()
        paths.append(output_path / "pointer_path_accuracy_vs_L.png")

    # NEW: Shortcut Ablation Plot
    # We interpret specific Ls as shortcuts for a nominal depth of 8
    shortcut_map = {8: "Standard (M)", 4: "M² Shortcuts", 2: "M⁴ Shortcuts", 1: "M⁸ Shortcuts"}
    shortcut_df = acc_df[acc_df["L"].isin(shortcut_map.keys())].copy()
    if not shortcut_df.empty:
        plt.figure(figsize=(8, 6))
        for L_val in sorted(shortcut_map.keys(), reverse=True):
            sub_df = shortcut_df[shortcut_df["L"] == L_val].sort_values("t")
            if not sub_df.empty:
                plt.plot(sub_df["t"], sub_df["accuracy_mean"], marker="s", label=f"{shortcut_map[L_val]}")
        
        plt.axhline(0.95, color="gray", linestyle="--", alpha=0.7)
        plt.title("Time-Size Tradeoff: Shortcuts vs Execution Time (Nominal Depth 8)")
        plt.xlabel("Internal Execution Time (t)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "pointer_shortcut_ablation.png", dpi=150)
        plt.close()
        paths.append(output_path / "pointer_shortcut_ablation.png")

    # First error histogram
    first_err = pdf["first_error_index"].dropna()
    if not first_err.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(first_err.astype(int), bins=range(int(first_err.max()) + 2), align="left",
                edgecolor="black", alpha=0.7)
        ax.set_xlabel("First Error Index")
        ax.set_ylabel("Count")
        ax.set_title("Pointer Chasing: First Error Index Distribution")
        fig.tight_layout()
        fe_path = output_path / "pointer_first_error_histogram.png"
        fig.savefig(fe_path, dpi=200)
        plt.close(fig)
        paths.append(fe_path)

    return paths
