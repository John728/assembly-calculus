from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_seen_suite_comparison_df(raw_results_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(raw_results_csv)
    seen = df[df["list_type"] == "Seen"].copy()
    seen["k"] = seen["k_test"].astype(int)
    seen["accuracy"] = seen["accuracy"].astype(float)
    seen["model_label"] = seen["model_name"]
    seen["family_sort"] = seen["family"].apply(lambda family: 0 if family == "MLP" else 1 if family == "AC" else 99)
    return seen.sort_values(by=["family_sort", "model_label", "k"], kind="stable").reset_index(drop=True)


def _best_mlp_label(comparison_df: pd.DataFrame) -> str:
    mlp_df = comparison_df[comparison_df["family"] == "MLP"]
    grouped = pd.DataFrame(mlp_df.groupby("model_label", as_index=False)[["accuracy"]].mean())
    return str(grouped.sort_values(["accuracy", "model_label"], ascending=[False, True]).iloc[0]["model_label"])


def _max_solved_hop_df(comparison_df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, model_label), group in comparison_df.groupby(["family", "model_label"], observed=False):
        solved = group[group["accuracy"] >= threshold]["k"]
        exemplar = group.iloc[0]
        rows.append(
            {
                "family": family,
                "model_label": model_label,
                "max_solved_hop": int(solved.max()) if len(solved) > 0 else 0,
                "params": exemplar.get("params"),
                "assembly_size": exemplar.get("assembly_size"),
            }
        )
    return pd.DataFrame(rows)


def _train_limit(comparison_df: pd.DataFrame) -> int:
    if "k_train_max" in comparison_df and not comparison_df["k_train_max"].dropna().empty:
        return int(comparison_df["k_train_max"].dropna().max())
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


def _save_best_mlp_vs_ac(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    best_mlp = _best_mlp_label(comparison_df)
    subset = comparison_df[(comparison_df["family"] == "AC") | (comparison_df["model_label"] == best_mlp)]
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
    family_df = comparison_df[comparison_df["family"] == family].copy()
    if family == "MLP":
        family_df["row_label"] = family_df.apply(lambda row: f"{row['model_label']} ({int(row['params'])}p)", axis=1)
    else:
        family_df["row_label"] = family_df.apply(lambda row: f"{row['model_label']} (asm={int(row['assembly_size'])})", axis=1)
    pivot_df = family_df.pivot_table(index="row_label", columns="k", values="accuracy", aggfunc="mean")
    plt.figure(figsize=(10, 5.5))
    sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".2f", vmin=0.0, vmax=1.0)
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
    mlp_df = ms_df[ms_df["family"] == "MLP"].copy().sort_values("params")
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


def _save_ac_time_tradeoff(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    ac_df = comparison_df[comparison_df["family"] == "AC"].copy()
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
    best_subset = comparison_df[(comparison_df["family"] == "AC") | (comparison_df["model_label"] == best_mlp)]
    mlp_ms = _max_solved_hop_df(comparison_df)
    mlp_ms = mlp_ms[mlp_ms["family"] == "MLP"].copy().sort_values("params")
    ac_df = comparison_df[comparison_df["family"] == "AC"].copy()
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
