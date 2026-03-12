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
    seen["internal_steps"] = seen["internal_steps"] if "internal_steps" in seen else pd.NA
    seen["model_label"] = seen["model_name"]
    family_order = pd.CategoricalDtype(categories=["MLP", "AC"], ordered=True)
    seen["family"] = seen["family"].astype(family_order)
    return seen.sort_values(["family", "model_label", "k"], kind="stable").reset_index(drop=True)


def _best_mlp_label(comparison_df: pd.DataFrame) -> str:
    mlp_df = comparison_df[comparison_df["family"] == "MLP"]
    grouped = mlp_df.groupby("model_label", as_index=False)["accuracy"].mean()
    return str(grouped.sort_values(["accuracy", "model_label"], ascending=[False, True]).iloc[0]["model_label"])


def _max_solved_hop_df(comparison_df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, model_label), group in comparison_df.groupby(["family", "model_label"]):
        solved = group[group["accuracy"] >= threshold]["k"]
        rows.append({"family": family, "model_label": model_label, "max_solved_hop": int(solved.max()) if len(solved) > 0 else 0})
    return pd.DataFrame(rows)


def _save_accuracy_vs_hop(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=comparison_df, x="k", y="accuracy", hue="model_label", style="family", markers=True, dashes=False)
    train_limit = int(comparison_df["k_train_max"].dropna().max()) if "k_train_max" in comparison_df and not comparison_df["k_train_max"].dropna().empty else 4
    plt.axvline(x=train_limit, color="red", linestyle="--", linewidth=1.5, label="Train limit")
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
    train_limit = int(comparison_df["k_train_max"].dropna().max()) if "k_train_max" in comparison_df and not comparison_df["k_train_max"].dropna().empty else 4
    plt.axvline(x=train_limit, color="red", linestyle="--", linewidth=1.5, label="Train limit")
    plt.title("Seen Lists: AC vs Best MLP")
    plt.xlabel("Hop Count")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    path = output_dir / "accuracy_vs_hop_seen_best_mlp_vs_ac.png"
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


def _save_ac_time_vs_hop(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    ac_df = comparison_df[comparison_df["family"] == "AC"].copy()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=ac_df, x="k", y="internal_steps", marker="o")
    plt.title("Seen Lists: AC Internal Steps vs Hop Count")
    plt.xlabel("Hop Count")
    plt.ylabel("Internal Steps")
    plt.tight_layout()
    path = output_dir / "ac_time_vs_hop.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _save_paper_panel(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    best_mlp = _best_mlp_label(comparison_df)
    best_subset = comparison_df[(comparison_df["family"] == "AC") | (comparison_df["model_label"] == best_mlp)]
    ms_df = _max_solved_hop_df(comparison_df)
    ac_df = comparison_df[comparison_df["family"] == "AC"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    sns.lineplot(data=best_subset, x="k", y="accuracy", hue="model_label", style="family", markers=True, dashes=False, ax=axes[0])
    train_limit = int(comparison_df["k_train_max"].dropna().max()) if "k_train_max" in comparison_df and not comparison_df["k_train_max"].dropna().empty else 4
    axes[0].axvline(x=train_limit, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title("Accuracy vs Hop")
    axes[0].set_xlabel("Hop Count")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(-0.02, 1.05)
    sns.barplot(data=ms_df, x="model_label", y="max_solved_hop", hue="family", ax=axes[1])
    axes[1].set_title("Max Solved Hop")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Hop")
    axes[1].tick_params(axis="x", rotation=20)
    sns.lineplot(data=ac_df, x="k", y="internal_steps", marker="o", color="#1f77b4", ax=axes[2])
    axes[2].set_title("AC Time Budget")
    axes[2].set_xlabel("Hop Count")
    axes[2].set_ylabel("Internal Steps")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles=handles, labels=labels, loc="best", fontsize=8)
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.suptitle("Seen Lists: Protocol-Trained AC vs MLP Baseline", fontsize=14)
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
        _save_max_solved_hop(comparison_df, output_path),
        _save_ac_time_vs_hop(comparison_df, output_path),
        _save_paper_panel(comparison_df, output_path),
    ]
