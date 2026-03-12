from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_suite.plots import generate_seen_suite_plots


ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE_RESULTS = ROOT / "outputs" / "seen-comparison" / "raw_results.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "seen-comparison" / "plots"


def build_seen_comparison_df(mlp_csv: str | Path, ac_csv: str | Path) -> pd.DataFrame:
    mlp_df = pd.read_csv(mlp_csv)
    ac_df = pd.read_csv(ac_csv)

    mlp_seen = mlp_df[mlp_df["List Type"] == "Seen"].copy()
    mlp_seen["family"] = "MLP"
    mlp_seen["model_label"] = mlp_seen["Model"]
    mlp_seen["list_type"] = mlp_seen["List Type"]
    mlp_seen["k"] = mlp_seen["k"].astype(int)
    mlp_seen["accuracy"] = mlp_seen["Accuracy"].astype(float)
    mlp_seen["internal_steps"] = pd.Series([None] * len(mlp_seen), dtype=object)

    ac_seen = ac_df[ac_df["List Type"] == "Seen"].copy()
    ac_seen["family"] = "AC"
    ac_seen["model_label"] = ac_seen["Model"]
    ac_seen["list_type"] = ac_seen["List Type"]
    ac_seen["k"] = ac_seen["k"].astype(int)
    ac_seen["accuracy"] = ac_seen["Accuracy"].astype(float)
    ac_seen["internal_steps"] = ac_seen["Internal Steps"].astype(int)

    combined = pd.concat(
        [
            mlp_seen[["family", "model_label", "list_type", "k", "accuracy", "internal_steps"]],
            ac_seen[["family", "model_label", "list_type", "k", "accuracy", "internal_steps"]],
        ],
        ignore_index=True,
    )
    family_order = pd.CategoricalDtype(categories=["MLP", "AC"], ordered=True)
    combined["family"] = combined["family"].astype(family_order)
    return combined.sort_values(["family", "model_label", "k"], kind="stable").reset_index(drop=True)


def generate_seen_comparison_plots(mlp_csv: str | Path, ac_csv: str | Path, output_dir: str | Path) -> list[Path]:
    combined = build_seen_comparison_df(mlp_csv, ac_csv)
    suite_like = combined.rename(columns={"model_label": "model_name"}).copy()
    suite_like["k_test"] = suite_like["k"]
    suite_like["k_train_max"] = 4
    temp_csv = Path(output_dir) / "_tmp_seen_comparison.csv"
    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    suite_like.to_csv(temp_csv, index=False)
    paths = generate_seen_suite_plots(temp_csv, output_dir)
    temp_csv.unlink(missing_ok=True)
    return paths


def main() -> None:
    if not DEFAULT_SUITE_RESULTS.exists():
        raise FileNotFoundError(f"Expected standardized suite results at {DEFAULT_SUITE_RESULTS}")
    paths = generate_seen_suite_plots(DEFAULT_SUITE_RESULTS, DEFAULT_OUTPUT_DIR)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
