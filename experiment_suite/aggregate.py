from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


def _serialize_csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_raw_results(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "raw_results.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {key: _serialize_csv_cell(value) for key, value in row.items()}
                for row in rows
            )
    return path


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    grouped: dict[tuple[object, ...], dict[str, Any]] = {}
    for row in rows:
        summary_dimensions = {
            "suite": row.get("suite"),
            "experiment": row.get("experiment"),
            "family": row["family"],
            "model_name": row["model_name"],
            "list_type": row["list_type"],
            "k_test": row["k_test"],
        }
        if row.get("experiment") == "pointer_chasing":
            summary_dimensions["pointer_variant"] = row.get("pointer_variant")
            summary_dimensions["L"] = row.get("L")
            summary_dimensions["t"] = row.get("t")
            summary_dimensions["c"] = row.get("c")
        if row.get("family") == "MNIST_AC":
            for dimension in (
                "stimulus_mode",
                "t",
                "s",
                "ell",
                "cue_duration_s",
                "retention_ell",
                "presentation_rounds",
                "settle_steps",
                "normalization_on",
                "beta_train",
                "T_train",
                "plasticity_train_on",
                "plasticity_eval_on",
            ):
                if dimension in row:
                    summary_dimensions[dimension] = row.get(dimension)

        key = tuple(summary_dimensions.values())
        entry = grouped.setdefault(
            key,
            {
                **summary_dimensions,
                "mean_accuracy": 0.0,
                "mean_path_accuracy": 0.0,
                "mean_first_error_index": 0.0,
                "mean_correct_score": 0.0,
                "mean_strongest_wrong_score": 0.0,
                "mean_margin": 0.0,
                "mean_retention_time": 0.0,
                "median_retention_time_values": [],
                "stayed_correct_rate": 0.0,
                "became_correct_later_rate": 0.0,
                "retained_full_horizon_rate": 0.0,
                "num_path_accuracy_rows": 0,
                "num_first_error_rows": 0,
                "num_correct_score_rows": 0,
                "num_strongest_wrong_score_rows": 0,
                "num_margin_rows": 0,
                "num_retention_time_rows": 0,
                "num_stayed_correct_rows": 0,
                "num_became_correct_later_rows": 0,
                "num_retained_full_horizon_rows": 0,
                "num_rows": 0,
            },
        )
        entry["mean_accuracy"] += float(row["accuracy"])
        if row.get("path_accuracy") is not None:
            entry["mean_path_accuracy"] += float(row["path_accuracy"])
            entry["num_path_accuracy_rows"] += 1
        if row.get("first_error_index") is not None:
            entry["mean_first_error_index"] += float(row["first_error_index"])
            entry["num_first_error_rows"] += 1
        if row.get("correct_score") is not None:
            entry["mean_correct_score"] += float(row["correct_score"])
            entry["num_correct_score_rows"] += 1
        elif row.get("correct_overlap") is not None:
            entry["mean_correct_score"] += float(row["correct_overlap"])
            entry["num_correct_score_rows"] += 1
        if row.get("strongest_wrong_score") is not None:
            entry["mean_strongest_wrong_score"] += float(row["strongest_wrong_score"])
            entry["num_strongest_wrong_score_rows"] += 1
        elif row.get("strongest_wrong_overlap") is not None:
            entry["mean_strongest_wrong_score"] += float(row["strongest_wrong_overlap"])
            entry["num_strongest_wrong_score_rows"] += 1
        if row.get("margin") is not None:
            entry["mean_margin"] += float(row["margin"])
            entry["num_margin_rows"] += 1
        if row.get("retention_time") is not None:
            retention_time = float(row["retention_time"])
            entry["mean_retention_time"] += retention_time
            entry["median_retention_time_values"].append(retention_time)
            entry["num_retention_time_rows"] += 1
        if row.get("stayed_correct") is not None:
            entry["stayed_correct_rate"] += 1.0 if bool(row["stayed_correct"]) else 0.0
            entry["num_stayed_correct_rows"] += 1
        if row.get("became_correct_later") is not None:
            entry["became_correct_later_rate"] += 1.0 if bool(row["became_correct_later"]) else 0.0
            entry["num_became_correct_later_rows"] += 1
        if row.get("retained_full_horizon") is not None:
            entry["retained_full_horizon_rate"] += 1.0 if bool(row["retained_full_horizon"]) else 0.0
            entry["num_retained_full_horizon_rows"] += 1
        entry["num_rows"] += 1

    summary_rows = []
    for entry in grouped.values():
        path_count = entry["num_path_accuracy_rows"]
        error_count = entry["num_first_error_rows"]
        correct_score_count = entry["num_correct_score_rows"]
        strongest_wrong_score_count = entry["num_strongest_wrong_score_rows"]
        margin_count = entry["num_margin_rows"]
        retention_time_count = entry["num_retention_time_rows"]
        stayed_correct_count = entry["num_stayed_correct_rows"]
        became_correct_later_count = entry["num_became_correct_later_rows"]
        retained_full_horizon_count = entry["num_retained_full_horizon_rows"]
        median_retention_values = entry.pop("median_retention_time_values")
        summary_rows.append(
            {
                **entry,
                "mean_accuracy": entry["mean_accuracy"] / max(entry["num_rows"], 1),
                "mean_path_accuracy": (
                    entry["mean_path_accuracy"] / path_count
                    if path_count
                    else None
                ),
                "mean_first_error_index": (
                    entry["mean_first_error_index"] / error_count
                    if error_count
                    else None
                ),
                "mean_correct_score": (
                    entry["mean_correct_score"] / correct_score_count
                    if correct_score_count
                    else None
                ),
                "mean_strongest_wrong_score": (
                    entry["mean_strongest_wrong_score"] / strongest_wrong_score_count
                    if strongest_wrong_score_count
                    else None
                ),
                "mean_margin": (
                    entry["mean_margin"] / margin_count
                    if margin_count
                    else None
                ),
                "mean_retention_time": (
                    entry["mean_retention_time"] / retention_time_count
                    if retention_time_count
                    else None
                ),
                "median_retention_time": (
                    sorted(median_retention_values)[len(median_retention_values) // 2]
                    if median_retention_values
                    else None
                ),
                "stayed_correct_rate": (
                    entry["stayed_correct_rate"] / stayed_correct_count
                    if stayed_correct_count
                    else None
                ),
                "became_correct_later_rate": (
                    entry["became_correct_later_rate"] / became_correct_later_count
                    if became_correct_later_count
                    else None
                ),
                "retained_full_horizon_rate": (
                    entry["retained_full_horizon_rate"] / retained_full_horizon_count
                    if retained_full_horizon_count
                    else None
                ),
            }
        )

    fieldnames: list[str] = []
    for row in summary_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    return path


def snapshot_config(config_path: str | Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "config_snapshot.yaml"
    shutil.copyfile(str(config_path), destination)
    return destination
