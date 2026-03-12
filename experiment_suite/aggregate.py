from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any


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
            writer.writerows(rows)
    return path


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    grouped: dict[tuple[object, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["family"], row["model_name"], row["list_type"], row["k_test"])
        entry = grouped.setdefault(
            key,
            {
                "family": row["family"],
                "model_name": row["model_name"],
                "list_type": row["list_type"],
                "k_test": row["k_test"],
                "mean_accuracy": 0.0,
                "num_rows": 0,
            },
        )
        entry["mean_accuracy"] += float(row["accuracy"])
        entry["num_rows"] += 1

    summary_rows = []
    for entry in grouped.values():
        summary_rows.append(
            {
                **entry,
                "mean_accuracy": entry["mean_accuracy"] / max(entry["num_rows"], 1),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    return path


def snapshot_config(config_path: str | Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "config_snapshot.yaml"
    shutil.copyfile(str(config_path), destination)
    return destination
