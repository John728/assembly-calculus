from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_approximation_module(ac_root: Path):
    path = ac_root / "scripts" / "thesis_c" / "static" / "test_class_level_approximation.py"
    spec = importlib.util.spec_from_file_location("class_level_approximation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(data) - 1) * stats.sem(data))
    return mean, mean - half, mean + half


def decoded_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, instance_id, readout), group in frame.groupby(
        ["seed", "instance_id", "readout_r"], sort=True
    ):
        winner = int(group.sort_values("class").loc[group["overlap"].idxmax(), "class"])
        rows.append(
            {
                "seed": int(seed),
                "instance_id": int(instance_id),
                "readout_r": int(readout),
                "target": int(group["target"].iloc[0]),
                "prediction": winner,
                "correct": int(winner == int(group["target"].iloc[0])),
            }
        )
    return pd.DataFrame(rows)


def temporal_seed_summary(frame: pd.DataFrame, exposure: int) -> dict[str, object]:
    predictions = decoded_trajectory(frame)
    pivot = predictions.pivot(index="instance_id", columns="readout_r", values="prediction")
    switched = pivot.nunique(axis=1) > 1
    row: dict[str, object] = {
        "exposure": exposure,
        "seed": int(predictions["seed"].iloc[0]),
        "switch_fraction_first_five": float(switched.mean()),
    }
    for readout in sorted(predictions["readout_r"].unique()):
        subset = predictions[predictions["readout_r"] == readout]
        row[f"accuracy_r{readout}"] = float(subset["correct"].mean())
    return row


def summarise(metrics: pd.DataFrame, keys: list[str], value_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group in metrics.groupby(keys, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key))
        for column in value_columns:
            mean, low, high = mean_ci(group[column])
            row[column] = mean
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_ints, default=tuple(range(42, 52)))
    parser.add_argument("--exposures", type=parse_ints, default=(1, 10, 50))
    parser.add_argument("--test-per-class", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    approximation, source_script = load_approximation_module(args.ac_root)
    sys.path.insert(0, str(args.ac_root / "pyac" / "src"))
    import pyac.tasks.mnist as mnist

    train = mnist.load_mnist_split(args.data_dir, "train")
    test = mnist.load_mnist_split(args.data_dir, "test")
    train_images, train_labels, train_ids = approximation.balanced_subset(
        train.images, train.labels, max(args.exposures)
    )
    test_images, test_labels, test_ids = approximation.balanced_subset(
        test.images, test.labels, args.test_per_class
    )

    all_metrics: list[pd.DataFrame] = []
    selectivity_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    temporal_rows: list[dict[str, object]] = []

    for exposure in args.exposures:
        approximation.TRAIN_PER_CLASS = exposure
        for seed in args.seeds:
            print(f"MNIST exposure={exposure} seed={seed}", flush=True)
            network, task, _ = approximation.train_model(
                seed, train_images, train_labels, mnist
            )
            rows, support = approximation.evaluate_model(
                seed, network, task, test_images, test_labels, test_ids
            )
            frame = pd.DataFrame(rows)
            metrics = approximation.seed_readout_metrics(frame)
            metrics.insert(0, "exposure", exposure)
            all_metrics.append(metrics)
            temporal_rows.append(temporal_seed_summary(frame, exposure))

            coding = task.area_map["coding"]
            recurrent = network.weights[(coding, coding)]
            diagonal_mass = float(approximation.K * np.trace(support))
            total_mass = float(recurrent.sum())
            off_diagonal = support[~np.eye(10, dtype=bool)]
            selectivity_rows.append(
                {
                    "exposure": exposure,
                    "seed": seed,
                    "diagonal_weight_fraction": diagonal_mass / total_mass,
                    "diagonal_support_mean": float(np.diag(support).mean()),
                    "off_diagonal_support_mean": float(off_diagonal.mean()),
                    "off_to_diagonal_support_ratio": float(
                        off_diagonal.mean() / np.diag(support).mean()
                    ),
                }
            )
            for target in range(10):
                for source in range(10):
                    support_rows.append(
                        {
                            "exposure": exposure,
                            "seed": seed,
                            "target_class": target,
                            "source_class": source,
                            "support": float(support[target, source]),
                        }
                    )

    metrics = pd.concat(all_metrics, ignore_index=True)
    selectivity = pd.DataFrame(selectivity_rows)
    support = pd.DataFrame(support_rows)
    temporal = pd.DataFrame(temporal_rows)

    metric_columns = [
        column for column in metrics.columns if column not in {"exposure", "seed", "readout_r"}
    ]
    metric_summary = summarise(metrics, ["exposure", "readout_r"], metric_columns)
    selectivity_summary = summarise(
        selectivity,
        ["exposure"],
        [
            "diagonal_weight_fraction",
            "diagonal_support_mean",
            "off_diagonal_support_mean",
            "off_to_diagonal_support_ratio",
        ],
    )
    temporal_summary = summarise(
        temporal,
        ["exposure"],
        [column for column in temporal.columns if column not in {"exposure", "seed"}],
    )

    metrics.to_csv(args.output / "mnist_metrics_per_seed_readout.csv", index=False)
    metric_summary.to_csv(args.output / "mnist_metrics_summary.csv", index=False)
    selectivity.to_csv(args.output / "mnist_selectivity_per_seed.csv", index=False)
    selectivity_summary.to_csv(args.output / "mnist_selectivity_summary.csv", index=False)
    support.to_csv(args.output / "mnist_support_matrices.csv", index=False)
    temporal.to_csv(args.output / "mnist_temporal_per_seed.csv", index=False)
    temporal_summary.to_csv(args.output / "mnist_temporal_summary.csv", index=False)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.ac_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=args.ac_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    metadata = {
        "protocol": {
            "seeds": list(args.seeds),
            "exposures": list(args.exposures),
            "train_pool_per_class": max(args.exposures),
            "test_per_class": args.test_per_class,
            "readouts": list(range(1, approximation.READOUTS + 1)),
            "n": approximation.N,
            "k": approximation.K,
            "p": approximation.P,
            "beta": approximation.BETA,
            "input_k": approximation.RAW_INPUT_K,
            "training": "class organised; nested first-N images per class",
            "evaluation": "held input; plasticity disabled",
            "prediction_models": {
                "diagonal": "diag(Lambda) * overlap",
                "full_matrix": "Lambda @ overlap",
            },
        },
        "source": {
            "repository_revision": revision,
            "repository_dirty": bool(dirty),
            "source_script": str(source_script),
            "source_script_sha256": hashlib.sha256(source_script.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "train_ids_sha256": hashlib.sha256(np.asarray(train_ids).tobytes()).hexdigest(),
            "test_ids_sha256": hashlib.sha256(np.asarray(test_ids).tobytes()).hexdigest(),
        },
    }
    (args.output / "mnist_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    headline = metric_summary[
        metric_summary["readout_r"].isin([1, approximation.READOUTS])
    ][
        [
            "exposure",
            "readout_r",
            "scalar_pearson_r",
            "matrix_pearson_r",
            "scalar_ordering_accuracy",
            "matrix_ordering_accuracy",
            "scalar_normalised_mae",
            "matrix_normalised_mae",
        ]
    ]
    print("\nSELECTIVITY")
    print(selectivity_summary.to_string(index=False))
    print("\nHEADLINE METRICS")
    print(headline.to_string(index=False))
    print("\nTEMPORAL")
    print(temporal_summary.to_string(index=False))


if __name__ == "__main__":
    main()
