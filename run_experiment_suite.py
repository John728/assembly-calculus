from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from experiment_suite.aggregate import snapshot_config, write_raw_results, write_summary
from experiment_suite.config import load_suite_config
from experiment_suite.jobs import expand_jobs


def _dispatch_job(job):
    if job.family == "MLP":
        from experiment_suite.runners.mlp_runner import run_mlp_job

        return run_mlp_job(job), None
    if job.family == "AC":
        from experiment_suite.runners.ac_runner import run_ac_job_with_artifacts

        return run_ac_job_with_artifacts(job)
    raise ValueError(f"Unsupported family: {job.family}")


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _generate_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return

    families = {str(row["family"]) for row in rows}
    list_types = {str(row["list_type"]) for row in rows}
    if len(list_types) != 1:
        return

    from experiment_suite import plots as suite_plots

    list_type = next(iter(list_types))
    raw_results_csv = output_dir / "raw_results.csv"
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        shutil.rmtree(plots_dir)

    if families == {"MLP"}:
        if list_type == "Seen":
            suite_plots.generate_seen_mlp_plots(raw_results_csv, plots_dir)
        elif list_type == "Unseen":
            suite_plots.generate_unseen_mlp_plots(raw_results_csv, plots_dir)
        return

    if families == {"AC"}:
        if list_type == "Seen":
            suite_plots.generate_seen_ac_plots(raw_results_csv, plots_dir)
        elif list_type == "Unseen":
            suite_plots.generate_unseen_ac_plots(raw_results_csv, plots_dir)
        return

    if families == {"MLP", "AC"}:
        if list_type == "Seen":
            suite_plots.generate_seen_suite_plots(raw_results_csv, plots_dir)
        elif list_type == "Unseen":
            suite_plots.generate_unseen_suite_plots(raw_results_csv, plots_dir)


def run_suite(config_path: str | Path) -> Path:
    config = load_suite_config(config_path)
    jobs = expand_jobs(config)
    output_dir = Path(config.output_dir)

    rows = []
    trace_artifact = None
    for job in jobs:
        job_rows, artifact = _dispatch_job(job)
        rows.extend(job_rows)
        if trace_artifact is None and artifact is not None:
            trace_artifact = artifact

    write_raw_results(rows, output_dir)
    write_summary(rows, output_dir)
    snapshot_config(config.config_path, output_dir)
    _generate_plots(rows, output_dir)

    if config.trace_plots and config.trace_plots.get("enabled") and trace_artifact is not None:
        from pyac.tasks.pointer.trace import record_rollout_trace
        from pyac.tasks.pointer.visualize import render_trace_visualizations

        trace = record_rollout_trace(
            trace_artifact["network"],
            trace_artifact["task"],
            trace_artifact["lists"],
            list_idx=_as_int(config.trace_plots.get("list_idx"), 0),
            start_node=_as_int(config.trace_plots.get("start_node"), 0),
            hops=_as_int(config.trace_plots.get("hops"), 1),
            settle_steps=_as_int(config.trace_plots.get("settle_steps"), 1),
            internal_steps=10,
        )
        render_trace_visualizations(trace, output_dir / "trace_plots")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run config-driven pointer experiment suite")
    parser.add_argument("--config", required=True, help="Path to suite YAML config")
    args = parser.parse_args()
    out_dir = run_suite(args.config)
    print(out_dir)


if __name__ == "__main__":
    main()
