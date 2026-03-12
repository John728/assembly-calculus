from __future__ import annotations

import argparse
from pathlib import Path

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

    if config.trace_plots and config.trace_plots.get("enabled") and trace_artifact is not None:
        from pyac.tasks.pointer.trace import record_rollout_trace
        from pyac.tasks.pointer.visualize import render_trace_visualizations

        trace = record_rollout_trace(
            trace_artifact["network"],
            trace_artifact["task"],
            trace_artifact["lists"],
            list_idx=int(config.trace_plots.get("list_idx", 0)),
            start_node=int(config.trace_plots.get("start_node", 0)),
            hops=int(config.trace_plots.get("hops", 1)),
            settle_steps=int(config.trace_plots.get("settle_steps", 1)),
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
