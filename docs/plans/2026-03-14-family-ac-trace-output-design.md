# Family AC Trace Output Design

## Goal

Move the AC trace diagnostics into the canonical experiment-family output structure so seen and unseen AC each produce their own trace plots under `outputs/experiments/...`, with a single canonical `k=6` example per family.

## Approved Scope

- Preserve the existing additive AC trace artifact `assembly_bars_over_time.png`.
- Stop relying on the standalone `outputs/ac-visualization` directory for the paper-facing workflow.
- Add one canonical trace example to `experiments/seen_ac.yaml` and one to `experiments/unseen_ac.yaml`.
- Set both canonical trace examples to `hops: 6`.
- Write AC trace diagnostics into:
  - `outputs/experiments/seen-ac/trace_plots/`
  - `outputs/experiments/unseen-ac/trace_plots/`
- Keep standard family plots in each family's `plots/` directory.

## Current State

- `run_experiment_suite.py` already supports suite-local trace generation via `trace_plots.enabled`.
- The mismatch is structural, not architectural: `experiments/ac_visualization.yaml` is a standalone suite writing to `outputs/ac-visualization`, while the paper-facing workflow now lives under `outputs/experiments/seen-ac` and `outputs/experiments/unseen-ac`.
- Because trace generation is already output-dir-relative, the cleanup is mostly config and verification work.

## Design

### 1. Canonical trace ownership

Seen AC and unseen AC each own one trace example. The trace plots are treated as family-local diagnostics, not a separate experiment class.

That means:

- `experiments/seen_ac.yaml` gets a `trace_plots` block.
- `experiments/unseen_ac.yaml` gets a `trace_plots` block.
- Each block specifies a single example with `hops: 6`.

This keeps the paper workflow simple: run the family config, then inspect CSVs, normal plots, and trace diagnostics in one directory tree.

### 2. Output layout

Each AC family directory remains self-contained:

- `raw_results.csv`
- `summary.csv`
- `plots/`
- `trace_plots/`

The `trace_plots/` directory contains the existing AC trace diagnostics, including:

- `assembly_heatmap.png`
- `assembly_bars_over_time.png`
- `assembly_connectivity_graph.png`
- `assembly_weight_matrix.png`

### 3. Legacy visualization config

`experiments/ac_visualization.yaml` should stop competing with the canonical paper workflow.

For this cleanup pass, the simplest acceptable outcome is to remove it from the active workflow and treat it as legacy or remove it entirely if nothing still depends on it.

### 4. Verification contract

The suite runner contract becomes:

- if an AC family config enables `trace_plots`, then running that family suite must write trace diagnostics into that family's own `trace_plots/` directory
- seen AC must support a canonical `k=6` trace path
- unseen AC must support a canonical `k=6` trace path

## Non-Goals

- Do not redesign the trace plots themselves in this cleanup.
- Do not create one trace plot set per model in the first version.
- Do not change the standard `plots/` pack layout.
- Do not change MLP family output structure.

## Risks

- Unseen AC trace generation at `k=6` depends on the current unseen runner path remaining stable under canonical family defaults.
- Removing `experiments/ac_visualization.yaml` without verifying references could break any manual habit or doc that still points to it.

## Mitigations

- Add tests that prove trace plots are written under family roots.
- Smoke-run both canonical AC family configs after the change.
- Only remove the standalone config after confirming the family configs fully cover the paper-facing use case.
