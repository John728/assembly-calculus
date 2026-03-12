# Scalable Pointer Experiments Design

## Goal

Build a unified, config-driven experiment platform for pointer-chasing research so both MLP and Assembly Calculus experiments can scale cleanly from MVP runs to large multi-seed sweeps on a high-performance machine.

This design also adds a proper AC internal-state visualization path so the thesis can show not just accuracy curves, but the internal temporal structure of learned assembly activity and connectivity.

## Why Change the Current Setup

The current repo has separate one-off runners such as `eval_mlp_baseline.py`, `eval_ac_seen.py`, and `plot_seen_ac_vs_mlp.py`. That is fine for early exploration, but it does not scale well for:

- large hyperparameter sweeps
- multi-seed experiments
- consistent seen/unseen comparisons across model families
- reproducible thesis figures and tables
- structural AC visualizations tied to real experiment outputs

The next stage should therefore move from script-per-experiment to a reusable experiment platform.

## Architecture Overview

Add a unified runner, for example `run_experiment_suite.py`, that reads a YAML config from `experiments/` and executes a full experiment suite.

The platform should have five layers:

1. `Config layer`: YAML files defining datasets, model families, hyperparameter grids, seeds, metrics, and requested plots.
2. `Job expansion layer`: converts config grids into concrete experiment jobs.
3. `Execution layer`: runs MLP and AC jobs using a shared result schema.
4. `Aggregation layer`: writes raw per-job results and derived summaries.
5. `Visualization layer`: produces accuracy plots, scaling plots, and AC trace/connectivity figures from saved results.

This keeps the experiment definition stable even when the implementation of any one model family evolves.

## Config-Driven Sweep Design

Store experiment definitions in files such as:

- `experiments/seen_small.yaml`
- `experiments/seen_large.yaml`
- `experiments/unseen_large.yaml`
- `experiments/ac_visualization.yaml`

Each config should define:

- suite name and output directory
- dataset settings (`N`, number of seen/unseen lists, hop ranges, samples per list)
- seeds
- enabled model families (`MLP`, `AC`)
- family-specific grids
- metrics to compute
- plots to generate

The job expansion layer should produce one concrete job per `(family, config choice, seed, dataset condition)` combination. This makes it easy to scale on a larger machine by partitioning jobs at the config level rather than editing Python code.

## Unified Result Schema

All model families should write to the same raw results table so downstream summaries and plots do not need family-specific special cases.

Recommended core fields:

- `suite`
- `seed`
- `family`
- `model_name`
- `list_type`
- `N`
- `num_train_lists`
- `num_test_lists`
- `k_train_min`
- `k_train_max`
- `k_test`
- `accuracy`
- `internal_steps`
- `params`
- `runtime_ms`

Recommended optional fields:

- MLP: `layers`, `hidden_dim`, `lr`, `epochs`
- AC: `assembly_size`, `density`, `plasticity`, `presentation_rounds`, `transition_rounds`, `association_steps`, `teacher_strength`

Outputs per suite should include:

- `outputs/<suite>/raw_results.csv`
- `outputs/<suite>/summary.csv`
- `outputs/<suite>/plots/*.png`
- `outputs/<suite>/config_snapshot.yaml`

## Execution Model

The first version can remain single-process and deterministic, but the design should make parallelization straightforward later.

- one job object = one independent unit of work
- job execution returns rows in the unified result schema
- aggregation appends rows after each job or after each batch
- the same runner can later support job sharding across processes or machines

This allows the MVP platform to stay simple while still preparing for super-PC scale.

## Model Integration Strategy

### MLP

Refactor the existing MLP experiment logic into reusable functions for:

- generating seen/unseen list splits
- training a single MLP config
- evaluating by hop count
- returning standardized result rows

### AC

Refactor the current AC seen-list experiment into reusable suite functions as well. The seen-list AC path should stay explicitly framed as a list-specific assembly memorization protocol with learned recurrent transitions.

For future unseen-list AC experiments, add a separate protocol path rather than overloading the seen-list assumptions.

## Plotting Design

The plotting layer should read the standardized results, not call model code directly.

Core thesis plots for the platform:

- accuracy vs hop count
- best-model comparison across families
- max solved hop vs model size / time budget
- multi-seed summary plots with means and error bars
- runtime vs accuracy tradeoff

This separates expensive experiment execution from cheap figure generation.

## AC Internal-State Visualization Design

Add a dedicated AC trace visualization module that can record and display the internal rollout of a single evaluated example.

### Plot 1: Assembly-over-time heatmap

Purpose: show that the active neural population moves through learned list-node assemblies over internal time.

Recommended encoding:

- x-axis: internal time step
- y-axis: neuron index
- group neurons by assembly with visible separators
- color neurons by assembly identity or firing state
- optionally annotate the expected pointer path above the plot

This should let the reader see the traversal unfold step by step.

### Plot 2: Assembly connectivity graph

Purpose: show that recurrent connectivity has learned the pointer structure.

Recommended encoding:

- one node per learned assembly
- node color = list identity and/or node identity
- edge width/intensity = learned recurrent weight between assemblies
- optional highlighted rollout path for a selected example

This visualizes the structural memory that supports the temporal rollout.

### Plot 3: Assembly-to-assembly weight matrix

Purpose: provide a more rigorous alternative to the graph view.

Recommended encoding:

- rows = source assemblies
- columns = destination assemblies
- value = aggregated recurrent weight mass between assembly pairs

This is useful when the graph becomes too dense for larger suites.

## AC Trace Recording Requirements

The visualization system should use actual recorded activations from rollout, not reconstructed trajectories.

For one selected example, record:

- active neurons at each step
- assembly membership of those neurons
- final decoded assembly
- aggregated inter-assembly weights relevant to the example

This makes the visual story faithful to the protocol rather than illustrative only.

## Thesis Framing Constraints

- Seen-list AC should be labeled honestly as memorization with learned recurrent traversal.
- Unseen-list claims should only be made after a separate unseen-list AC protocol exists.
- Visualizations should distinguish symbolic assembly identity from raw neuron identity.
- Large-scale sweep plots should always separate seen and unseen conditions.

## Testing Strategy

Use a staged approach.

1. Unit-level checks for config loading and job expansion.
2. Small integration tests for one tiny suite that runs both AC and MLP and writes standardized outputs.
3. Plot tests that verify expected files are produced from fixture CSVs.
4. AC trace tests that verify recorded activations, assembly assignments, and weight summaries have the expected schema.

## MVP-to-Scale Roadmap

### MVP

- one config-driven runner
- one standardized results schema
- seen-list MLP and AC support
- basic comparison plots
- one AC trace heatmap and one connectivity-style visualization

### Scale-Up Phase

- unseen-list AC path
- larger model grids
- multi-seed sweeps
- summary tables with error bars
- sharded job execution
- paper-ready figure packs from saved outputs only

### Starter Config Guidance

- `experiments/seen_small.yaml`: laptop-scale smoke test for the unified runner.
- `experiments/seen_large.yaml`: first large seen-list suite for the bigger machine.
- `experiments/unseen_large.yaml`: unseen-list MLP baseline suite while unseen-AC protocol work is still separate.
- `experiments/seen_ac_mlp_scaling.yaml`: direct AC-vs-MLP scaling comparison for paper figures.

## Expected Outcome

This design gives a clean research workflow:

- define a suite in YAML
- run it once on any machine size
- collect standardized outputs
- regenerate comparison plots and AC internal-state figures without rerunning models

That should make the next round of thesis experiments much easier to scale, defend, and present.
