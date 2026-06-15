# Pointer Theory Results Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up the result layout and add Theory-map-compliant pointer-chasing outputs.

**Architecture:** Keep existing MNIST outputs intact, add a repository-level results README, and implement a pointer-specific evaluation path that records per-instance rows rather than only aggregate accuracy. Reuse the existing proper-unseen pointer network/training path, but evaluate each `(L,t)` combination with frozen weights and save trajectories, path accuracy, and first-error diagnostics.

**Tech Stack:** Python, pytest, numpy, pandas, matplotlib/seaborn, existing `run_experiment_suite.py` YAML runner.

---

### Task 1: Repository Results Cleanup

**Files:**
- Create: `results/README.md`
- Modify: `results/mnist/README.md`

**Step 1: Write concise results README**

Document `results/mnist/` and planned `results/pointer/`, with explicit notes that MNIST sequence probes are exploratory.

**Step 2: Verify current result tree**

Run: inspect `results/` directory and ensure it contains only intended top-level result families.

### Task 2: Pointer Per-Instance Metrics

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Test: `tests/test_pointer_theory_metrics.py`

**Step 1: Write failing tests**

Test helper behavior for true pointer paths, path accuracy, first-error index, and per-instance row schema.

**Step 2: Implement minimal helpers**

Add a per-instance evaluation helper that returns rows with `experiment="pointer_chasing"`, `L`, `t`, `true_trajectory`, decoded `trajectory`, `path_accuracy`, `first_error_index`, final prediction/correctness, and frozen-eval metadata.

**Step 3: Run focused pointer tests**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_pointer_theory_metrics.py -q`

### Task 3: Runner Integration

**Files:**
- Modify: `experiment_suite/runners/ac_runner.py`
- Test: `tests/test_experiment_suite_runners.py`

**Step 1: Write failing runner test**

Verify a `theory_pointer: true` AC unseen job evaluates all configured `time_budgets` for all hop depths and returns per-instance rows instead of aggregate-only rows.

**Step 2: Implement minimal dispatch**

In proper-unseen AC jobs, if `theory_pointer: true`, call the per-instance evaluator with configured `time_budgets`; otherwise preserve existing behavior.

**Step 3: Run runner tests**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_experiment_suite_runners.py -q`

### Task 4: Pointer Plots

**Files:**
- Modify: `experiment_suite/plots.py`
- Modify: `run_experiment_suite.py`
- Test: `tests/test_pointer_theory_plots.py`

**Step 1: Write failing plot test**

Synthetic pointer rows should generate heatmap, accuracy-vs-time, accuracy-vs-depth, path-accuracy, and first-error plots.

**Step 2: Implement pointer plot dispatch**

Detect `experiment == "pointer_chasing"` before generic list-type plots and write pointer-specific figures under `plots/`.

**Step 3: Run plot tests**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_pointer_theory_plots.py tests/test_run_experiment_suite.py -q`

### Task 5: Config, Run, Docs

**Files:**
- Create: `experiments/pointer_ac_theory.yaml`
- Create: `results/pointer/README.md`
- Create/update: `results/pointer/theory/README.md`

**Step 1: Add small Theory pointer config**

Use proper-unseen pointer, output to `results/pointer/theory`, sweep `L=1..6`, `t=[0,1,2,3,4,6,8,10]`, and start with small sample counts for runtime.

**Step 2: Run focused tests**

Run pointer and suite tests before experiment execution.

**Step 3: Run experiment**

Run: `PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/pointer_ac_theory.yaml`

**Step 4: Write result README**

Document settings, generated files, plots, and initial interpretation.

**Step 5: Final verification**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_pointer*.py tests/test_experiment_suite_runners.py tests/test_run_experiment_suite.py -q`

No git commit is part of this plan unless explicitly requested.
