# MNIST Sequence Hold Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an organized MNIST sequence hold-length sweep that tests whether longer presentation of each digit helps the continuous AC state switch out of sticky attractors.

**Architecture:** Reuse the existing trained MNIST AC sequence evaluator, but let one suite model evaluate multiple `steps_per_digit` values after a single training run. Add hold-sweep plots that summarize switching latency and final per-phase accuracy across hold lengths, and store outputs under `results/mnist/probes/sequence_hold_sweep/`.

**Tech Stack:** Python, NumPy, pandas/seaborn/matplotlib, pytest, existing YAML suite runner.

---

### Task 1: Runner Support

**Files:**
- Modify: `experiment_suite/runners/mnist_ac_runner.py`
- Test: `tests/test_mnist_suite_runner.py`

**Step 1: Write the failing test**

Add a test where a sequence model specifies `steps_per_digit_values: [2, 4]`. Monkeypatch training/evaluation and assert training runs once while `evaluate_mnist_sequence` runs once per hold length.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_mnist_suite_runner.py::test_mnist_runner_dispatches_sequence_hold_sweep_after_one_training_run -q`

Expected: FAIL because the runner ignores `steps_per_digit_values`.

**Step 3: Implement minimal runner logic**

If `sequence_digits` exists, parse `steps_per_digit_values` when provided; otherwise use the existing single `steps_per_digit`. Evaluate each hold length against the same trained network/task and label rows with `hold_steps`/`steps_per_digit` and model name suffixes.

**Step 4: Run test to verify it passes**

Run the focused test above.

### Task 2: Hold-Sweep Plots

**Files:**
- Modify: `experiment_suite/plots.py`
- Test: `tests/test_mnist_plots.py`

**Step 1: Write the failing test**

Add a plot test with sequence rows for two hold lengths. Expect hold-sweep PNGs for final accuracy, switch latency, and prediction timelines.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_mnist_plots.py::test_generate_mnist_ac_plots_writes_hold_sweep_pngs -q`

Expected: FAIL because hold-sweep plots do not exist.

**Step 3: Implement minimal plotting**

When sequence rows contain multiple `steps_per_digit` values, generate three additional plots under the same output directory. Keep standard MNIST plots excluded for sequence-only results.

**Step 4: Run test to verify it passes**

Run the focused plot test.

### Task 3: Config, Results, Docs

**Files:**
- Create: `experiments/mnist_ac_sequence_hold_sweep.yaml`
- Create: `results/mnist/probes/sequence_hold_sweep/README.md`
- Modify: `results/mnist/README.md`

**Step 1: Add config**

Use the tuned MNIST AC settings, `sequence_digits: [0,1,2,3,4,5,6,7,8,9]`, and `steps_per_digit_values: [3, 10, 30, 100]`.

**Step 2: Run experiment**

Run: `PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/mnist_ac_sequence_hold_sweep.yaml`

Expected: output under `results/mnist/probes/sequence_hold_sweep/`.

**Step 3: Summarize and document**

Use `raw_results.csv` to summarize final prediction per digit and whether longer holds improved switching. Document this as exploratory, not a thesis benchmark.

### Task 4: Verification

**Files:**
- Test all MNIST-focused tests.

**Step 1: Run focused suite**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_mnist_*.py -q`

Expected: all tests pass.

**Step 2: Verify result organization**

Confirm `results/mnist/probes/sequence_hold_sweep/` contains raw results, summary, config snapshot, README, and hold-sweep plots.
