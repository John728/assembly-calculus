# MNIST Sequence Probe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an organized, explicitly non-thesis MNIST probe that shows how a trained AC coding assembly responds when digits 0 through 9 are presented sequentially.

**Architecture:** Reuse the existing MNIST AC training path and add a small sequence evaluator that keeps the coding-area state continuous across stimulus changes. Add a new runner branch controlled by `sequence_digits`/`steps_per_digit`, plus a sequence-specific plotting path and README documentation under `results/mnist/probes/sequence_0_to_9/`.

**Tech Stack:** Python, NumPy, pandas/matplotlib plotting, existing `run_experiment_suite.py`, existing PYAC MNIST protocol functions, pytest.

---

### Task 1: Sequence Evaluator

**Files:**
- Modify: `pyac/src/pyac/tasks/mnist/protocol.py`
- Test: `tests/test_mnist_evaluation.py`

**Step 1: Write the failing test**

Add a test that imports `evaluate_mnist_sequence`, constructs a tiny fake network/task, presents two images for two steps each, and asserts:
- total rows equal `len(sequence_digits) * steps_per_digit`
- `phase_digit` changes according to the requested sequence
- `step_in_phase` resets per digit
- network reset happens once before the whole sequence, not between digits
- returned rows include `overlaps`, `trajectory`, `margin`, `prediction`, and `correct`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_mnist_evaluation.py::test_mnist_sequence_keeps_state_across_digit_changes -q`

Expected: FAIL with import/name error for `evaluate_mnist_sequence`.

**Step 3: Write minimal implementation**

Implement `evaluate_mnist_sequence(...)` using existing helpers:
- validate one representative image per requested digit
- reset once before sequence
- for each digit phase, apply that digit's stimulus for `steps_per_digit` steps
- call `network.step(..., plasticity_on=False, biases={coding_area: task.coding_bias})`
- record per-step overlaps, prediction, target/phase digit, correctness, margin, and stimulus metadata

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_mnist_evaluation.py::test_mnist_sequence_keeps_state_across_digit_changes -q`

Expected: PASS.

### Task 2: Runner Integration

**Files:**
- Modify: `experiment_suite/runners/mnist_ac_runner.py`
- Test: `tests/test_mnist_suite_runner.py`

**Step 1: Write the failing test**

Add a runner test where model config includes `sequence_digits: [0,1,2]` and `steps_per_digit: 3`. Monkeypatch `evaluate_mnist_sequence`, assert the runner trains once and calls the sequence evaluator instead of `evaluate_mnist_t_sweep`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_mnist_suite_runner.py::test_mnist_runner_dispatches_sequence_probe -q`

Expected: FAIL until runner imports/calls `evaluate_mnist_sequence`.

**Step 3: Write minimal implementation**

In `run_mnist_ac_job`, if `sequence_digits` exists:
- parse `sequence_digits` and `steps_per_digit`
- call `evaluate_mnist_sequence(...)`
- annotate rows with the same suite/family/model/output fields as normal MNIST rows
- set `k_test` and `internal_steps` from `sequence_step`

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_mnist_suite_runner.py::test_mnist_runner_dispatches_sequence_probe -q`

Expected: PASS.

### Task 3: Plotting and Organization

**Files:**
- Modify: `experiment_suite/plots.py`
- Modify: `run_experiment_suite.py` if needed
- Create: `experiments/mnist_ac_sequence_0_to_9.yaml`
- Modify: `results/mnist/README.md`

**Step 1: Add sequence plot generation**

If raw results have `experiment == "mnist_sequence"`, generate:
- `mnist_sequence_predictions_0_to_9.png`
- `mnist_sequence_overlaps_0_to_9.png`
- `mnist_sequence_margin_0_to_9.png`

**Step 2: Create organized config**

Create `experiments/mnist_ac_sequence_0_to_9.yaml` with output `results/mnist/probes/sequence_0_to_9`, seed `[42]`, tuned model parameters, `sequence_digits: [0,1,2,3,4,5,6,7,8,9]`, and `steps_per_digit: 3`.

**Step 3: Run focused tests**

Run: `PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_mnist_*.py -q`

Expected: all MNIST-focused tests pass.

### Task 4: Execute Probe

**Files:**
- Output: `results/mnist/probes/sequence_0_to_9/`
- Modify: `results/mnist/README.md`

**Step 1: Run the sequence config**

Run: `PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/mnist_ac_sequence_0_to_9.yaml`

Expected: writes `results/mnist/probes/sequence_0_to_9`.

**Step 2: Inspect outputs**

Verify `summary.csv`, `raw_results.csv`, `config_snapshot.yaml`, `plots/`, and local README/update are present.

**Step 3: Update documentation**

Document the sequence probe as an exploratory visualization, not primary thesis evidence.
