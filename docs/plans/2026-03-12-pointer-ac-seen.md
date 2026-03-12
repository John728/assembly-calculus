# Pointer AC Seen-Lists Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a proper protocol-trained Assembly Calculus seen-lists pointer-chasing experiment, run it, and save reproducible results for comparison against the MLP baseline.

**Architecture:** Add a new `pyac.tasks.pointer` package that owns pointer data generation, training protocol, and hop-sweep metrics. Keep the experiment runner separate at the repo root so it mirrors `eval_mlp_baseline.py` while using PyAC task functions instead of `nn.Module` training.

**Tech Stack:** Python, NumPy, PyTorch only for tensor-friendly sample containers if needed, PyAC core/network primitives, pytest, pandas/csv output.

---

### Task 1: Add pointer-task data utilities

**Files:**
- Create: `pyac/src/pyac/tasks/pointer/__init__.py`
- Create: `pyac/src/pyac/tasks/pointer/data.py`
- Test: `pyac/tests/integration/test_pointer_task.py`

**Step 1: Write the failing test**

Add tests that verify unique full-cycle list generation and pointer-chasing target computation on tiny deterministic examples.

**Step 2: Run test to verify it fails**

Run: `pytest pyac/tests/integration/test_pointer_task.py -q`
Expected: FAIL because the pointer task module does not exist yet.

**Step 3: Write minimal implementation**

Implement data helpers for generating unique full-cycle lists and for computing the target node after `k` hops.

**Step 4: Run test to verify it passes**

Run: `pytest pyac/tests/integration/test_pointer_task.py -q`
Expected: PASS for the new data tests.

### Task 2: Build the pointer protocol

**Files:**
- Create: `pyac/src/pyac/tasks/pointer/protocol.py`
- Modify: `pyac/src/pyac/tasks/pointer/__init__.py`
- Test: `pyac/tests/integration/test_pointer_task.py`

**Step 1: Write the failing test**

Add tests that verify `build_pointer_network(...)` constructs the expected areas and that training learns node prototypes on a tiny setup.

**Step 2: Run test to verify it fails**

Run: `pytest pyac/tests/integration/test_pointer_task.py -q`
Expected: FAIL because the protocol functions do not exist.

**Step 3: Write minimal implementation**

Implement the task-specific network builder, reset/stimulus helpers, node-assembly training, transition training, and pointer rollout / decode functions.

**Step 4: Run test to verify it passes**

Run: `pytest pyac/tests/integration/test_pointer_task.py -q`
Expected: PASS for protocol construction and deterministic training tests.

### Task 3: Add seen-list metrics and experiment runner

**Files:**
- Create: `pyac/src/pyac/tasks/pointer/metrics.py`
- Create: `eval_ac_seen.py`
- Modify: `pyac/src/pyac/tasks/pointer/__init__.py`
- Test: `pyac/tests/integration/test_pointer_task.py`

**Step 1: Write the failing test**

Add tests that verify the seen-list hop sweep returns records with the expected fields and nontrivial accuracy on a tiny deterministic problem.

**Step 2: Run test to verify it fails**

Run: `pytest pyac/tests/integration/test_pointer_task.py -q`
Expected: FAIL because metrics and runner-facing helpers do not exist.

**Step 3: Write minimal implementation**

Implement hop-sweep evaluation helpers plus a CLI runner that trains on seen lists, evaluates accuracy by hop count, and writes CSV output.

**Step 4: Run test to verify it passes**

Run: `pytest pyac/tests/integration/test_pointer_task.py -q`
Expected: PASS for metric schema and end-to-end tiny experiment tests.

### Task 4: Run the experiment and inspect the story

**Files:**
- Modify: `eval_ac_seen.py`
- Output: `outputs_ac_seen/ac_seen_results.csv`

**Step 1: Run a small smoke experiment**

Run: `python eval_ac_seen.py --num_train_lists 8 --samples_per_list_train 8 --samples_per_list_eval 8 --train_rounds 4 --k_train_max 4 --k_test_max 6`
Expected: completes successfully and writes a small result CSV.

**Step 2: Inspect the result curve**

Check whether seen-list accuracy remains high beyond the MLP training range when enough internal steps are available.

**Step 3: Adjust protocol hyperparameters minimally if needed**

Tune only the parameters needed to get a clean seen-list story: assembly size, density, plasticity, settle steps, transition rounds.

**Step 4: Run the target experiment**

Run: `python eval_ac_seen.py`
Expected: writes the main seen-list results to `outputs_ac_seen/ac_seen_results.csv`.

### Task 5: Verify and summarize

**Files:**
- Test: `pyac/tests/integration/test_pointer_task.py`
- Output: `outputs_ac_seen/ac_seen_results.csv`

**Step 1: Run verification**

Run: `pytest pyac/tests/integration/test_pointer_task.py -q`
Expected: PASS.

**Step 2: Re-run the experiment command used for the final result**

Expected: completes without errors and reproduces the output file.

**Step 3: Compare against MLP baseline**

Use `outputs_mlp_baseline/mlp_results.csv` and the AC result CSV to extract the thesis-relevant narrative for seen lists.
