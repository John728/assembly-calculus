# Family AC Trace Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move AC trace diagnostics into the canonical seen/unseen AC family outputs, with one `k=6` trace example for each family.

**Architecture:** Reuse the existing `trace_plots` execution path in `run_experiment_suite.py` instead of maintaining a separate standalone visualization suite. The main work is config wiring, regression tests, and smoke verification that the trace artifacts land under `outputs/experiments/seen-ac/trace_plots/` and `outputs/experiments/unseen-ac/trace_plots/`.

**Tech Stack:** Python, YAML experiment configs, direct test-function harnesses via `venv/bin/python`, existing pointer trace visualization pipeline.

---

### Task 1: Add Failing Tests For Family Trace Outputs

**Files:**
- Modify: `tests/test_experiment_suite_config.py`
- Modify: `tests/test_run_experiment_suite.py`

**Step 1: Write the failing config test**

Add a test that loads `experiments/seen_ac.yaml` and `experiments/unseen_ac.yaml` and asserts both expose `trace_plots.enabled == true` with `hops == 6`.

**Step 2: Run the config test to verify it fails**

Run a direct harness with `venv/bin/python` that imports and calls the new config test function.

Expected: failure because the canonical AC family configs do not yet contain those `trace_plots` blocks.

**Step 3: Write the failing suite-runner test**

Add a test in `tests/test_run_experiment_suite.py` that uses a temporary AC suite config rooted in a temp directory and asserts `run_suite(...)` produces a `trace_plots/` directory beneath the suite output root.

**Step 4: Run the suite-runner test to verify it fails**

Run a direct harness with `venv/bin/python` that imports and calls the new runner test function.

Expected: failure because the test fixture/config does not yet exercise the family-local trace behavior.

### Task 2: Wire Trace Plots Into Canonical AC Family Configs

**Files:**
- Modify: `experiments/seen_ac.yaml`
- Modify: `experiments/unseen_ac.yaml`
- Optionally modify or remove: `experiments/ac_visualization.yaml`

**Step 1: Update the seen AC config**

Add a `trace_plots` block to `experiments/seen_ac.yaml` with one canonical example using `hops: 6`.

**Step 2: Update the unseen AC config**

Add a matching `trace_plots` block to `experiments/unseen_ac.yaml` with one canonical example using `hops: 6`.

**Step 3: Clean up the standalone AC visualization config**

Either remove `experiments/ac_visualization.yaml` or clearly demote it from the paper-facing workflow so it no longer competes with the family configs.

**Step 4: Run the Task 1 tests to verify they now pass**

Run the same direct `venv/bin/python` harnesses again.

Expected: both the config assertion and the suite-runner trace-output assertion pass.

### Task 3: Smoke-Verify Seen And Unseen AC Family Trace Outputs

**Files:**
- Verify: `outputs/experiments/seen-ac/trace_plots/`
- Verify: `outputs/experiments/unseen-ac/trace_plots/`

**Step 1: Run the seen AC family suite**

Run:

```bash
venv/bin/python run_experiment_suite.py --config experiments/seen_ac.yaml
```

**Step 2: Run the unseen AC family suite**

Run:

```bash
venv/bin/python run_experiment_suite.py --config experiments/unseen_ac.yaml
```

**Step 3: Verify trace plot outputs exist under family roots**

Use a direct `venv/bin/python` assertion script to confirm both family directories contain:

- `trace_plots/assembly_heatmap.png`
- `trace_plots/assembly_bars_over_time.png`
- `trace_plots/assembly_connectivity_graph.png`
- `trace_plots/assembly_weight_matrix.png`

**Step 4: Verify the normal family plot pack still exists**

Confirm each family still also has its normal `plots/` outputs, so the cleanup is additive to the family structure rather than replacing it.

### Task 4: Final Verification And Summary

**Files:**
- Review: `experiments/seen_ac.yaml`
- Review: `experiments/unseen_ac.yaml`
- Review: `run_experiment_suite.py`
- Review: `tests/test_experiment_suite_config.py`
- Review: `tests/test_run_experiment_suite.py`

**Step 1: Run the full direct verification bundle**

Run the direct test harnesses plus the two suite smoke commands again in one final pass.

**Step 2: Check git diff for scope**

Confirm the diff only contains the family trace-output cleanup and associated tests/config changes.

**Step 3: Summarize exact output locations**

Report the final canonical locations:

- `outputs/experiments/seen-ac/trace_plots/`
- `outputs/experiments/unseen-ac/trace_plots/`

and note whether `experiments/ac_visualization.yaml` was removed or left as legacy.
