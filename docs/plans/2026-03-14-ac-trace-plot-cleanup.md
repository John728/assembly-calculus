# AC Trace Plot Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up AC family trace diagnostics by shortening the canonical trace example to `hops: 4`, moving pointer-list text out of the main plots, and adding a separate `pointer_reference.png` artifact.

**Architecture:** Keep the existing family-local trace pipeline in `run_experiment_suite.py` and `render_trace_visualizations(...)`. The work is a narrow presentation cleanup: update family configs, add regression tests, adjust the visualization writer to emit the new pointer-reference artifact, and verify both canonical AC families regenerate the expected trace outputs.

**Tech Stack:** Python, matplotlib, existing pointer trace visualization helpers, direct `venv/bin/python` test harnesses

---

### Task 1: Add failing regression tests for the cleaned trace artifact set

**Files:**
- Modify: `tests/test_pointer_visualize.py`
- Modify: `tests/test_experiment_suite_config.py`
- Modify: `tests/test_suite_trace_plots.py`

**Step 1: Write the failing test**

Add/adjust tests so they require:
- `pointer_reference.png` to be written by `render_trace_visualizations(...)`
- canonical AC family configs to use `trace_plots.hops == 4`
- suite-level trace hooks to write all five artifacts:
  - `assembly_heatmap.png`
  - `assembly_bars_over_time.png`
  - `assembly_connectivity_graph.png`
  - `assembly_weight_matrix.png`
  - `pointer_reference.png`

**Step 2: Run test to verify it fails**

Run direct harnesses with `venv/bin/python` for the new/changed test functions.

Expected:
- config test fails because canonical AC family configs still use `hops: 6`
- visualization/suite tests fail because `pointer_reference.png` does not exist yet

**Step 3: Write minimal implementation**

Do not implement yet; stop after confirming the expected failures.

**Step 4: Run test to verify it passes**

Skip until Tasks 2 and 3 are complete.

**Step 5: Commit**

Defer commit until the implementation and verification tasks are complete.

### Task 2: Update canonical AC family trace configs

**Files:**
- Modify: `experiments/seen_ac.yaml`
- Modify: `experiments/unseen_ac.yaml`

**Step 1: Write the failing test**

Use the config regression from Task 1 as the failing test.

**Step 2: Run test to verify it fails**

Run the direct config harness and confirm failure on `trace_plots.hops == 4`.

**Step 3: Write minimal implementation**

Change both canonical family configs so their trace blocks use:

```yaml
trace_plots:
  enabled: true
  list_idx: 0
  start_node: 0
  hops: 4
```

**Step 4: Run test to verify it passes**

Rerun the direct config harness and confirm success.

**Step 5: Commit**

Defer commit until visualization output changes are also complete.

### Task 3: Add `pointer_reference.png` and declutter the main AC trace plots

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/visualize.py`
- Test: `tests/test_pointer_visualize.py`
- Test: `tests/test_suite_trace_plots.py`

**Step 1: Write the failing test**

Use the failing tests from Task 1 that require `pointer_reference.png` in both direct visualization and suite trace-hook paths.

**Step 2: Run test to verify it fails**

Run direct harnesses and confirm the missing-artifact failure.

**Step 3: Write minimal implementation**

In `pyac/src/pyac/tasks/pointer/visualize.py`:
- add a new helper that writes `pointer_reference.png`
- include compact context in that artifact: family label if available, start node, hops, target, prediction, pointer mapping/list
- remove full pointer-list text from the suptitles of:
  - `assembly_heatmap.png`
  - `assembly_bars_over_time.png`
  - `assembly_connectivity_graph.png`
  - `assembly_weight_matrix.png`
- keep those main figures otherwise behaviorally unchanged
- update `render_trace_visualizations(...)` so it returns all five artifact paths

**Step 4: Run test to verify it passes**

Rerun the direct visualization and suite trace-hook harnesses until they pass.

**Step 5: Commit**

Defer commit until real family smoke verification is complete.

### Task 4: Smoke-verify canonical seen/unseen AC family outputs

**Files:**
- Verify: `outputs/experiments/seen-ac/trace_plots/`
- Verify: `outputs/experiments/unseen-ac/trace_plots/`

**Step 1: Write the failing test**

Use a post-run assertion script that checks both family trace directories contain exactly:

```python
{
    "assembly_heatmap.png",
    "assembly_bars_over_time.png",
    "assembly_connectivity_graph.png",
    "assembly_weight_matrix.png",
    "pointer_reference.png",
}
```

**Step 2: Run test to verify it fails**

If needed, run the assertion before regeneration to confirm stale outputs do not yet match.

**Step 3: Write minimal implementation**

Run:
- `venv/bin/python run_experiment_suite.py --config experiments/seen_ac.yaml`
- `venv/bin/python run_experiment_suite.py --config experiments/unseen_ac.yaml`

**Step 4: Run test to verify it passes**

Run the assertion script and confirm both family trace directories contain the five expected files.

**Step 5: Commit**

```bash
git add experiments/seen_ac.yaml experiments/unseen_ac.yaml pyac/src/pyac/tasks/pointer/visualize.py tests/test_pointer_visualize.py tests/test_experiment_suite_config.py tests/test_suite_trace_plots.py
git commit -m "Clean up AC family trace plots"
```
