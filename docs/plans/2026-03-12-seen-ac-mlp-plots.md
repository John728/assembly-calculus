# Seen AC vs MLP Plots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build reproducible seen-list comparison plots for the protocol-trained AC experiment and the existing MLP baseline.

**Architecture:** Add a small plotting module that normalizes the AC and MLP CSV schemas into one comparison dataframe, then generate a focused set of thesis-ready figures from that dataframe. Keep the logic testable by separating dataframe construction from figure rendering and the CLI entrypoint.

**Tech Stack:** Python, pandas, matplotlib, seaborn, pathlib.

---

### Task 1: Add a failing schema-normalization test

**Files:**
- Create: `pyac/tests/integration/test_seen_comparison_plots.py`
- Create: `plot_seen_ac_vs_mlp.py`

**Step 1: Write the failing test**

Add a test that builds tiny AC and MLP CSV fixtures, imports `plot_seen_ac_vs_mlp`, calls a `build_seen_comparison_df(...)` helper, and asserts that the combined dataframe contains normalized columns for model family, model label, list type, hop count, accuracy, and internal steps.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH="pyac/src" venv/bin/python -m pytest pyac/tests/integration/test_seen_comparison_plots.py -q`
Expected: FAIL because the plotting module/helper does not exist yet.

**Step 3: Write minimal implementation**

Create `plot_seen_ac_vs_mlp.py` with a dataframe-builder helper that reads both CSVs, filters to seen lists, tags AC as family `AC`, tags MLP rows as family `MLP`, and standardizes the output columns.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH="pyac/src" venv/bin/python -m pytest pyac/tests/integration/test_seen_comparison_plots.py -q`
Expected: PASS.

### Task 2: Add rendering helpers and output test

**Files:**
- Modify: `plot_seen_ac_vs_mlp.py`
- Modify: `pyac/tests/integration/test_seen_comparison_plots.py`

**Step 1: Write the failing test**

Add a test that uses temporary CSV fixtures and a temporary output directory, calls a `generate_seen_comparison_plots(...)` helper, and asserts that the expected PNG files are created.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH="pyac/src" venv/bin/python -m pytest pyac/tests/integration/test_seen_comparison_plots.py -q`
Expected: FAIL because the rendering helper does not exist yet.

**Step 3: Write minimal implementation**

Implement plotting helpers and generate:
- `accuracy_vs_hop_seen.png`
- `accuracy_vs_hop_seen_best_mlp_vs_ac.png`
- `max_solved_hop_seen.png`
- `ac_time_vs_hop.png`
- `paper_panel_seen_comparison.png`

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH="pyac/src" venv/bin/python -m pytest pyac/tests/integration/test_seen_comparison_plots.py -q`
Expected: PASS.

### Task 3: Run the plotting script on real data

**Files:**
- Modify: `plot_seen_ac_vs_mlp.py`
- Output: `outputs_seen_comparison/*.png`

**Step 1: Run the script**

Run: `venv/bin/python plot_seen_ac_vs_mlp.py`
Expected: writes all expected PNGs under `outputs_seen_comparison/`.

**Step 2: Verify outputs exist and inspect the comparison trend**

Check that AC remains high across seen hops while MLP collapses outside the trained range.

**Step 3: Keep the labeling honest**

Ensure titles/legends say `Seen Lists` and label AC as a protocol-trained recurrent model with internal steps rather than implying unseen-list generalization.
