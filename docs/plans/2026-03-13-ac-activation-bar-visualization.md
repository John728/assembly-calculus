# AC Activation Bar Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new AC trace plot that shows per-neuron activation bars grouped by assembly over time while leaving the existing `assembly_heatmap.png` unchanged.

**Architecture:** Reuse the existing pointer trace payload and extend `pyac.tasks.pointer.visualize` with one new plot writer. Keep `render_trace_visualizations(...)` as the single entry point, add the new output file to its returned list, and lock the new artifact into the existing visualization test.

**Tech Stack:** Python, matplotlib, numpy, pathlib.

---

### Task 1: Add failing visualization test for the new artifact

**Files:**
- Modify: `tests/test_pointer_visualize.py`
- Modify: `pyac/src/pyac/tasks/pointer/visualize.py`

**Step 1: Write the failing test**

Update `test_pointer_visualizations_write_expected_files()` so the expected file set includes:

```python
assert {path.name for path in paths} == {
    "assembly_heatmap.png",
    "assembly_bars_over_time.png",
    "assembly_connectivity_graph.png",
    "assembly_weight_matrix.png",
}
```

**Step 2: Run test to verify it fails**

Run a direct test harness such as:

```bash
venv/bin/python - <<'PY'
from pathlib import Path
import tempfile
from tests.test_pointer_visualize import test_pointer_visualizations_write_expected_files

with tempfile.TemporaryDirectory() as tmp:
    test_pointer_visualizations_write_expected_files(Path(tmp))
print("pointer visualize test ok")
PY
```

Expected: FAIL because `assembly_bars_over_time.png` does not exist yet.

**Step 3: Write minimal implementation**

Do not implement the whole figure yet; only add enough structure to proceed to the next task once the failing expectation is confirmed.

**Step 4: Run test to verify it still fails for the right reason**

Run the same harness again.
Expected: FAIL specifically because the new file is missing.

**Step 5: Commit**

```bash
git add tests/test_pointer_visualize.py pyac/src/pyac/tasks/pointer/visualize.py
git commit -m "test: require AC activation bar trace plot"
```

### Task 2: Implement the new grouped activation-bar figure

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/visualize.py`
- Test: `tests/test_pointer_visualize.py`

**Step 1: Write the minimal plot function**

Add a new helper, for example:

```python
def _save_assembly_bars_over_time(trace: dict[str, Any], output_dir: Path) -> Path:
    ...
```

The function should:

- iterate over `trace["steps"]`
- build a strength vector per time step from `active_neurons`
- render one subplot row per time step
- color bars by assembly using the existing `_assembly_colors(...)`
- use `assembly_spans` to preserve grouping and boundary markers
- save to `output_dir / "assembly_bars_over_time.png"`

**Step 2: Wire it into the rendering pipeline**

Update:

```python
def render_trace_visualizations(...):
```

so it returns the new path in addition to the existing three outputs, while keeping `assembly_heatmap.png` behavior unchanged.

**Step 3: Run test to verify it passes**

Run:

```bash
venv/bin/python - <<'PY'
from pathlib import Path
import tempfile
from tests.test_pointer_visualize import test_pointer_visualizations_write_expected_files

with tempfile.TemporaryDirectory() as tmp:
    test_pointer_visualizations_write_expected_files(Path(tmp))
print("pointer visualize test ok")
PY
```

Expected: PASS.

**Step 4: Refactor lightly if needed**

Extract any tiny shared helpers only if they remove obvious duplication without changing output semantics.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/visualize.py tests/test_pointer_visualize.py
git commit -m "feat: add AC activation bar trace visualization"
```

### Task 3: Smoke-verify the new artifact in the real trace path

**Files:**
- Modify: none unless verification reveals a bug
- Check: `plot_ac_trace.py`
- Check: `run_experiment_suite.py`

**Step 1: Run a real trace-producing path**

Use an existing AC trace flow, for example a suite config that already writes `trace_plots/`, or the dedicated trace plotting path if present.

Run one of:

```bash
venv/bin/python run_experiment_suite.py --config experiments/ac_visualization.yaml
```

or the current project-standard trace command if that is the maintained path.

**Step 2: Verify the new file exists in output**

Check that the trace output directory now includes:

- `assembly_heatmap.png`
- `assembly_bars_over_time.png`
- `assembly_connectivity_graph.png`
- `assembly_weight_matrix.png`

**Step 3: Inspect for stale-regeneration issues**

If the file is missing because a caller bypasses `render_trace_visualizations(...)`, fix that caller minimally and re-run the smoke check.

**Step 4: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/visualize.py tests/test_pointer_visualize.py
git commit -m "chore: verify AC trace plot pack includes activation bars"
```

### Task 4: Final verification and summary

**Files:**
- Check: `pyac/src/pyac/tasks/pointer/visualize.py`
- Check: `tests/test_pointer_visualize.py`

**Step 1: Run final direct test verification**

Run the direct visualization test harness again.

Expected: PASS.

**Step 2: Run final smoke verification**

Run the chosen real trace generation command again and verify the new file exists.

Expected: the trace plot directory contains the full four-file set.

**Step 3: Summarize exact artifact contract**

Confirm that:

- `assembly_heatmap.png` is unchanged in name and role
- `assembly_bars_over_time.png` is new
- the rendering pipeline remains additive, not replacement-based

**Step 4: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/visualize.py tests/test_pointer_visualize.py docs/plans/2026-03-13-ac-activation-bar-visualization-design.md docs/plans/2026-03-13-ac-activation-bar-visualization.md
git commit -m "docs: plan AC activation bar visualization"
```
