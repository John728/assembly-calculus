# Scalable Pointer Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a config-driven experiment platform that runs scalable seen/unseen pointer-chasing suites for both MLP and AC, and add AC trace visualizations that show firing assemblies, neurons, and learned recurrent structure over time.

**Architecture:** Add a unified suite runner that reads YAML experiment configs, expands them into standardized jobs, and writes one shared results schema for all model families. Build plotting and AC trace visualization as read-only consumers of saved outputs so large sweeps can be run once and re-plotted later without rerunning training.

**Tech Stack:** Python, YAML, pandas, matplotlib, seaborn, existing PyTorch MLP code, existing PyAC protocol code.

---

### Task 1: Add config loading and job expansion

**Files:**
- Create: `experiments/seen_small.yaml`
- Create: `experiments/ac_visualization.yaml`
- Create: `experiment_suite/config.py`
- Create: `experiment_suite/jobs.py`
- Create: `tests/test_experiment_suite_config.py`

**Step 1: Write the failing test**

Add tests that load a tiny YAML config and verify:
- suite metadata is parsed correctly
- model grids expand into concrete jobs
- seeds and list conditions are preserved in each job

Example assertions to include:

```python
def test_expand_jobs_produces_family_seed_combinations(tmp_path):
    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text(
        "suite_name: demo\n"
        "seeds: [1, 2]\n"
        "conditions:\n"
        "  - list_type: Seen\n"
        "models:\n"
        "  MLP:\n"
        "    - model_name: MLP-01\n"
        "  AC:\n"
        "    - model_name: AC-Seen\n",
        encoding="utf-8",
    )

    config = load_suite_config(cfg_path)
    jobs = expand_jobs(config)

    assert len(jobs) == 4
    assert {job.family for job in jobs} == {"MLP", "AC"}
    assert {job.seed for job in jobs} == {1, 2}
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_config.py -q`
Expected: FAIL because the suite config modules do not exist.

**Step 3: Write minimal implementation**

Implement:
- YAML config loader
- simple dataclasses for suite config and job records
- job expansion across `(family, config, seed, condition)`

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_config.py -q`
Expected: PASS.

### Task 2: Standardize result rows across MLP and AC

**Files:**
- Create: `experiment_suite/schema.py`
- Create: `tests/test_experiment_suite_schema.py`
- Modify: `eval_mlp_baseline.py`
- Modify: `eval_ac_seen.py`

**Step 1: Write the failing test**

Add tests that verify helper functions can convert one MLP result row and one AC result row into a common schema with required fields.

Example assertions to include:

```python
def test_standardize_rows_aligns_family_specific_outputs():
    mlp_row = {"List Type": "Seen", "Model": "MLP-01", "k": 4, "Accuracy": 1.0}
    ac_row = {"List Type": "Seen", "Model": "AC-Seen", "k": 4, "Accuracy": 1.0, "Internal Steps": 5}

    std_mlp = standardize_mlp_row(mlp_row, suite="demo", seed=1)
    std_ac = standardize_ac_row(ac_row, suite="demo", seed=1)

    assert std_mlp["family"] == "MLP"
    assert std_ac["family"] == "AC"
    assert std_mlp["k_test"] == 4
    assert std_ac["internal_steps"] == 5
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_schema.py -q`
Expected: FAIL because schema helpers do not exist.

**Step 3: Write minimal implementation**

Implement shared schema helpers and expose adapters so existing MLP and AC scripts can emit standardized rows.

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_schema.py -q`
Expected: PASS.

### Task 3: Add reusable suite executors for MLP and AC

**Files:**
- Create: `experiment_suite/runners/mlp_runner.py`
- Create: `experiment_suite/runners/ac_runner.py`
- Create: `experiment_suite/runners/__init__.py`
- Create: `tests/test_experiment_suite_runners.py`
- Modify: `eval_mlp_baseline.py`
- Modify: `eval_ac_seen.py`

**Step 1: Write the failing test**

Add tests that run one tiny MLP job and one tiny AC seen-list job and assert each runner returns non-empty standardized rows with valid `accuracy` and `family` fields.

Example assertions to include:

```python
def test_ac_runner_returns_standardized_rows():
    rows = run_ac_job(tiny_ac_job())
    assert rows
    assert all(row["family"] == "AC" for row in rows)
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in rows)
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_runners.py -q`
Expected: FAIL because suite runners do not exist.

**Step 3: Write minimal implementation**

Implement small runner wrappers that call refactored MLP/AC experiment helpers and return standardized rows.

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_runners.py -q`
Expected: PASS.

### Task 4: Build the unified suite runner CLI

**Files:**
- Create: `run_experiment_suite.py`
- Create: `experiment_suite/aggregate.py`
- Create: `tests/test_run_experiment_suite.py`

**Step 1: Write the failing test**

Add a tiny integration test that loads a demo config, runs the suite, and asserts that:
- `raw_results.csv` exists
- `summary.csv` exists
- `config_snapshot.yaml` exists

Example assertions to include:

```python
def test_run_experiment_suite_writes_outputs(tmp_path):
    out_dir = run_suite(config_path=tiny_suite_config(tmp_path))
    assert (out_dir / "raw_results.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_run_experiment_suite.py -q`
Expected: FAIL because the suite runner does not exist.

**Step 3: Write minimal implementation**

Implement:
- suite CLI entrypoint
- job dispatch loop
- raw-results writer
- summary writer
- config snapshot copy

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_run_experiment_suite.py -q`
Expected: PASS.

### Task 5: Move comparison plotting onto standardized suite outputs

**Files:**
- Modify: `plot_seen_ac_vs_mlp.py`
- Create: `experiment_suite/plots.py`
- Create: `tests/test_experiment_suite_plots.py`

**Step 1: Write the failing test**

Add a test that builds a tiny standardized `raw_results.csv` fixture and verifies the plotting helpers can generate seen-list comparison outputs from the standardized schema alone.

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_plots.py -q`
Expected: FAIL because suite plotting helpers do not exist.

**Step 3: Write minimal implementation**

Refactor plotting so it reads standardized result rows and writes:
- accuracy vs hop
- best-model comparison
- max solved hop
- runtime/step budget views

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_plots.py -q`
Expected: PASS.

### Task 6: Add AC rollout trace recording

**Files:**
- Create: `pyac/src/pyac/tasks/pointer/trace.py`
- Create: `tests/test_pointer_trace.py`
- Modify: `pyac/src/pyac/tasks/pointer/protocol.py`

**Step 1: Write the failing test**

Add tests that run one tiny AC rollout and verify the trace record contains:
- per-step active neurons
- per-step active assembly labels
- final decoded node
- aggregated assembly-to-assembly weights

Example assertions to include:

```python
def test_record_rollout_trace_has_expected_fields():
    trace = record_rollout_trace(...)
    assert "steps" in trace
    assert "assembly_weight_matrix" in trace
    assert trace["steps"]
    assert "active_neurons" in trace["steps"][0]
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_pointer_trace.py -q`
Expected: FAIL because trace recording does not exist.

**Step 3: Write minimal implementation**

Implement trace-recording helpers that use actual rollout activations and actual learned recurrent weights.

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_pointer_trace.py -q`
Expected: PASS.

### Task 7: Add AC neuron/assembly visualization plots

**Files:**
- Create: `plot_ac_trace.py`
- Create: `pyac/src/pyac/tasks/pointer/visualize.py`
- Create: `tests/test_pointer_visualize.py`

**Step 1: Write the failing test**

Add tests that feed a tiny synthetic trace into visualization helpers and assert that they write:
- `assembly_heatmap.png`
- `assembly_connectivity_graph.png`
- `assembly_weight_matrix.png`

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_pointer_visualize.py -q`
Expected: FAIL because visualization helpers do not exist.

**Step 3: Write minimal implementation**

Implement:
- neuron-by-time heatmap grouped by assembly
- assembly connectivity view using learned recurrent weight summaries
- assembly-to-assembly weight matrix heatmap

Make the neuron view visually explicit:
- separators between assemblies
- consistent assembly colors
- optional highlighted expected traversal path

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_pointer_visualize.py -q`
Expected: PASS.

### Task 8: Add suite-level plot hooks for AC trace outputs

**Files:**
- Modify: `run_experiment_suite.py`
- Modify: `experiment_suite/plots.py`
- Modify: `plot_ac_trace.py`
- Create: `tests/test_suite_trace_plots.py`

**Step 1: Write the failing test**

Add a test that enables a trace-plot block in a tiny config and verifies the suite run writes the requested AC trace PNGs for a selected evaluation example.

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_suite_trace_plots.py -q`
Expected: FAIL because suite trace hooks do not exist.

**Step 3: Write minimal implementation**

Implement optional trace-plot generation in the suite runner so configs can request one or more AC trace visualizations after evaluation.

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_suite_trace_plots.py -q`
Expected: PASS.

### Task 9: Create large-run starter configs

**Files:**
- Create: `experiments/seen_large.yaml`
- Create: `experiments/unseen_large.yaml`
- Create: `experiments/seen_ac_mlp_scaling.yaml`
- Modify: `docs/plans/2026-03-12-scalable-pointer-experiments-design.md`

**Step 1: Write the configs**

Create thesis-oriented starter configs with larger sweep grids for:
- seen-list scaling
- unseen-list evaluation
- AC/MLP direct family comparison

Include comments or a short doc note describing which config is intended for laptop-scale versus super-PC-scale use.

**Step 2: Run a smoke suite**

Run: `venv/bin/python run_experiment_suite.py --config experiments/seen_small.yaml`
Expected: suite completes and writes standardized outputs.

**Step 3: Verify config readability**

Check that each config clearly exposes the sweep knobs you care about: model size, list counts, seeds, hop ranges, AC time/assembly settings.

### Task 10: Final verification

**Files:**
- Test: `tests/test_experiment_suite_config.py`
- Test: `tests/test_experiment_suite_schema.py`
- Test: `tests/test_experiment_suite_runners.py`
- Test: `tests/test_run_experiment_suite.py`
- Test: `tests/test_experiment_suite_plots.py`
- Test: `tests/test_pointer_trace.py`
- Test: `tests/test_pointer_visualize.py`
- Test: `tests/test_suite_trace_plots.py`

**Step 1: Run the new test set**

Run: `venv/bin/python -m pytest tests/test_experiment_suite_config.py tests/test_experiment_suite_schema.py tests/test_experiment_suite_runners.py tests/test_run_experiment_suite.py tests/test_experiment_suite_plots.py tests/test_pointer_trace.py tests/test_pointer_visualize.py tests/test_suite_trace_plots.py -q`
Expected: PASS.

**Step 2: Run end-to-end suite smoke command**

Run: `venv/bin/python run_experiment_suite.py --config experiments/seen_small.yaml`
Expected: standardized CSV outputs plus suite plots.

**Step 3: Run AC trace plot command**

Run: `venv/bin/python plot_ac_trace.py --suite outputs/seen_small`
Expected: assembly heatmap, connectivity graph, and weight matrix outputs.

**Step 4: Document the first recommended super-PC commands**

Add a short usage section describing the first three large commands to run when the bigger machine is available.
