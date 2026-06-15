# Pointer Cleanup + C-Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean repo to pointer/DFA/MNIST only, add Theory-required c sweep to pointer config, verify.

**Architecture:** Remove MLP/notebook runners and configs. Add `c` parameter to proper-unseen evaluator. Add `c_values` to pointer config. Preserve MNIST probes organized.

**Tech Stack:** Python, numpy, pyac, pytest

---

### Task 1: Remove MLP and notebook infrastructure

**Files:**
- Delete: `experiments/unseen_mlp_dev.yaml`
- Delete: `experiments/unseen_mlp_paper.yaml`
- Delete: `experiments/mnist_nb_full.yaml`
- Delete: `experiment_suite/runners/mlp_runner.py`
- Delete: `experiment_suite/runners/mnist_nb_runner.py`
- Modify: `run_experiment_suite.py`

**Step 1: Remove MLP/NB dispatch from run_experiment_suite.py**

Remove `mlp` and `mnist_nb` imports and handler branches. Keep `ac`, `mnist_ac`.

**Step 2: Delete the config files**

```bash
rm experiments/unseen_mlp_dev.yaml experiments/unseen_mlp_paper.yaml experiments/mnist_nb_full.yaml
```

**Step 3: Delete the runner files**

```bash
rm experiment_suite/runners/mlp_runner.py experiment_suite/runners/mnist_nb_runner.py
```

**Step 4: Run existing tests for regressions**

```bash
PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_experiment_suite_runners.py -q
```

---

### Task 2: Remove intermediate pointer probes and old dev configs

**Files:**
- Delete: `experiments/pointer_ac_theory_probe.yaml`
- Delete: `experiments/pointer_ac_theory_push.yaml`
- Delete: `experiments/mnist_ac_dev.yaml`
- Delete: `experiments/mnist_ac_legit.yaml`
- Delete: `experiments/unseen_ac_proper_dev.yaml`
- Delete: `results/pointer/theory_probe/` (entire directory)
- Delete: `results/pointer/theory_push/` (entire directory)
- Delete: `results/dfa/` (empty, placeholder for later)
- Delete: `results/binary_search/` (empty, placeholder for later)

**Step 1: Delete files**

```bash
rm experiments/pointer_ac_theory_probe.yaml experiments/pointer_ac_theory_push.yaml \
   experiments/mnist_ac_dev.yaml experiments/mnist_ac_legit.yaml \
   experiments/unseen_ac_proper_dev.yaml
rm -rf results/pointer/theory_probe results/pointer/theory_push results/dfa results/binary_search
```

---

### Task 3: Add c sweep to pointer evaluator

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Modify: `tests/test_pointer_theory_metrics.py`
- Modify: `experiment_suite/runners/ac_runner.py`
- Modify: `experiments/pointer_ac_theory.yaml`

**Step 1: TDD - add failing test for c=2 evaluation**

Add `test_evaluate_proper_unseen_per_instance_with_c_equals_2()`: c=2, L=2, t=4, decoded=[0,0,1,1,2,2] → trajectory=[0,1,2], path_accuracy=1.0, correct=True.

**Step 2: Red run - verify failure**

**Step 3: Modify evaluate_proper_unseen_per_instance** to accept `c` param instead of computing it internally. Remove `_compute_c`. Runner passes `c` from config.

**Step 4: Green run**

**Step 5: Add c_values to pointer_ac_theory.yaml**

```yaml
c_values: [1, 2, 4, 8]
```

**Step 6: Update runner to loop over c_values** in theory_pointer branch, passing `c` to evaluator and including it in row metadata.

**Step 7: Update pointer plots** - add `pointer_one_step_error_vs_c.png`

---

### Task 4: Regenerate pointer results with c sweep

**Step 1: Run pointer_ac_theory.yaml**

```bash
PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/pointer_ac_theory.yaml
```

**Step 2: Verify output structure and accuracy matrices**

---

### Task 5: Update docs

**Files:**
- Modify: `results/README.md`
- Modify: `results/pointer/README.md`
- Modify: `results/pointer/theory/README.md`

Update to reflect c sweep, removed configs, cleaned structure.

---

### Task 6: Final verification

```bash
PYTHONPATH=pyac/src .venv/bin/python -m pytest tests/test_pointer_*.py tests/test_mnist_*.py -q
```
