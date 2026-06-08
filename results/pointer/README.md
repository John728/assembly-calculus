# Pointer Chasing Results

## Directory

- `results/pointer/theory/` — Proper-unseen pointer chasing experiment (thesis main), with c sweep
- `results/pointer/seen_theory/` — Seen-list sanity check (appendix)

## What Original Pointer Chasing Means Here

The pointer-chasing mechanism learns a recurrent transition so that one free AC update advances one symbolic pointer transition `s_l -> M(s_l)`. In Theory notation this is `c=1` (updates per transition), so the predicted boundary is `t >= cL`. The `c` sweep tests whether more updates per transition improve reliability.

This is different from a table-scan algorithm where one hop costs `N` updates.

## Proper-Unseen Pointer Chasing (`theory/`)

**Purpose:** Thesis main experiment per Theory §9. Tests execution `s_L = M^L(s_0)` on unseen tables.

**Setup:** Proper-unseen protocol, N=6, L=1..5, t=[0,1,2,3,4,6,8,10], c=[1,2,4,8], seeds [1,2,3]. Each unseen table is plastically written into episodic memory, then rollout/read is frozen.

**Result:** Accuracy near random (~10-20%) across all L, t, c. No `t >= cL` boundary, no c-reliability improvement visible.

**Interpretation:** The episodic-memory unbinding-and-rerouting controller path does not generalize to unseen tables. This is negative evidence: the seen-list mechanism works perfectly, but unseen-map generalization is the boundary.

See `results/pointer/theory/README.md` for full result tables and c-sweep breakdown.

## Seen Pointer Sanity Check (`seen_theory/`)

**Purpose:** Verify the original seen-list pointer mechanism satisfies the Theory time-depth prediction with per-instance measurements.

**Setup:** Original seen-list protocol, N=10, L=1..6, t=[0,1,2,3,4,6,8,10], c=1, seeds [42,43,44]. Weights frozen at evaluation.

**Result:** Exact diagonal boundary — accuracy is 0 before `t >= L` and 1 once the budget covers the depth.

| L | t=0 | t=1 | t=2 | t=3 | t=4 | t=6 | t=8 | t=10 |
|---|-----|-----|-----|-----|-----|-----|-----|------|
| 1 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 3 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 4 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |

## Schema

Each per-instance row records: `experiment="pointer_chasing"`, `pointer_variant`, `L`, `t`, `c`, `completed_hops`, `readout_step`, `N`, `seed`, `theta_id`, unique `instance_id`, `list_idx`, `sample_idx`, `start_node`, `target`, `prediction`, `correct`, `true_trajectory`, decoded `trajectory`, `path_accuracy`, `first_error_index`. Proper-unseen rows also record `episodic_write_plasticivity_on` and `rollout_plasticity_on`.

## Plot Reference

- `pointer_accuracy_heatmap_L_t.png` — final accuracy over (L,t)
- `pointer_accuracy_vs_t_by_L.png` — accuracy vs time, separated by depth
- `pointer_accuracy_vs_L_by_t.png` — accuracy vs depth, separated by time budget
- `pointer_path_accuracy_vs_L.png` — path accuracy vs depth, preserving time-budget curves
- `pointer_first_error_histogram.png` — distribution of first wrong transition

## Thesis-Safe Takeaway

The seen pointer task demonstrates that AC internal time implements iterative execution depth: after `t` recurrent updates the model completes up to `t` pointer transitions. The proper-unseen variant fails to generalize this mechanism to novel tables, identifying the generalization boundary for the episodic-memory controller path.

## Reproduce

```bash
PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/pointer_ac_theory.yaml
PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/pointer_ac_seen_theory.yaml
```
