# Proper-Unseen Pointer Chasing (Theory §9)

## Purpose

Primary temporal experiment for the thesis. Tests whether the AC episodic-memory/controller model can execute `s_L = M^L(s_0)` on unseen pointer tables, sweeping internal time `t`, depth `L`, and per-transition cost `c`.

## Setup

- Config: `experiments/pointer_ac_theory.yaml`
- Variant: `pointer_variant="proper_unseen"`
- N=6, depth sweep L=1..5, time sweep t=[0,1,2,3,4,6,8,10], c sweep [1,2,4,8]
- Seeds: [1,2,3], 12 train lists, 5 test lists, 16 samples/list/depth/time/c
- Each unseen table is written into episodic memory with plasticity on; the subsequent rollout/read phase is frozen
- All per-instance rows have unique `instance_id` and record `episodic_write_plasticity_on`/`rollout_plasticity_on`

## Files

- `raw_results.csv` — 38400 per-instance Theory rows
- `summary.csv` — aggregate rows preserving L, t, and c
- `config_snapshot.yaml` — exact config used for this run
- `plots/` — pointer-specific figures

## Main Result

### c=1

| L | t=0 | t=1 | t=2 | t=3 | t=4 | t=6 | t=8 | t=10 |
|---|-----|-----|-----|-----|-----|-----|-----|------|
| 1 | 0.000 | 0.196 | 0.142 | 0.158 | 0.121 | 0.150 | 0.200 | 0.175 |
| 2 | 0.000 | 0.121 | 0.150 | 0.133 | 0.150 | 0.154 | 0.142 | 0.142 |
| 3 | 0.000 | 0.125 | 0.129 | 0.129 | 0.142 | 0.142 | 0.146 | 0.133 |
| 4 | 0.000 | 0.162 | 0.158 | 0.142 | 0.104 | 0.158 | 0.167 | 0.121 |
| 5 | 0.000 | 0.150 | 0.129 | 0.162 | 0.154 | 0.171 | 0.158 | 0.179 |

### c=2

| L | t=0..1 | t=2 | t=3 | t=4 | t=6 | t=8 | t=10 |
|---|--------|-----|-----|-----|-----|-----|------|
| 1 | 0.000 | 0.100 | 0.179 | 0.154 | 0.158 | 0.158 | 0.154 |
| 2 | 0.000 | 0.146 | 0.188 | 0.108 | 0.183 | 0.125 | 0.092 |
| 3 | 0.000 | 0.125 | 0.125 | 0.142 | 0.125 | 0.150 | 0.150 |
| 4 | 0.000 | 0.117 | 0.117 | 0.146 | 0.188 | 0.167 | 0.100 |
| 5 | 0.000 | 0.108 | 0.129 | 0.125 | 0.108 | 0.108 | 0.154 |

### c=4

| L | t=0..3 | t=4 | t=6 | t=8 | t=10 |
|---|--------|-----|-----|-----|------|
| 1 | 0.000 | 0.158 | 0.121 | 0.183 | 0.138 |
| 2 | 0.000 | 0.138 | 0.100 | 0.129 | 0.133 |
| 3 | 0.000 | 0.112 | 0.088 | 0.125 | 0.150 |
| 4 | 0.000 | 0.167 | 0.154 | 0.125 | 0.192 |
| 5 | 0.000 | 0.167 | 0.138 | 0.150 | 0.133 |

### c=8

| L | t=0..6 | t=8 | t=10 |
|---|--------|-----|------|
| 1 | 0.000 | 0.138 | 0.154 |
| 2 | 0.000 | 0.150 | 0.158 |
| 3 | 0.000 | 0.146 | 0.167 |
| 4 | 0.000 | 0.138 | 0.150 |
| 5 | 0.000 | 0.146 | 0.138 |

## Interpretation

The proper-unseen episodic-memory/controller model does not yet show the Theory-predicted `t >= cL` boundary or the reliability improvement with larger `c`. Accuracy remains near random (~10-20%) across all depths, time budgets, and per-transition costs.

This is a negative result: the episodic-memory unbinding-and-rerouting controller path does not generalize to unseen pointer tables with the current architecture. The seen-list sanity check (`results/pointer/seen_theory/`) demonstrates that the basic recurrent pointer mechanism works perfectly when tables are known.

Thesis use: separate the clean time-depth mechanism (seen) from the harder unseen-map generalization problem (proper-unseen). This is the generalization boundary, not a measurement failure.

## Reproduce

```bash
PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/pointer_ac_theory.yaml
```
