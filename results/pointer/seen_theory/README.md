# Seen Pointer Theory Sanity Check

## Purpose

This directory contains the original seen-list pointer-chasing mechanism, recast into the Theory Map row schema. It verifies that the working pointer experiment exhibits the predicted internal-time boundary: one recurrent AC update advances one pointer transition, so success appears when `t >= L`.

## Setup

- Config: `experiments/pointer_ac_seen_theory.yaml`
- Output: `results/pointer/seen_theory/`
- Variant: `pointer_variant="seen"`
- N=10 nodes, full-cycle pointer tables
- Depth sweep: `L=1..6`
- Time sweep: `t=[0,1,2,3,4,6,8,10]`
- Transition cost: `c=1`
- Seeds: `[42,43,44]`
- Seen training lists: 10
- Samples per list/depth/time: 20
- Evaluation weights are frozen

## Files

- `raw_results.csv` — 28800 per-instance Theory rows
- `summary.csv` — aggregate rows preserving both `L` and `t`
- `config_snapshot.yaml` — exact config used for this run
- `plots/` — pointer-specific figures

## Main Result

Final-state accuracy follows the expected diagonal boundary exactly:

| L | t=0 | t=1 | t=2 | t=3 | t=4 | t=6 | t=8 | t=10 |
|---|-----|-----|-----|-----|-----|-----|-----|------|
| 1 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 3 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 4 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |

Path accuracy also rises linearly with completed hops and reaches 1 once the time budget covers the requested depth. The path-accuracy plot preserves separate time-budget curves rather than averaging all `t` values together.

## Interpretation

This is the positive mechanism sanity check for the thesis: in the original seen pointer protocol, internal AC time behaves like iterative execution depth. The result supports the Theory prediction `L_max(t) ~= floor(t/c)` with `c=1`.

This is not an unseen-table generalization claim. The tables are seen during training, so this result should be presented as evidence for temporal execution in AC, while `results/pointer/theory/` documents the harder proper-unseen failure case.
