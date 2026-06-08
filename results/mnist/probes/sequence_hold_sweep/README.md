# MNIST Sequence Hold Sweep

## Purpose

This is an exploratory visualization, not a primary thesis benchmark. It asks whether holding each digit longer helps a continuous MNIST AC sequence switch out of sticky recurrent attractors.

## Setup

- Config: `experiments/mnist_ac_sequence_hold_sweep.yaml`
- Output: `results/mnist/probes/sequence_hold_sweep/`
- Model: tuned MNIST AC (`n=2000`, `k=200`, `p=0.1`, `beta=1.0`, `raw_k=200`, `presentation_rounds=100`)
- Seed: `42`
- Sequence: `0,1,2,3,4,5,6,7,8,9`
- Hold lengths: `3`, `10`, `30`, `100` stimulus steps per digit
- State reset: once before each full sequence only; no reset between digits
- Plasticity: off during sequence evaluation
- Training: one trained network reused for all hold lengths

## Generated Files

- `raw_results.csv`: one row per sequence step, including overlaps, prediction, margin, and trajectory so far
- `summary.csv`: one row per sequence step and hold length (`k_test` is the sequence step)
- `config_snapshot.yaml`: exact run configuration
- `plots/mnist_sequence_hold_sweep_final_accuracy.png`: final-step accuracy after each digit phase by hold length
- `plots/mnist_sequence_hold_sweep_switch_latency.png`: first correct step within each digit phase, if any
- `plots/mnist_sequence_hold_sweep_predictions.png`: prediction timelines aligned by presented digit and within-digit progress

## Result Summary

| Hold steps | Overall step accuracy | Final-step accuracy | Correct final digits | Dominant final attractor pattern |
|---:|---:|---:|---|---|
| 3 | 20.0% | 20.0% | 0, 4 | sticks on 0 early, then mostly 4 |
| 10 | 20.0% | 20.0% | 0, 5 | sticks on 0 early, then mostly 5 |
| 30 | 19.7% | 20.0% | 0, 4 | sticks on 0 early, then mostly 4 |
| 100 | 19.9% | 20.0% | 0, 4 | sticks on 0 early, then mostly 4 |

Final predictions by digit:

| Hold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 0 | 0 | 5 | 4 | 4 | 8 | 4 | 4 | 4 |
| 10 | 0 | 0 | 0 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| 30 | 0 | 0 | 0 | 5 | 4 | 4 | 4 | 4 | 4 | 4 |
| 100 | 0 | 0 | 0 | 5 | 4 | 4 | 4 | 4 | 4 | 4 |

Switch latency, measured as the first correct step within a digit phase:

| Hold | digit 0 | digit 1 | digit 2 | digit 3 | digit 4 | digit 5 | digit 6 | digit 7 | digit 8 | digit 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | never | never | never | 0 | never | never | never | never | never |
| 10 | 0 | never | never | never | never | 0 | never | never | never | never |
| 30 | 0 | never | never | never | 1 | never | never | never | never | never |
| 100 | 0 | never | never | never | 1 | never | never | never | never | never |

## Interpretation

Holding the new digit longer does **not** make the continuous sequence track the presented digits. Accuracy stays near 20% for all hold lengths. Longer holds mainly strengthen whichever attractor captures the recurrent state: 0 dominates the early sequence, then either 4 or 5 dominates the later sequence.

This reinforces the interpretation from the 3-step sequence probe: the trained MNIST AC model is a reasonable reset-per-image classifier, but it is a poor continuous switching system. Without a reset, refractory mechanism, or explicit transition control, recurrent state persistence can overpower the new stimulus.
