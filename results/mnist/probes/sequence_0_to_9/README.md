# MNIST Sequence Probe: 0 to 9

## Purpose

This is an exploratory visualization, not a primary thesis result. It asks what a trained MNIST AC model does when the stimulus changes continuously without resetting the coding area.

## Setup

- Config: `experiments/mnist_ac_sequence_0_to_9.yaml`
- Output: `results/mnist/probes/sequence_0_to_9/`
- Model: tuned MNIST AC (`n=2000`, `k=200`, `p=0.1`, `beta=1.0`, `raw_k=200`, `presentation_rounds=100`)
- Seed: `42`
- Sequence: `0,1,2,3,4,5,6,7,8,9`
- Duration: 3 held stimulus steps per digit
- State reset: once before the full sequence only; no reset between digits
- Plasticity: off during sequence evaluation

## Generated Files

- `raw_results.csv`: one row per sequence step, including overlaps, prediction, margin, and trajectory so far
- `summary.csv`: one row per sequence step (`k_test` is the sequence step)
- `config_snapshot.yaml`: exact run configuration
- `plots/mnist_sequence_predictions_0_to_9.png`: presented digit vs predicted digit over time
- `plots/mnist_sequence_overlaps_0_to_9.png`: all class overlaps over the sequence
- `plots/mnist_sequence_margin_0_to_9.png`: margin for the currently presented digit

## Result Summary

| Presented digit | Predictions over 3 steps | Accuracy | Margins |
|---:|---|---:|---|
| 0 | 0, 0, 0 | 100% | 0.150, 0.510, 0.960 |
| 1 | 0, 0, 0 | 0% | -0.955, -0.935, -0.935 |
| 2 | 0, 0, 0 | 0% | -0.965, -0.975, -0.980 |
| 3 | 0, 0, 5 | 0% | -0.840, -0.330, -0.645 |
| 4 | 4, 4, 4 | 100% | 0.150, 0.725, 0.970 |
| 5 | 4, 4, 4 | 0% | -0.970, -0.965, -0.950 |
| 6 | 4, 4, 8 | 0% | -0.805, -0.395, -0.605 |
| 7 | 8, 4, 4 | 0% | -0.360, -0.650, -0.855 |
| 8 | 4, 4, 4 | 0% | -0.905, -0.930, -0.950 |
| 9 | 4, 4, 4 | 0% | -1.000, -1.000, -1.000 |

Overall step accuracy is 20% (`6/30` correct steps), but this number is not the main point of the probe.

## Interpretation

The model does not smoothly track every new digit. It locks strongly onto digit 0, remains stuck on 0 through digits 1-3, switches to digit 4, then remains mostly stuck on 4 for the rest of the sequence. This is consistent with attractor hysteresis: without a reset, the recurrent coding state carries history and can resist new sensory evidence.

Use this as a visual demonstration of AC state persistence and switching failure, not as a classification benchmark.
