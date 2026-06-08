# MNIST Assembly Calculus Results

## Overview

These experiments test Assembly Calculus (AC) on MNIST digit classification using a **legitimate AC model** with:

- **Sparse random fibres**: `X → Y` (sensory-to-coding) with probability `p`, plus recurrent `Y → Y`
- **k-cap dynamics**: top-k inhibition selects the `k` most active coding neurons each step
- **Hebbian multiplicative plasticity**: weight updates during training only
- **Frozen evaluation weights**: no learning during test
- **Recurrent internal time `t`**: variable number of recurrent steps during evaluation
- **Homeostatic class-separation bias**: persistent negative bias on previously active neurons (matched to Papadimitriou et al. reference notebook)
- **Real MNIST IDX data** from `data/mnist/`, raw-pixel encoding (top-200 activated input neurons per image)
- **Overlap-based decoding**: classify by largest overlap between test-image assembly and class prototypes (no supervised softmax)

**Key model parameters** (shared across all experiments unless varied in probes):

| Parameter | Value | Description |
|---|---|---|
| `n` | 2000 | Coding area size |
| `k` | 200 | Cap size (top-k inhibition) |
| `p` | 0.1 | Fibre connection probability |
| `beta` | 1.0 | Hebbian plasticity rate |
| `raw_k` | 200 | Input neurons activated per image |
| stimulus | held | Input present during all evaluation steps |

---

## Directory Structure

```
results/mnist/
├── README.md              ← this file
├── parity/                ← parity run (matches reference notebook settings)
├── tuned/                 ← best tuned model (5 seeds)
└── probes/
    ├── scale/             ← n,k scaling probe
    ├── rounds/            ← presentation rounds probe
    ├── sparsity/          ← fibre sparsity p probe
    ├── input_cap/         ← input encoding cap raw_k probe
    ├── t100_held/         ← long-time held-stimulus probe (t=0..100)
    ├── t100_transient/    ← long-time transient/cue-only probe (t=0..100)
    ├── t100_compare/      ← held vs transient comparison plots
    ├── sequence_0_to_9/   ← exploratory continuous digit-switching probe
    └── sequence_hold_sweep/ ← continuous sequence with longer digit holds
```

Each directory contains:
- `summary.csv` — aggregate accuracy per `t`
- `raw_results.csv` — per-image overlaps, margins, correctness
- `config_snapshot.yaml` — exact config used for that run
- `plots/` — auto-generated diagnostic plots (see Plot Reference below)

---

## Experiments

### 1. Parity (`parity/`)

**Purpose:** Run AC at the exact settings from the Papadimitriou et al. reference notebook to establish a legitimate baseline.

**Settings:** `n=2000, k=200, p=0.1, beta=1.0, presentation_rounds=5, class_organized, held stimulus, 5 seeds [42-46], 500 test images/seed`

**Results:**

| t | Accuracy |
|---:|---:|
| 0 | 60.76% |
| 1 | 60.84% |
| 2 | 60.84% |
| 4 | 60.92% |
| 8 | 60.88% |
| 10 | 60.84% |

**Finding:** Matches the reference notebook's overlap-based readout (~59-61%). This is the "apples-to-apples" baseline — the notebook's reported ~92-96% result comes from a trained supervised softmax, not pure overlap decoding.

---

### 2. Tuned (`tuned/`)

**Purpose:** Best MNIST result found after probing hyperparameters. Uses 100 presentation rounds (20x more training than parity).

**Settings:** `n=2000, k=200, p=0.1, beta=1.0, presentation_rounds=100, class_organized, held stimulus, 5 seeds [42-46], 500 test images/seed`

**Results:**

| t | Accuracy |
|---:|---:|
| 0 | 68.24% |
| 1 | 69.00% |
| 2 | 69.20% |
| 4 | 69.32% |
| 8 | 69.32% |
| 10 | 69.32% |

**Per-class accuracy at best t=4:**

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Acc | 90.0% | 96.7% | 77.5% | 52.4% | 71.3% | 67.6% | 34.4% | 82.0% | 54.5% | 51.9% |

**Finding:** More training rounds improve accuracy from ~61% to ~69%. But class-specific failures are severe: digit 6 at only 34.4%. Visual similarity groups (3/5/8, 4/9/7) drive most errors. This gap is irreducible with purely overlap-based decoding on sparse raw-pixel assemblies.

---

## Parameter Probes

All probes use **1 seed, 500 test images** for fast diagnostic sweeps.

### 3. Scale (`probes/scale/`)

**Purpose:** Does larger n,k improve performance? (Theory predicts size helps.)

**Models:** `n=2000,k=200` (parity), `n=5000,k=500` (proportional scale), `n=5000,k=200` (capacity-only)

**Finding:** Naive scaling did not help. `n=5000,k=500` stayed at ~60.4% (same as parity). `n=5000,k=200` collapsed to ~8.4% chance-level. Conclusion: scaling is not a free win; density (`k/n`) and capacity must be retuned together.

### 4. Rounds (`probes/rounds/`)

**Purpose:** How much does training data repetition help?

**Models:** `presentation_rounds=5, 20, 100`

**Finding:** More rounds help monotonically. Rounds=5: ~62%, Rounds=20: ~63.4%, Rounds=100: ~69.6%. Training sees 50k images/class in all cases; rounds controls how many times each image is re-presented.

### 5. Sparsity (`probes/sparsity/`)

**Purpose:** What fibre sparsity `p` is best at 100 rounds?

**Models:** `p=0.05, 0.10, 0.20`

**Finding:** `p=0.10` is best (peak 69.8%). `p=0.05` is weaker (~66.4%). `p=0.20` is slightly worse (~69.0%). Too sparse = not enough connectivity; too dense = too much crosstalk.

### 6. Input Cap (`probes/input_cap/`)

**Purpose:** How many raw pixels (`raw_k`) should feed into the coding area?

**Models:** `raw_k=100, 200, 300`

**Finding:** `raw_k=200` is best (peak 69.8%). `raw_k=100` is weakest (~57.4%). `raw_k=300` is slightly worse (~68.2%). Too few pixels = not enough information; too many = noise.

---

## Long-Time Probes (`probes/t100_*`)

**Purpose:** Test theory predictions about internal time. Theory says: (a) stimulus time allows settling, (b) non-stimulus time may cause drift/collapse. These probes run `t ∈ {0,1,2,4,8,10,20,40,60,80,100}`.

**Settings:** `n=2000, k=200, p=0.1, beta=1.0, presentation_rounds=100, 1 seed, 200 test images`

### 7. Held (`probes/t100_held/`)

Stimulus stays present during all recurrent steps.

| t | Accuracy |
|---:|---:|
| 0 | 69.0% |
| 1 | 69.0% |
| 2 | 70.5% |
| 4 | 70.5% |
| 10 | 70.5% |
| 40 | 70.5% |
| 100 | 70.5% |

**Finding:** No collapse. Accuracy improves by t=2 and then saturates completely. Held-stimulus AC reaches a stable attractor.

### 8. Transient (`probes/t100_transient/`)

Stimulus applied at step 0 only; removed for all subsequent recurrent steps.

| t | Accuracy |
|---:|---:|
| 0 | 69.0% |
| 1 | 56.5% |
| 2 | 36.5% |
| 4 | 14.5% |
| 10 | 10.0% |
| 40 | 5.5% |
| 100 | 7.5% |

**Finding:** Collapse. Without the image held, recurrent dynamics drift rapidly. By t=20, accuracy is near chance. This directly supports the theory's distinction between stimulus-time and non-stimulus-time dynamics.

### 9. Comparison (`probes/t100_compare/`)

Combined held-vs-transient plots:
- `mnist_accuracy_held_vs_transient_t100.png`
- `mnist_margin_held_vs_transient_t100.png`

---

## Exploratory Sequence Probe (`probes/sequence_0_to_9/`)

**Purpose:** Visualize AC state persistence when digits are presented continuously without resetting the coding area. This is for intuition, not a primary thesis benchmark.

**Settings:** tuned MNIST AC model, 1 seed, one representative test image per digit, sequence `0→1→2→3→4→5→6→7→8→9`, 3 held stimulus steps per digit, plasticity off, no reset between digit phases.

**Result:** The model locks onto digit 0 for the first four phases (`0,1,2,3`), then switches to digit 4 and remains mostly stuck on 4 for later phases. Overall step accuracy is 20% (`6/30`), but the important observation is hysteresis: the recurrent AC state carries history and can resist new stimuli unless reset.

**Sequence-specific plots:**
- `mnist_sequence_predictions_0_to_9.png`
- `mnist_sequence_overlaps_0_to_9.png`
- `mnist_sequence_margin_0_to_9.png`

### 11. Sequence Hold Sweep (`probes/sequence_hold_sweep/`)

**Purpose:** Test whether longer stimulus holds let the continuous sequence recover from sticky attractors.

**Settings:** same tuned MNIST AC model and `0→9` sequence, but compare `steps_per_digit ∈ {3,10,30,100}`. The network is trained once and reused for all hold lengths. Evaluation resets once before each full sequence, not between digits.

**Result:** Longer holds do not solve switching. Overall step accuracy stays near 20% for all hold lengths, and final-step accuracy is exactly 20% in each case. The state initially locks onto digit 0, then later falls into either a 4 or 5 attractor depending on hold length.

| Hold steps | Overall step accuracy | Final-step accuracy | Correct final digits |
|---:|---:|---:|---|
| 3 | 20.0% | 20.0% | 0, 4 |
| 10 | 20.0% | 20.0% | 0, 5 |
| 30 | 19.7% | 20.0% | 0, 4 |
| 100 | 19.9% | 20.0% | 0, 4 |

**Finding:** This is stronger evidence that the continuous sequence failure is not just too-short exposure. The model is a reset-per-image classifier; without reset/transition control, recurrent hysteresis dominates new sensory input.

**Hold-sweep plots:**
- `mnist_sequence_hold_sweep_final_accuracy.png`
- `mnist_sequence_hold_sweep_switch_latency.png`
- `mnist_sequence_hold_sweep_predictions.png` (aligned by presented digit and within-digit progress)

---

## Plot Reference

Each experiment directory generates the following plots:

| Plot | What It Shows |
|---|---|
| `mnist_accuracy_vs_t.png` | Mean accuracy across internal time t, with SE bands |
| `mnist_per_class_accuracy_vs_t.png` | Accuracy broken out per digit class (0-9) |
| `mnist_margin_vs_t.png` | Mean correct-overlap margin and its 10th percentile over t |
| `mnist_confusion_early.png` | Confusion matrix at smallest t |
| `mnist_confusion_best.png` | Confusion matrix at t with best accuracy |
| `mnist_confusion_late.png` | Confusion matrix at largest t |
| `mnist_pair_drift_3_5.png` | Overlap trajectories for digit 3 vs 5 test images |
| `mnist_pair_drift_4_9.png` | Overlap trajectories for digit 4 vs 9 test images |
| `mnist_pair_drift_7_9.png` | Overlap trajectories for digit 7 vs 9 test images |
| `mnist_sequence_predictions_0_to_9.png` | Presented-vs-predicted digit timeline for the exploratory 0→9 sequence probe |
| `mnist_sequence_overlaps_0_to_9.png` | Class-overlap trajectories during continuous digit switching |
| `mnist_sequence_margin_0_to_9.png` | Margin for the currently presented digit during continuous digit switching |
| `mnist_sequence_hold_sweep_final_accuracy.png` | Final-step sequence accuracy across digit hold lengths |
| `mnist_sequence_hold_sweep_switch_latency.png` | First correct step after each digit switch for each hold length |
| `mnist_sequence_hold_sweep_predictions.png` | Prediction timelines for hold lengths 3, 10, 30, and 100, aligned by presented digit and within-digit progress |

**Margin** is defined as `m_y(t) = o_y(t) - max_{z≠y} o_z(t)` where `o_y` is the overlap between the test assembly and class-y prototype.

---

## Theory Assessment

### Supported

1. **Time as settling**: Accuracy and margins improve in early internal steps (t=0→2/4), then saturate.
2. **Stimulus vs non-stimulus time**: Held-stimulus AC is stable; transient/cue-only AC collapses. The theory's distinction between these regimes is clearly visible.
3. **Margin mediation**: Improvements in mean margin correlate with accuracy improvements early on.
4. **Readout matters**: Overlap decoding is much harder than supervised softmax (~69% vs ~92%). The theory should not conflate representational quality with classification accuracy.

### Not Supported (or requires qualification)

1. **Size as improvement**: Naive scaling of n,k did not improve performance in our probes. If the theory predicts that larger networks help, it must specify the scaling law (density, input fan-in, etc.).
2. **Clean class assemblies**: MNIST digit classes are not compact under raw sparse pixel features. Digit 6 accuracy is 34.4% while digit 1 is 96.7%. The theory's "one assembly per class" idealization is too strong.
3. **No homeostatic mechanisms**: The working model requires a notebook-matched homeostatic inhibitory bias. Pure Hebbian AC alone was insufficient (~16% before these fixes).

### Thesis-Safe Conclusion

> The MNIST experiment provides qualitative support for the AC theory's predictions about internal time: accuracy and margins improve briefly with recurrent steps (settling), then saturate under held stimulus, while non-stimulus time causes rapid drift. However, the experiment also exposes important limits: overlap-based decoding saturates at ~69%, class-specific failures are severe, naive scaling is insufficient, and homeostatic mechanisms are required. MNIST supports AC as a useful representational substrate more strongly than it supports AC as a standalone high-accuracy classifier.

---

## Reproduction

All experiments use the same suite runner:

```bash
PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/<config>.yaml
```

Configs live in `experiments/mnist_ac_*.yaml`. Requires real MNIST IDX files in `data/mnist/`.
