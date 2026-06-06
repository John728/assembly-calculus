# MNIST-First PYAC Design

Date: 2026-06-06

## Purpose

PYAC should become a useful general Assembly Calculus library, while the thesis experiment suite remains a thin layer for running and measuring specific claims. Phase 1 starts with MNIST because it is the simplest static task and directly tests the theory's claims about internal time, margin settling, saturation, and drift.

The implementation must follow real Assembly Calculus / NEMO-style mechanisms. It must not replace the model with a generic classifier, shortcut computation, synthetic metrics, or fake results.

## Constraints

- Use real MNIST files as the phase-1 data source.
- Fail clearly if real MNIST data is missing; do not silently fall back to toy digits or generated data.
- Use sparse assemblies, k-cap dynamics, random sparse fibres, Hebbian plasticity, and frozen evaluation weights.
- Sweep internal time `t` while holding the same trained instantiated model fixed.
- Record raw overlaps and margins so the theory can be checked after the run.
- Treat MNIST as a static task: extra internal time can only change representation settling, completion, or drift. It must not be claimed as execution depth.
- Keep phase 1 minimal: do not refactor pointer chasing, DFA, binary search, or all plotting code unless required for MNIST.

## Architecture

### Library Layer: `pyac`

Keep reusable AC primitives in `pyac`:

- `pyac.core`: network specs, areas, fibres, k-cap, dynamics, plasticity, RNG.
- `pyac.measures`: overlap, normalized overlap vectors, margins, confusion helpers.
- `pyac.tasks.mnist` or `pyac.tasks.static_classification`: MNIST-specific AC model construction, data loading, training, evaluation, and decoding.

The MNIST task layer should expose reusable functions rather than embed CLI or experiment-suite concerns.

### Experiment Layer: `experiment_suite`

The experiment suite should orchestrate configured runs and write outputs. For MNIST phase 1, it should support:

- `experiment: mnist` or equivalent family-specific config.
- AC model hyperparameters: `n`, `k`, `p`, `beta`, recurrent on/off, stimulus handling, train/eval counts, seed, and `t` sweep.
- Standard output rows with theory-required fields.
- MNIST plots generated from saved raw outputs.

Pointer-specific fields such as `list_type`, `k_test`, and `num_train_lists` should not be forced into MNIST rows. They may remain for pointer jobs until later cleanup.

## MNIST AC Mechanism

The target model follows the Thesis A framing and the Dabagia/Papadimitriou style:

- Sensory area `X`: encodes an MNIST image as a sparse stimulus.
- Coding/readout area `Y`: recurrent area where digit assemblies form.
- Class assemblies: one learned/stabilized assembly per digit class, stored as `S_0 ... S_9` in the coding area.
- Readout: predict the class with maximum normalized overlap between the active coding cap and class assemblies.

Training should use Hebbian plasticity. Evaluation must set `plasticity_on=False` and reuse the same trained weights `W*` for every tested `t`.

The image encoder should be explicit and auditable. It may map active/intense pixels to fixed sparse sensory subsets, but it must not compute labels or class templates outside the AC dynamics.

## Data Flow

1. Load real MNIST train/test files.
2. Build one AC model instance from seed and hyperparameters.
3. Train class assemblies with labelled training examples and Hebbian plasticity.
4. Freeze weights.
5. For each test image and each internal time `t`:
   - reset activity;
   - present the image cue according to the configured stimulus protocol;
   - run exactly `t` internal AC updates;
   - decode active coding assembly;
   - record full overlap vector, margin, prediction, target, and correctness.
6. Aggregate only after raw per-instance outputs have been written.

## Required Result Schema

Each MNIST evaluation row should include at least:

- `experiment`: `mnist`
- `seed`
- `theta_id`
- `n`, `k`, `p`, `beta`
- `t`
- `instance_id`
- `target`
- `prediction`
- `correct`
- `overlaps`: vector of 10 class overlaps
- `correct_overlap`
- `strongest_wrong_overlap`
- `margin`
- `stimulus_mode`: for example `held` or `cue_only`
- `plasticity_on`: should be false during evaluation

Optional but useful fields:

- `trajectory`: decoded class by internal update
- `overlap_trajectory`: overlap vector by internal update
- `runtime_ms`
- `confusion_pair`

## Plots

Minimum phase-1 plots:

- Global accuracy vs `t`.
- Per-class accuracy vs `t`.
- Correct overlap and strongest wrong overlap vs `t`.
- Mean and lower-quantile margin vs `t`.
- Confusion matrices at early, best, and late `t`.
- Pair drift plots for likely confusions: `7/9`, `3/5`, `4/9`, and the Thesis A observed `5/3`, `9/7` pairs where present.

## What To Strip Or Avoid

- Do not add MLP as part of the AC MNIST proof.
- Do not use sklearn digits as thesis evidence.
- Do not synthesize overlaps from predictions.
- Do not keep applying plasticity during evaluation.
- Do not let `experiment_suite` assume every task is pointer chasing.
- Do not claim execution-depth results from MNIST.

## What To Keep

- Existing `pyac.core.Network`, `AreaSpec`, `FiberSpec`, `k_cap`, RNG, and Hebbian update behavior.
- Existing pointer code for later temporal experiments.
- Config-driven suite execution, but extended so MNIST can have its own schema and runner.
- Raw-result-first workflow: plots and summaries must derive from saved row data.

## Testing Requirements

Tests should verify:

- MNIST loading fails clearly when real data is unavailable.
- Evaluation calls run with plasticity disabled.
- A `t` sweep reuses one trained model instance rather than retraining per `t`.
- Overlap vectors have length 10 and margins equal `correct_overlap - strongest_wrong_overlap`.
- Raw outputs include the required theory fields.
- Plot generation works from raw MNIST rows.

## Phase-1 Acceptance Criteria

Phase 1 is complete when the repository can run one real MNIST AC experiment that produces:

- raw per-instance outputs for a `t` sweep;
- aggregated summary statistics by seed, class, and `t`;
- the minimum MNIST plot package;
- tests covering the data, evaluation, schema, and plotting contracts;
- no silent fallback to fake data or non-AC shortcuts.
