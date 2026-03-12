# Pointer AC Seen-Lists Design

## Goal

Build a proper Assembly Calculus pointer-chasing experiment for seen lists using the `pyac` task/protocol style, so the thesis can compare a fixed-computation MLP baseline against an iterative AC system that improves with internal time.

## Problem Framing

`eval_mlp_baseline.py` already shows the desired experimental shape: train on a bank of seen pointer lists, then evaluate accuracy by hop count. The current AC wrapper in `models/ac.py` is not suitable for the thesis story because it directly writes the pointer map into recurrent weights instead of learning through repeated plastic exposures.

For the seen-lists setting, we want a protocol-trained AC model that:

- learns stable node assemblies and list-specific transition pathways through repeated presentations
- freezes plasticity during evaluation
- exposes internal-time behavior by varying the number of recurrent rollout steps
- writes results to CSV so they can be compared directly with the MLP seen-list baseline

## Architecture

Create a new task package under `pyac/src/pyac/tasks/pointer/` and keep the experiment runner separate at the repository root.

The task package will provide three layers:

1. `data.py` generates unique full-cycle lists and pointer-chasing samples.
2. `protocol.py` builds the network, trains node assemblies plus seen-list transitions with plasticity on, and evaluates pointer rollouts with plasticity off.
3. `metrics.py` runs accuracy-by-hop sweeps for the seen-list bank and returns plain Python records suitable for CSV export.

The top-level runner will mirror `eval_mlp_baseline.py`: parse arguments, build the seen-list bank, train AC, evaluate over hop counts, save `outputs_ac_seen/ac_seen_results.csv`, and print enough diagnostics to understand whether the expected story is emerging.

## Network Design

Use a task-specific `NetworkSpec` built from `pyac` primitives rather than `nn.Module` wrappers.

- `input` area: presents the currently active node as an external stimulus.
- `state` area: recurrent area in which stable node assemblies form and pointer-following unfolds over time.

The first version keeps readout simple: after the rollout ends, compare the final `state` assembly with the stored node prototypes using overlap / intersection size and choose the best-matching node.

## Training Protocol

Training has two phases.

### Phase 1: Learn node assemblies

Train one stable state assembly per node by repeatedly stimulating each node in the input area and letting the recurrent state area settle under plasticity. Use a bias / reuse-avoidance mechanism similar to the MNIST protocol so node assemblies become reasonably distinct.

### Phase 2: Learn seen-list transitions

For each seen list and for each source node in that list:

- present the source node stimulus
- settle the source node state assembly
- present or bias the destination node during the plastic update window
- strengthen the recurrent pathway that maps source-node state activity to destination-node state activity

Repeat this for multiple rounds over the fixed bank of seen lists so transitions become stable. This keeps learning proper: the pointer map is acquired through repeated co-activation rather than direct weight assignment.

## Evaluation Protocol

Evaluation runs with `plasticity_on=False`.

For each sample `(list, start, k)` from the seen-list bank:

- reset activations
- stimulate the start node once
- allow the state area to settle onto the start-node assembly
- run recurrent internal steps to follow transitions for `k` hops
- decode the final active state by matching against stored node assemblies

Seen-list accuracy is measured separately for each hop count. This lets us test the intended thesis claim: increasing internal recurrent steps should let AC solve longer pointer chains, unlike the MLP which collapses outside the training hop range.

## Error Handling and Constraints

- Validate that each pointer list is a permutation and optionally a single full cycle.
- Keep randomness deterministic with `make_rng` / `spawn_rngs`.
- Reset activations between examples and normalize relevant weights between training phases.
- Store learned node assemblies and any task metadata on the network object or in a task state record returned by the protocol.

## Testing Strategy

Follow TDD at the task-package level.

- unit-ish task tests for list generation / sample correctness
- integration tests for building the pointer network
- integration tests that training improves seen-list accuracy on a tiny deterministic problem
- determinism tests with fixed seeds
- metric tests that the hop sweep returns the expected schema and monotonic internal-step accounting

## Expected Story

For seen lists, the target qualitative result is:

- MLP memorizes only the trained hop range and collapses beyond it.
- Protocol-trained AC keeps solving longer hops by spending more internal recurrent steps.
- This supports the thesis framing that AC trades time for computation.

The first milestone is therefore not unseen-list generalization yet. It is a clean, reproducible seen-list result that clearly differs from the MLP baseline.
