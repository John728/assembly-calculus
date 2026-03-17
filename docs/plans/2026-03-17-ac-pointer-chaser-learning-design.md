# AC Pointer-Chaser Learning Design

## Goal

Replace the current unseen Assembly Calculus pointer-chasing system with a clean implementation of the plan in `genetic_AC_Train_Plan.md`: a reusable controller that learns an internal recurrent routine for iterative pointer chasing on unseen list instances, with per-instance memory loading limited to writing `SRC -> DST` bindings for the current list.

## Success Bar

- Primary success criterion: unseen multi-hop rollout is clearly above chance and stable across several hops.
- Mechanistic criterion: traces must show the intended route `CUR -> SRC -> DST -> CUR` rather than a shortcut.
- Failure criterion: stop and report immediately if the local controller primitives cannot be made reliable, or if the system only works through hidden direct shortcuts or Python-side control.

## Current Problem

The current unseen AC path is not cleanly aligned with this claim.

- The existing `proper_unseen` code in `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py` still uses a hand-shaped control area and direct training patterns that risk acting like an engineered controller.
- The rollout contract already exposes the right external interface, but the internal mechanism is not yet convincingly a learned self-sustaining `CUR -> SRC -> DST -> CUR` loop.
- Keeping that code as the canonical unseen AC path would blur the scientific story and make it harder to tell whether the new plan really works.

## Replacement Strategy

Build a fresh unseen AC implementation with the new controller logic and staged curriculum, integrate it into the experiment suite, verify it against the mechanistic contract, and then delete the old unseen AC path once the new path is credible.

This is preferred over an in-place rewrite because it keeps the migration legible and prevents old assumptions from leaking into the new system.

## Architecture

### Areas

- `CUR`
  - current node register
  - one assembly per node identity
- `SRC`
  - memory key / address side
  - identity-aligned with `CUR`
- `DST`
  - memory value side
  - identity-aligned with `SRC`
- `LOOP`
  - small recurrent controller area
  - learns query triggering, write-back, and autonomous re-triggering
- optional `READOUT`
  - decoded final node state for experiment logging and metrics

### Identity Alignment

The same node index must correspond to aligned assemblies across `CUR`, `SRC`, and `DST`. This is a deliberate scaffold, not an unwanted shortcut. It isolates the scientific question to whether the controller can learn the iterative routine rather than forcing it to also invent a shared node vocabulary.

### Memory

For each episode, the list-specific memory is written by plastic co-activation so that `SRC[i]` retrieves `DST[next(i)]`. Test-time external code may:

1. build that fresh `SRC -> DST` memory for the unseen list,
2. cue `CUR[start]` once,
3. stop the simulation after a fixed horizon.

No Python-side per-hop query or phase logic is allowed.

### Internal Routine

The intended learned computation is:

`CUR[i] -> SRC[i] -> DST[next(i)] -> CUR[next(i)] -> ...`

`LOOP` is not allowed to become an explicit symbolic scheduler. Its role is to provide the recurrent scaffold that learns:

- query transformation: `CUR[i] -> SRC[i]`
- write-back transformation: `DST[j] -> CUR[j]`
- loop closure: newly written `CUR[j]` causes the same routine to run again

## Training Design

Training follows the staged curriculum from the user plan.

### Stage 0: Assembly Formation

Verify stable assemblies in `CUR`, `SRC`, and `DST` for every node identity. Partial cues should pattern-complete reliably enough to support later stages.

### Stage 1: Instance Memory

For each training list instance, write episodic `SRC[i] -> DST[j]` bindings with plasticity and verify that one-hop lookup works before training the controller.

### Stage 2: Query Primitive

Teacher-force `CUR[i]` followed by `SRC[i]` during training only until `CUR[i]` alone reliably produces `SRC[i]`.

### Stage 3: Write-Back Primitive

Teacher-force `DST[j]` followed by `CUR[j]` during training only until `DST[j]` alone reliably produces `CUR[j]`.

### Stage 4: One-Hop Composition

Compose live memory retrieval with the two learned primitives so the network naturally performs one-hop transitions on the current episodic memory.

### Stage 5: Multi-Hop Self-Sustained Recurrence

Train short trajectories with only the initial `CUR[start]` cue, gradually reducing teacher forcing until the network re-triggers the same routine on its own.

## Testing Contract

For an unseen list instance:

1. write the unseen list into episode memory,
2. do not retrain the controller,
3. cue only `CUR[start]`,
4. run for a fixed horizon,
5. decode `CUR` over time and optionally `SRC` and `DST`.

The central question is whether the controller generalizes unchanged to unseen mappings while only the episodic memory changes.

## Metrics

- primitive accuracy for `CUR -> SRC`
- primitive accuracy for `DST -> CUR`
- one-hop accuracy on held-out mappings
- sequence accuracy across several hops
- stability of single-dominant `CUR` state over time
- mechanism validity: evidence of `CUR -> SRC -> DST -> CUR`

## Failure Conditions

Stop and report the approach as not working if any of the following happen after reasonable tuning:

- `CUR -> SRC` does not generalize across node identities
- `DST -> CUR` does not generalize across node identities
- one-hop behavior only appears through direct `CUR -> CUR` shortcuts
- multi-hop recurrence cannot re-trigger without external procedural stepping
- mechanism traces show that memory is being bypassed

## Code Structure

- Replace the canonical unseen AC implementation under `pyac/src/pyac/tasks/pointer/`
- Update exports in `pyac/src/pyac/tasks/pointer/__init__.py`
- Update experiment integration in `experiment_suite/runners/ac_runner.py`
- Replace protocol-focused tests in `tests/test_unseen_ac_protocol.py`
- Update unseen AC experiment YAMLs in `experiments/`

## Notes

- The old unseen AC implementation should be removed once the new protocol reaches the success bar.
- No git commit is included at design stage because the user has not asked for commits.
