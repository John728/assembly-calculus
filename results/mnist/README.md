# MNIST Assembly Calculus Results

This directory contains MNIST result families used to study static-time behavior in Assembly Calculus.

The active thesis role is narrow:

- held stimuli test cue-supported settling;
- transient and retention protocols test autonomous basin stability after cue removal;
- normalization protocols test how training changes the frozen recurrent structure used at evaluation.

Older tuned, parity, broad beta-sweep, and broad capacity-sweep folders have been removed from the active results package.

## Active Result Families

| Directory | Role |
|---|---|
| `probes/t100_compare/` | Held-versus-transient comparison used for the static boundary claim. |
| `../mnist/retention_phase/` or Obsidian export | Cue-duration and post-removal retention sweep. |
| `canonical_time/` | Canonical held/transient time protocol when regenerated locally. |
| `canonical_normalization/` | Canonical normalization comparison when regenerated locally. |

Generated CSVs, YAML snapshots, and plots are ignored by Git unless explicitly copied into thesis notes.

## Appendix Probes

These probes are useful diagnostics but should not carry the main thesis argument:

| Directory | Role |
|---|---|
| `probes/input_cap/` | Input encoding strength sensitivity. |
| `probes/rounds/` | Training exposure sensitivity. |
| `probes/scale/` | Naive size scaling caution. |
| `probes/sparsity/` | Signal/crosstalk sensitivity. |
| `probes/sequence_0_to_9/` | Static classifier state used as a sequence-like process. |
| `probes/sequence_hold_sweep/` | Longer static holds do not make a reset-classifier into a sequence machine. |

## Removed From Active Use

The following older result families were removed because they were superseded, weakly framed, or not directly connected to the current thesis spine:

- `parity/`
- `tuned/`
- `probes/t100_held/`
- `probes/t100_transient/`
- `beta_sweep/`
- `capacity_sweep/`
- `capacity_sweep_low_beta/`
- `capacity_sweep_no_norm/`

Use the current thesis notes for interpretation rather than these historical exploratory folders.
