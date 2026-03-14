# AC Trace Plot Cleanup Design

## Goal

Clean up the canonical AC trace diagnostics so they are easier to read in the paper-facing `outputs/experiments/...` layout, while keeping the new time-neuron activation plot and preserving the current family-local trace workflow.

## Approved Scope

- Change the canonical family trace example from `hops: 6` to `hops: 4` in both:
  - `experiments/seen_ac.yaml`
  - `experiments/unseen_ac.yaml`
- Remove pointer-list text from the main AC trace figures:
  - `assembly_heatmap.png`
  - `assembly_bars_over_time.png`
  - `assembly_connectivity_graph.png`
  - `assembly_weight_matrix.png`
- Add a new separate artifact:
  - `pointer_reference.png`
- Keep trace artifacts inside the canonical family trace directories:
  - `outputs/experiments/seen-ac/trace_plots/`
  - `outputs/experiments/unseen-ac/trace_plots/`

## Rationale

The current trace figures are crowded because the full pointer list appears in the suptitle of every main diagnostic plot. That makes stacked subplot figures look layered or glitched, especially for the seen AC traces. Moving pointer-list context into a dedicated artifact keeps the main figures focused on dynamics while still preserving the episode specification.

Using `hops: 4` also makes the canonical traces shorter and visually cleaner while still showing the sequential assembly progression clearly.

## Artifact Contract

### Main Trace Plots

These stay focused on dynamics and should not include the full pointer list in their suptitles:

- `assembly_heatmap.png`
- `assembly_bars_over_time.png`
- `assembly_connectivity_graph.png`
- `assembly_weight_matrix.png`

They may still include compact contextual text such as seen/unseen family, target node, prediction, or rollout summary if that does not crowd the layout.

### New Pointer Reference Plot

`pointer_reference.png` should be a compact standalone figure that shows:

- whether the trace comes from the seen or unseen family
- start node
- hops
- target node
- final prediction
- the pointer mapping/list itself in a readable compact form

This artifact is intentionally separate so the main trace plots remain uncluttered.

## Scientific Framing

The current unseen AC result remains very strong under the present structured episodic-memory setup. It is valid to describe it as strong evidence for the current time-as-compute story, but not yet as definitive proof of fully general unseen pointer-chasing. This cleanup does not change the experiment logic; it only improves plot readability and presentation.

## Non-Goals

- Do not remove `assembly_bars_over_time.png`
- Do not redesign the trace pipeline or move trace outputs out of the family directories
- Do not change the main seen/unseen result CSV schema
- Do not weaken or strengthen AC scientifically in this cleanup pass
- Do not add per-model trace sets; keep one canonical trace example per family
