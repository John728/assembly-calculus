# AC Activation Bar Visualization Design

## Goal

Add a new AC trace visualization that shows per-neuron activation strength over time while keeping the existing `assembly_heatmap.png` unchanged.

The new plot should make the rollout order visually obvious by:

- placing neurons on the x-axis
- placing activation strength on the y-axis
- grouping neurons by assembly
- coloring neurons by assembly
- showing one panel per time step

## Why Add A New Plot

The current `assembly_heatmap.png` is already useful and should remain stable for continuity. The new request is not to replace that figure, but to add a second, more literal view of the model activity.

That second view should make it easy to see:

- which assembly is active at each internal step
- how activity moves from one assembly block to the next
- whether the rollout follows the intended pointer order

## Output

Add one new trace artifact alongside the current three files:

- `assembly_heatmap.png` (unchanged)
- `assembly_bars_over_time.png` (new)
- `assembly_connectivity_graph.png`
- `assembly_weight_matrix.png`

The new file should be produced by the same `render_trace_visualizations(...)` pipeline so suite-level trace generation picks it up automatically.

## Plot Structure

`assembly_bars_over_time.png` should use one subplot row per recorded time step.

Within each row:

- x-axis is neuron index
- y-axis is activation strength
- bars are grouped by `assembly_spans`
- each assembly block keeps a consistent color across all rows
- inactive neurons remain visible but muted
- assembly boundaries are explicit so block structure is easy to read

The bottom axis should keep assembly-centered tick labels, and the figure title should continue to include the pointer list plus `(target, pred)` summary.

## Activation Strength Semantics

The trace currently records `active_neurons` as a binary set per step, not analog neuron amplitudes. To stay honest to the data already captured, the first version of this new plot should render:

- `1.0` for active neurons
- `0.0` for inactive neurons

The plot is still valuable because the user’s main goal is to see grouped ordered firing over time. The implementation should be structured so that if later trace capture records real-valued strengths, the plotting path can accept them without redesign.

## Data Contract

The new plot should consume existing trace fields only:

- `steps`
- `assembly_spans`
- `pointer`
- `target_node`
- `final_prediction`

No runner or protocol changes should be required for the first version.

Optional future extension:

- allow each step to provide `neuron_strengths` directly; if present, the new plot can use those values instead of binary reconstruction from `active_neurons`

## Visual Design

The figure should remain paper-friendly and readable in static form.

Design choices:

- use bars rather than a dense raster so group structure is explicit
- keep the same assembly color across all panels
- use light background shading or separators to reinforce assembly grouping
- show active assembly names in each subplot title
- keep subplot spacing generous enough that time progression reads top-to-bottom

This plot should complement the heatmap, not compete with it.

## Testing

Existing visualization tests already verify the trace rendering pipeline writes expected files. Extend those tests so they now require the new artifact as well.

No image-content test is needed for the first pass; file existence is enough to lock in the interface.

## Non-Goals

This change should not:

- alter the current `assembly_heatmap.png`
- rename existing trace artifacts
- change trace recording semantics
- add animation, video, or per-step file explosions in the first version

## Summary

The correct change is an additive visualization: keep the current heatmap exactly as-is, and add a new grouped per-neuron activation-strength figure that makes the internal assembly sequence obvious over time.
