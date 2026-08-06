# Temporal correction experiment

This experiment starts from the thesis MNIST baseline, freezes the sensory
weights, and teacher-forces only recurrent transitions from misclassified first
caps to their labelled assemblies. Each target neuron's original recurrent
weight budget is restored after every update.

The correction strength and reported readout are selected using a disjoint
validation subset. The plotted test subset is not used during selection.

Files:

- `mnist_temporal_correction_accuracy.{png,pdf,svg}`: mean test accuracy over
  network seeds, with 95% t intervals across seeds.
- `mnist_temporal_correction_time_series.csv`: per-seed test accuracy.
- `mnist_temporal_correction_trajectories.csv`: per-example test trajectories.
- `mnist_temporal_correction_validation.csv`: validation sweep used for
  selection.
- `mnist_temporal_correction_summary.json`: protocol, controls, selected
  configuration, and headline results.
