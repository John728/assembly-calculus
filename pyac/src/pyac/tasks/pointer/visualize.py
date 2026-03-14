from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def _assembly_colors(trace: dict[str, Any]) -> dict[str, tuple[float, float, float, float]]:
    spans = trace.get("assembly_spans", [])
    labels = [str(span["label"]) for span in spans]
    if not labels:
        labels = [str(label) for label in trace["assembly_weight_matrix"]["labels"]]
    palette = cm.get_cmap("tab20", max(len(labels), 1))
    return {label: palette(idx) for idx, label in enumerate(labels)}


def _span_lookup(trace: dict[str, Any]) -> dict[str, tuple[int, int]]:
    lookup: dict[str, tuple[int, int]] = {}
    for span in trace.get("assembly_spans", []):
        lookup[str(span["label"])] = (int(span["start"]), int(span["end"]))
    return lookup


def _save_assembly_heatmap(trace: dict[str, Any], output_dir: Path) -> Path:
    steps = trace["steps"]
    spans = trace.get("assembly_spans", [])
    colors = _assembly_colors(trace)
    max_neuron = max(int(span["end"]) for span in spans) if spans else max((max(step["active_neurons"]) if step["active_neurons"] else -1) for step in steps)
    neuron_indices = np.arange(max_neuron + 1, dtype=np.int64)

    fig, axes = plt.subplots(len(steps), 1, figsize=(18, max(2.6 * len(steps), 5.5)), sharex=True)
    if len(steps) == 1:
        axes = [axes]

    for axis, step in zip(axes, steps):
        strengths = np.zeros(max_neuron + 1, dtype=np.float64)
        active_set = {int(neuron) for neuron in step["active_neurons"]}
        bar_colors: list[tuple[float, float, float, float] | str] = ["#d9d9d9"] * (max_neuron + 1)

        for span in spans:
            label = str(span["label"])
            start = int(span["start"])
            end = int(span["end"])
            axis.axvspan(start - 0.5, end + 0.5, color=colors[label], alpha=0.08)
            axis.axvline(end + 0.5, color="#9e9e9e", linewidth=0.6, alpha=0.7)
            for neuron in range(start, end + 1):
                bar_colors[neuron] = colors[label]

        for neuron in active_set:
            strengths[neuron] = 1.0

        axis.bar(neuron_indices, strengths, color=bar_colors, width=0.95, edgecolor="none")
        active_labels = ", ".join(step["active_assemblies"]) if step["active_assemblies"] else "none"
        axis.set_ylim(0.0, 1.1)
        axis.set_ylabel(f"t={step['time']}\nstrength")
        axis.set_title(f"Active assemblies: {active_labels}", fontsize=10, loc="left")
        axis.grid(axis="y", alpha=0.2)

    if spans:
        tick_positions = [(int(span["start"]) + int(span["end"])) / 2.0 for span in spans]
        tick_labels = [str(span["label"]) for span in spans]
        axes[-1].set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Neuron index grouped by assembly")
    fig.suptitle(
        f"Assembly firing over time (target={trace['target_node']}, pred={trace['final_prediction']})",
        fontsize=14,
        y=0.995,
    )
    path = output_dir / "assembly_heatmap.png"
    fig.subplots_adjust(top=0.94, bottom=0.16, hspace=0.45)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _save_assembly_bars_over_time(trace: dict[str, Any], output_dir: Path) -> Path:
    steps = trace["steps"]
    spans = trace.get("assembly_spans", [])
    colors = _assembly_colors(trace)
    max_from_spans = max((int(span["end"]) for span in spans), default=-1)
    max_from_active = max((max((int(neuron) for neuron in step["active_neurons"]), default=-1) for step in steps), default=-1)
    max_from_strengths = max(
        (
            max((int(neuron) for neuron in step.get("neuron_strengths", {}).keys()), default=-1)
            for step in steps
        ),
        default=-1,
    )
    max_neuron = max(max_from_spans, max_from_active, max_from_strengths)
    if max_neuron < 0:
        max_neuron = 0
    neuron_indices = np.arange(max_neuron + 1, dtype=np.int64)

    fig, axes = plt.subplots(len(steps), 1, figsize=(18, max(2.9 * len(steps), 6.0)), sharex=True)
    if len(steps) == 1:
        axes = [axes]

    for axis, step in zip(axes, steps):
        strengths = np.zeros(max_neuron + 1, dtype=np.float64)
        bar_colors: list[tuple[float, float, float, float] | str] = ["#d9d9d9"] * (max_neuron + 1)
        step_strengths = step.get("neuron_strengths")
        if isinstance(step_strengths, dict):
            for neuron, strength in step_strengths.items():
                neuron_idx = int(neuron)
                if 0 <= neuron_idx <= max_neuron:
                    strengths[neuron_idx] = float(strength)
        else:
            for neuron in step["active_neurons"]:
                strengths[int(neuron)] = 1.0

        for span in spans:
            label = str(span["label"])
            start = int(span["start"])
            end = int(span["end"])
            axis.axvspan(start - 0.5, end + 0.5, color=colors[label], alpha=0.08)
            axis.axvline(end + 0.5, color="#7f7f7f", linewidth=0.8, alpha=0.9)
            for neuron in range(start, min(end + 1, max_neuron + 1)):
                bar_colors[neuron] = colors[label]

        if not spans and step["active_assemblies"]:
            fallback_color = colors.get(step["active_assemblies"][0], "#4c78a8")
            bar_colors = [fallback_color] * (max_neuron + 1)

        axis.bar(
            neuron_indices,
            strengths,
            color=bar_colors,
            width=0.9,
            edgecolor="white",
            linewidth=0.2,
        )

        active_labels = ", ".join(step["active_assemblies"]) if step["active_assemblies"] else "none"
        axis.set_ylim(0.0, max(1.05, float(np.max(strengths)) + 0.05))
        axis.set_ylabel(f"t={step['time']}\nstrength")
        axis.set_title(f"Active assemblies: {active_labels}", fontsize=10, loc="left")
        axis.grid(axis="y", alpha=0.2)

    if spans:
        tick_positions = [(int(span["start"]) + int(span["end"])) / 2.0 for span in spans]
        tick_labels = [str(span["label"]) for span in spans]
        axes[-1].set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Neuron index grouped by assembly")
    fig.suptitle(
        f"Assembly-grouped neuron activations over time (target={trace['target_node']}, pred={trace['final_prediction']})",
        fontsize=14,
        y=0.995,
    )
    path = output_dir / "assembly_bars_over_time.png"
    fig.subplots_adjust(top=0.94, bottom=0.16, hspace=0.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _save_connectivity_graph(trace: dict[str, Any], output_dir: Path) -> Path:
    colors = _assembly_colors(trace)
    labels = [str(span["label"]) for span in trace.get("assembly_spans", []) if int(span.get("list_idx", -1)) == int(trace["list_idx"])]
    if not labels:
        labels = [str(label) for label in trace["assembly_weight_matrix"]["labels"]]
    label_to_row = {label: idx for idx, label in enumerate(labels)}
    rollout_labels = [label for label in trace.get("rollout_path_labels", []) if label in label_to_row]
    rollout_matrix = np.zeros((len(labels), max(len(rollout_labels), 1)), dtype=np.float64)
    for col_idx, label in enumerate(rollout_labels):
        rollout_matrix[label_to_row[label], col_idx] = 1.0

    expected_text = "\n".join(f"{edge['src']} -> {edge['dst']}" for edge in trace.get("expected_edges", []))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"width_ratios": [1.3, 1.0]})
    axes[0].imshow(rollout_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("Rollout trajectory by assembly")
    axes[0].set_xlabel("Internal time")
    axes[0].set_ylabel("Assembly")
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xticks(range(max(len(rollout_labels), 1)), [str(idx) for idx in range(max(len(rollout_labels), 1))])

    axes[1].axis("off")
    axes[1].set_title("Expected pointer transitions")
    axes[1].text(
        0.02,
        0.98,
        expected_text if expected_text else "No expected transitions recorded",
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    fig.suptitle("Internal rollout vs expected transitions", fontsize=14)
    path = output_dir / "assembly_connectivity_graph.png"
    fig.subplots_adjust(top=0.84, bottom=0.14, wspace=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _save_weight_matrix(trace: dict[str, Any], output_dir: Path) -> Path:
    labels = list(trace["assembly_weight_matrix"]["labels"])
    weights = np.asarray(trace["assembly_weight_matrix"]["values"], dtype=np.float64)
    expected = np.zeros_like(weights)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    for edge in trace.get("expected_edges", []):
        src = str(edge["src"])
        dst = str(edge["dst"])
        if src in label_to_index and dst in label_to_index:
            expected[label_to_index[src], label_to_index[dst]] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), sharey=True)
    axes[0].imshow(expected, cmap="Greys", aspect="auto", vmin=0.0, vmax=1.0)
    axes[0].set_title("Expected adjacency")
    axes[1].imshow(weights, cmap="viridis", aspect="auto")
    axes[1].set_title("Learned weight mass")
    for axis in axes:
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Destination assembly")
    axes[0].set_ylabel("Source assembly")
    fig.suptitle("Expected vs learned assembly transitions", fontsize=14)
    fig.subplots_adjust(top=0.86, bottom=0.24, wspace=0.12)
    path = output_dir / "assembly_weight_matrix.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _save_pointer_reference(trace: dict[str, Any], output_dir: Path) -> Path:
    pointer = [int(value) for value in trace.get("pointer", [])]
    path_labels = trace.get("rollout_path_labels", [])
    list_type = "Unseen" if any(str(label).startswith("N") for label in path_labels) else "Seen"
    pointer_mapping = "\n".join(
        f"{idx} -> {value}" for idx, value in enumerate(pointer)
    ) or "No pointer mapping recorded"

    fig, axis = plt.subplots(figsize=(8.5, max(4.5, 1.4 + 0.28 * max(len(pointer), 1))))
    axis.axis("off")
    summary = "\n".join(
        [
            f"family: {list_type}",
            f"list_idx: {trace['list_idx']}",
            f"start_node: {trace.get('start_node', 'n/a')}",
            f"hops: {trace.get('hops', 'n/a')}",
            f"target: {trace['target_node']}",
            f"prediction: {trace['final_prediction']}",
        ]
    )
    axis.text(0.02, 0.98, summary, va="top", ha="left", family="monospace", fontsize=11)
    axis.text(
        0.42,
        0.98,
        "pointer mapping\n" + pointer_mapping,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )
    fig.suptitle("Pointer reference", fontsize=14)
    fig.subplots_adjust(top=0.86, left=0.05, right=0.97, bottom=0.08)
    path = output_dir / "pointer_reference.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def render_trace_visualizations(trace: dict[str, Any], output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return [
        _save_assembly_heatmap(trace, output_path),
        _save_assembly_bars_over_time(trace, output_path),
        _save_connectivity_graph(trace, output_path),
        _save_weight_matrix(trace, output_path),
        _save_pointer_reference(trace, output_path),
    ]
