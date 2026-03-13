from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYAC_SRC = ROOT / "pyac" / "src"
if str(PYAC_SRC) not in sys.path:
    sys.path.insert(0, str(PYAC_SRC))


def test_pointer_visualizations_write_expected_files(tmp_path: Path) -> None:
    from pyac.tasks.pointer.visualize import render_trace_visualizations

    trace = {
        "list_idx": 0,
        "start_node": 0,
        "hops": 3,
        "target_node": 3,
        "final_prediction": 3,
        "pointer": [2, 1, 4, 0, 3],
        "rollout_path_labels": ["L0:N0", "L0:N2", "L0:N4", "L0:N3"],
        "expected_edges": [
            {"src": "L0:N0", "dst": "L0:N2"},
            {"src": "L0:N2", "dst": "L0:N4"},
            {"src": "L0:N4", "dst": "L0:N3"},
            {"src": "L0:N3", "dst": "L0:N0"},
        ],
        "assembly_spans": [
            {"label": "L0:N0", "list_idx": 0, "node_idx": 0, "start": 0, "end": 2},
            {"label": "L0:N2", "list_idx": 0, "node_idx": 2, "start": 3, "end": 5},
            {"label": "L0:N4", "list_idx": 0, "node_idx": 4, "start": 6, "end": 8},
            {"label": "L0:N3", "list_idx": 0, "node_idx": 3, "start": 9, "end": 11},
        ],
        "steps": [
            {"time": 0, "active_neurons": [0, 1, 2], "active_assemblies": ["L0:N0"]},
            {"time": 1, "active_neurons": [3, 4, 5], "active_assemblies": ["L0:N2"]},
            {"time": 2, "active_neurons": [6, 7, 8], "active_assemblies": ["L0:N4"]},
            {"time": 3, "active_neurons": [9, 10, 11], "active_assemblies": ["L0:N3"]},
        ],
        "assembly_weight_matrix": {
            "labels": ["L0:N0", "L0:N2", "L0:N4", "L0:N3"],
            "values": [
                [0.0, 1.0, 0.1, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
                [0.1, 0.0, 0.0, 0.0],
            ],
        },
    }

    paths = render_trace_visualizations(trace, tmp_path)

    assert {path.name for path in paths} == {
        "assembly_bars_over_time.png",
        "assembly_heatmap.png",
        "assembly_connectivity_graph.png",
        "assembly_weight_matrix.png",
    }
    assert all(path.exists() for path in paths)


def test_pointer_visualizations_support_neuron_strengths_without_active_neurons(tmp_path: Path) -> None:
    from pyac.tasks.pointer.visualize import render_trace_visualizations

    trace = {
        "list_idx": 0,
        "start_node": 0,
        "hops": 1,
        "target_node": 1,
        "final_prediction": 1,
        "pointer": [1, 0],
        "rollout_path_labels": ["L0:N0", "L0:N1"],
        "expected_edges": [{"src": "L0:N0", "dst": "L0:N1"}],
        "assembly_spans": [],
        "steps": [
            {"time": 0, "active_neurons": [], "active_assemblies": ["L0:N0"], "neuron_strengths": {0: 0.8, 1: 0.3}},
            {"time": 1, "active_neurons": [], "active_assemblies": ["L0:N1"], "neuron_strengths": {2: 1.0, 3: 0.4}},
        ],
        "assembly_weight_matrix": {
            "labels": ["L0:N0", "L0:N1"],
            "values": [[0.0, 1.0], [0.0, 0.0]],
        },
    }

    paths = render_trace_visualizations(trace, tmp_path)

    assert (tmp_path / "assembly_bars_over_time.png") in paths
    assert (tmp_path / "assembly_bars_over_time.png").exists()
